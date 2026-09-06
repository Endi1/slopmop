mod clustering;
mod database;
mod jina;
mod jina_model;
mod sqlite_vector;

use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use database::Database;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use jina::JinaEmbedder;
use tree_sitter::{Node, Parser};

struct CodeEntity {
    name: String,
    content: String,
}

struct PendingFile {
    filepath: String,
    entities: Vec<CodeEntity>,
}

struct IndexedEmbedding {
    node_name: String,
    embedding: Vec<f32>,
}

struct IndexedFile {
    filepath: String,
    embeddings: Vec<IndexedEmbedding>,
}

fn collect_entities(node: Node<'_>, source: &str, entities: &mut Vec<CodeEntity>) {
    let is_indexed_node = match node.kind() {
        "function_declaration" | "method_declaration" => true,
        "type_spec" => node
            .child_by_field_name("type")
            .is_some_and(|node| matches!(node.kind(), "struct_type" | "interface_type")),
        _ => false,
    };

    if is_indexed_node {
        let name = node
            .child_by_field_name("name")
            .and_then(|name| name.utf8_text(source.as_bytes()).ok())
            .unwrap_or("<unknown>");
        let content = node
            .utf8_text(source.as_bytes())
            .expect("Code entity content was not valid UTF-8");

        entities.push(CodeEntity {
            name: name.to_owned(),
            content: content.to_owned(),
        });
        return;
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_entities(child, source, entities);
    }
}

fn load_ignore_rules(project_root: &Path) -> Result<Gitignore> {
    let ignore_path = project_root.join(".slopmopignore");
    let mut builder = GitignoreBuilder::new(project_root);

    if ignore_path
        .try_exists()
        .with_context(|| format!("failed to inspect {}", ignore_path.display()))?
        && let Some(error) = builder.add(&ignore_path)
    {
        return Err(error).with_context(|| format!("failed to parse {}", ignore_path.display()));
    }

    builder.build().with_context(|| {
        format!(
            "failed to build ignore rules from {}",
            ignore_path.display()
        )
    })
}

fn find_go_files(
    directory: &Path,
    ignore_rules: &Gitignore,
    files: &mut Vec<PathBuf>,
) -> Result<()> {
    for entry in fs::read_dir(directory)
        .with_context(|| format!("failed to read directory {}", directory.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;

        if entry.file_name() == ".git"
            || ignore_rules.matched(&path, file_type.is_dir()).is_ignore()
        {
            continue;
        }

        if file_type.is_dir() {
            find_go_files(&path, ignore_rules, files)?;
        } else if file_type.is_file() && path.extension().is_some_and(|extension| extension == "go")
        {
            files.push(path);
        }
    }
    Ok(())
}

fn relative_path(path: &Path, project_root: &Path) -> String {
    path.strip_prefix(project_root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

fn parse_project(project_root: &Path) -> Result<Vec<PendingFile>> {
    let ignore_rules = load_ignore_rules(project_root)?;
    let mut paths = Vec::new();
    find_go_files(project_root, &ignore_rules, &mut paths)?;
    paths.sort();

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_go::LANGUAGE.into())
        .context("failed to load Go grammar")?;

    paths
        .into_iter()
        .map(|path| {
            let source = fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            let tree = parser
                .parse(&source, None)
                .context("parser failed to produce a syntax tree")?;
            let mut entities = Vec::new();
            collect_entities(tree.root_node(), &source, &mut entities);

            Ok(PendingFile {
                filepath: relative_path(&path, project_root),
                entities,
            })
        })
        .collect()
}

fn project_root(directory: &str) -> Result<PathBuf> {
    let root = fs::canonicalize(directory)
        .with_context(|| format!("failed to open project directory {directory}"))?;
    if !root.is_dir() {
        bail!("{} is not a directory", root.display());
    }
    Ok(root)
}

fn index_project(project_root: &Path) -> Result<()> {
    let extension_path = sqlite_vector::extension_path()?;
    let pending_files = parse_project(project_root)?;
    let entity_count = pending_files
        .iter()
        .map(|file| file.entities.len())
        .sum::<usize>();

    let embedder = if entity_count > 0 {
        eprintln!("Loading jinaai/jina-embeddings-v2-base-code...");
        Some(JinaEmbedder::load()?)
    } else {
        None
    };

    let mut indexed_files = Vec::with_capacity(pending_files.len());
    for file in pending_files {
        let mut embeddings = Vec::with_capacity(file.entities.len());
        for entity in file.entities {
            eprintln!("Embedding {} in {}", entity.name, file.filepath);
            embeddings.push(IndexedEmbedding {
                node_name: entity.name,
                embedding: embedder
                    .as_ref()
                    .expect("embedder exists when entities exist")
                    .embed(&entity.content)?,
            });
        }
        indexed_files.push(IndexedFile {
            filepath: file.filepath,
            embeddings,
        });
    }

    let database_path = project_root.join(".slopmop");
    let mut database = Database::open(&database_path, &extension_path)?;
    database.replace_project(&indexed_files)?;

    println!(
        "Indexed {} nodes from {} Go files into {}",
        entity_count,
        indexed_files.len(),
        database_path.display()
    );
    Ok(())
}

fn cluster_command(arguments: &[String]) -> Result<()> {
    let mut directory = ".";
    let mut threshold = 0.8_f32;
    let mut index = 0;

    while index < arguments.len() {
        match arguments[index].as_str() {
            "--threshold" => {
                index += 1;
                threshold = arguments
                    .get(index)
                    .context("--threshold requires a value")?
                    .parse()
                    .context("invalid cosine similarity threshold")?;
            }
            argument if argument.starts_with("--threshold=") => {
                threshold = argument["--threshold=".len()..]
                    .parse()
                    .context("invalid cosine similarity threshold")?;
            }
            argument if argument.starts_with('-') => bail!("unknown option: {argument}"),
            argument => directory = argument,
        }
        index += 1;
    }

    clustering::list_largest_clusters(&project_root(directory)?, threshold)
}

fn print_usage() {
    println!(
        "Usage:\n  slopmop index [PROJECT_DIRECTORY]\n  slopmop cluster [PROJECT_DIRECTORY] [--threshold 0.8]"
    );
}

fn main() -> Result<()> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    match arguments.first().map(String::as_str) {
        Some("cluster") => cluster_command(&arguments[1..]),
        Some("index") => {
            let directory = arguments.get(1).map_or(".", String::as_str);
            index_project(&project_root(directory)?)
        }
        Some("--help" | "-h") => {
            print_usage();
            Ok(())
        }
        Some(directory) => index_project(&project_root(directory)?),
        None => index_project(&project_root(".")?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn missing_slopmopignore_is_allowed() -> Result<()> {
        let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root =
            env::temp_dir().join(format!("slopmop-no-ignore-{}-{unique}", std::process::id()));
        fs::create_dir_all(&root)?;
        fs::write(root.join("keep.go"), "package example\n")?;

        let rules = load_ignore_rules(&root)?;
        let mut files = Vec::new();
        find_go_files(&root, &rules, &mut files)?;

        assert_eq!(files, [root.join("keep.go")]);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn slopmopignore_excludes_matching_files_and_directories() -> Result<()> {
        let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = env::temp_dir().join(format!("slopmop-ignore-{}-{unique}", std::process::id()));
        fs::create_dir_all(root.join("generated"))?;
        fs::create_dir_all(root.join("nested"))?;

        fs::write(
            root.join(".slopmopignore"),
            "ignored.go\ngenerated/\n*_mock.go\n!important_mock.go\n/root_only.go\n",
        )?;
        for path in [
            "keep.go",
            "ignored.go",
            "generated/code.go",
            "nested/thing_mock.go",
            "nested/important_mock.go",
            "root_only.go",
            "nested/root_only.go",
        ] {
            fs::write(root.join(path), "package example\n")?;
        }

        let rules = load_ignore_rules(&root)?;
        let mut files = Vec::new();
        find_go_files(&root, &rules, &mut files)?;
        let mut relative = files
            .iter()
            .map(|path| relative_path(path, &root))
            .collect::<Vec<_>>();
        relative.sort();

        assert_eq!(
            relative,
            ["keep.go", "nested/important_mock.go", "nested/root_only.go"]
        );
        fs::remove_dir_all(root)?;
        Ok(())
    }
}
