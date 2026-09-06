mod clustering;
mod database;
mod jina;
mod jina_model;
mod parsing;
mod sqlite_vector;

use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use database::Database;
use jina::JinaEmbedder;

struct IndexedEmbedding {
    node_name: String,
    embedding: Vec<f32>,
}

struct IndexedFile {
    filepath: String,
    embeddings: Vec<IndexedEmbedding>,
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
    let pending_files = parsing::parse_project(project_root)?;
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
