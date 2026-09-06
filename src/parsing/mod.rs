mod filesystem;
mod go;

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use ignore::gitignore::{Gitignore, GitignoreBuilder};

pub(crate) struct CodeEntity {
    pub(crate) name: String,
    pub(crate) content: String,
}

pub(crate) struct PendingFile {
    pub(crate) filepath: String,
    pub(crate) entities: Vec<CodeEntity>,
}

#[derive(Debug, Eq, PartialEq)]
enum Language {
    Go,
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

fn language_for_path(path: &Path) -> Option<Language> {
    if go::supports(path) {
        Some(Language::Go)
    } else {
        None
    }
}

fn select_source_files(paths: Vec<PathBuf>) -> Vec<(PathBuf, Language)> {
    paths
        .into_iter()
        .filter_map(|path| language_for_path(&path).map(|language| (path, language)))
        .collect()
}

pub(crate) fn parse_project(project_root: &Path) -> Result<Vec<PendingFile>> {
    let ignore_rules = load_ignore_rules(project_root)?;
    let discovered_files = filesystem::discover_files(project_root, &ignore_rules)?;
    let mut source_files = select_source_files(discovered_files);
    source_files.sort_by(|(left, _), (right, _)| left.cmp(right));

    let mut go_parser = go::parser()?;
    source_files
        .into_iter()
        .map(|(path, language)| {
            let source = fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            let entities = match language {
                Language::Go => go::parse(&mut go_parser, &source)?,
            };

            Ok(PendingFile {
                filepath: filesystem::relative_path(&path, project_root),
                entities,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_root(name: &str) -> Result<PathBuf> {
        let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root =
            std::env::temp_dir().join(format!("slopmop-{name}-{}-{unique}", std::process::id()));
        fs::create_dir_all(&root)?;
        Ok(root)
    }

    #[test]
    fn selects_only_supported_source_files() {
        let paths = vec![
            PathBuf::from("main.go"),
            PathBuf::from("README.md"),
            PathBuf::from("src/main.rs"),
        ];

        assert_eq!(
            select_source_files(paths),
            [(PathBuf::from("main.go"), Language::Go)]
        );
    }

    #[test]
    fn missing_slopmopignore_is_allowed() -> Result<()> {
        let root = temporary_root("no-ignore")?;
        fs::write(root.join("keep.go"), "package example\n")?;

        let files = parse_project(&root)?;

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].filepath, "keep.go");
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn slopmopignore_excludes_matching_files_and_directories() -> Result<()> {
        let root = temporary_root("ignore")?;
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

        let files = parse_project(&root)?;
        let relative = files
            .iter()
            .map(|file| file.filepath.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            relative,
            ["keep.go", "nested/important_mock.go", "nested/root_only.go"]
        );
        fs::remove_dir_all(root)?;
        Ok(())
    }
}
