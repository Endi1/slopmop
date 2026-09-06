use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use ignore::gitignore::Gitignore;

pub(super) fn discover_files(directory: &Path, ignore_rules: &Gitignore) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

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
            files.extend(discover_files(&path, ignore_rules)?);
        } else if file_type.is_file() {
            files.push(path);
        }
    }

    Ok(files)
}

pub(super) fn relative_path(path: &Path, project_root: &Path) -> String {
    path.strip_prefix(project_root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}
