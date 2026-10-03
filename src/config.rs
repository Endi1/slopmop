use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub project_directory: Option<String>,
    pub cluster: ClusterConfig,
    pub languages: HashMap<String, LanguageConfig>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct ClusterConfig {
    pub threshold: Option<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct LanguageConfig {
    pub enabled: bool,
    #[serde(alias = "excluded_constructs", alias = "exclude_constructs")]
    pub exclude: Vec<String>,
}

impl Default for LanguageConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            exclude: Vec::new(),
        }
    }
}

pub fn load(path: &Path) -> Result<Config> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read config file {}", path.display()))?;
    toml::from_str(&contents)
        .with_context(|| format!("failed to parse config file {}", path.display()))
}

pub fn discover(directory: &Path, explicit_path: Option<&Path>) -> Result<Config> {
    match explicit_path {
        Some(path) => load(path),
        None => {
            let path = directory.join("slopmop.toml");
            if path.is_file() {
                load(&path)
            } else {
                Ok(Config::default())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_language_level_settings() -> Result<()> {
        let config: Config = toml::from_str(
            "[languages.go]\nenabled = false\nexclude = [\"struct\", \"func\"]\n[languages.rust]\nenabled = true\n",
        )?;

        assert!(!config.languages["go"].enabled);
        assert_eq!(config.languages["go"].exclude, ["struct", "func"]);
        assert!(config.languages["rust"].enabled);
        Ok(())
    }
}

pub fn config_path(arguments: &[String]) -> Result<(Vec<String>, Option<PathBuf>)> {
    let mut remaining = Vec::with_capacity(arguments.len());
    let mut path = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--config" => {
                index += 1;
                path = Some(PathBuf::from(
                    arguments.get(index).context("--config requires a path")?,
                ));
            }
            argument if argument.starts_with("--config=") => {
                path = Some(PathBuf::from(&argument["--config=".len()..]));
            }
            _ => remaining.push(arguments[index].clone()),
        }
        index += 1;
    }
    Ok((remaining, path))
}
