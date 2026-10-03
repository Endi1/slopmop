use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub project_directory: Option<String>,
    pub threshold: Option<f32>,
    pub cluster: ClusterConfig,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct ClusterConfig {
    pub threshold: Option<f32>,
}

impl Config {
    pub fn threshold(&self) -> Option<f32> {
        self.cluster.threshold.or(self.threshold)
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
