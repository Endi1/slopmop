mod clustering;
mod config;
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

fn index_project(project_root: &Path, config: &config::Config) -> Result<()> {
    let extension_path = sqlite_vector::extension_path()?;
    let pending_files = parsing::parse_project(project_root, config)?;
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
    let (arguments, config_path) = config::config_path(arguments)?;
    let mut directory = None;
    let mut threshold = None;
    let mut index = 0;

    while index < arguments.len() {
        match arguments[index].as_str() {
            "--threshold" => {
                index += 1;
                threshold = Some(
                    arguments
                        .get(index)
                        .context("--threshold requires a value")?
                        .parse()
                        .context("invalid cosine similarity threshold")?,
                );
            }
            argument if argument.starts_with("--threshold=") => {
                threshold = Some(
                    argument["--threshold=".len()..]
                        .parse()
                        .context("invalid cosine similarity threshold")?,
                );
            }
            argument if argument.starts_with('-') => bail!("unknown option: {argument}"),
            argument => directory = Some(argument),
        }
        index += 1;
    }

    let supplied_directory = directory;
    let config = config::discover(
        Path::new(supplied_directory.unwrap_or(".")),
        config_path.as_deref(),
    )?;
    let directory = supplied_directory
        .or(config.project_directory.as_deref())
        .unwrap_or(".");
    let threshold = threshold.or(config.cluster.threshold).unwrap_or(0.8);
    clustering::list_clusters(&project_root(directory)?, threshold)
}

fn index_command(arguments: &[String]) -> Result<()> {
    let (arguments, config_path) = config::config_path(arguments)?;
    let directory = arguments.first().map(String::as_str);
    if arguments.len() > 1 {
        bail!("only one project directory may be supplied");
    }
    let config = config::discover(Path::new(directory.unwrap_or(".")), config_path.as_deref())?;
    let directory = directory
        .or(config.project_directory.as_deref())
        .unwrap_or(".");
    index_project(&project_root(directory)?, &config)
}

fn print_usage() {
    println!(
        "Usage:\n  slopmop index [PROJECT_DIRECTORY] [--config PATH]\n  slopmop cluster [PROJECT_DIRECTORY] [--threshold 0.8] [--config PATH]\n\nConfig: slopmop.toml (automatically discovered in the project directory)"
    );
}

fn main() -> Result<()> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    match arguments.first().map(String::as_str) {
        Some("cluster") => cluster_command(&arguments[1..]),
        Some("index") => index_command(&arguments[1..]),
        Some("--help" | "-h") => {
            print_usage();
            Ok(())
        }
        Some(_) | None => index_command(&arguments),
    }
}
