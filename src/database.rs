use std::path::Path;

use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, params};

use crate::IndexedFile;

const EMBEDDING_DIMENSIONS: usize = 768;

pub struct Database {
    connection: Connection,
}

pub struct StoredEmbedding {
    pub id: i64,
    pub embedding: Vec<u8>,
    pub node_name: String,
    pub filepath: String,
}

impl Database {
    pub fn open(database_path: &Path, extension_path: &Path) -> Result<Self> {
        let connection = Connection::open(database_path)
            .with_context(|| format!("failed to open {}", database_path.display()))?;

        unsafe {
            connection
                .load_extension(extension_path, None::<&str>)
                .with_context(|| {
                    format!(
                        "failed to load SQLite-Vector extension from {}",
                        extension_path.display()
                    )
                })?;
        }

        connection.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA journal_mode = WAL;

             CREATE TABLE IF NOT EXISTS files (
                 id       INTEGER PRIMARY KEY,
                 filepath TEXT NOT NULL UNIQUE
             );

             CREATE TABLE IF NOT EXISTS embeddings (
                 id        INTEGER PRIMARY KEY,
                 embedding BLOB NOT NULL,
                 node_name TEXT NOT NULL,
                 file_id   INTEGER NOT NULL,
                 FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
             );

             CREATE INDEX IF NOT EXISTS embeddings_file_id
                 ON embeddings(file_id);",
        )?;
        connection.query_row(
            "SELECT vector_init(
                'embeddings',
                'embedding',
                'type=FLOAT32,dimension=768,distance=COSINE,normalized=1'
             )",
            [],
            |_| Ok(()),
        )?;

        Ok(Self { connection })
    }

    pub fn embeddings(&self) -> Result<Vec<StoredEmbedding>> {
        let mut statement = self.connection.prepare(
            "SELECT embeddings.id, embeddings.embedding, embeddings.node_name, files.filepath
             FROM embeddings
             JOIN files ON files.id = embeddings.file_id
             ORDER BY embeddings.id",
        )?;
        let rows = statement.query_map([], |row| {
            Ok(StoredEmbedding {
                id: row.get(0)?,
                embedding: row.get(1)?,
                node_name: row.get(2)?,
                filepath: row.get(3)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn similar_embedding_ids(
        &self,
        embedding: &[u8],
        minimum_similarity: f32,
    ) -> Result<Vec<i64>> {
        let maximum_distance = 1.0 - minimum_similarity;
        let mut statement = self.connection.prepare(
            "SELECT rowid
             FROM vector_full_scan('embeddings', 'embedding', ?1)
             WHERE distance <= ?2",
        )?;
        let rows = statement.query_map(params![embedding, maximum_distance], |row| row.get(0))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn replace_project(&mut self, files: &[IndexedFile]) -> Result<()> {
        let transaction = self.connection.transaction()?;
        transaction.execute("DELETE FROM embeddings", [])?;
        transaction.execute("DELETE FROM files", [])?;

        for file in files {
            transaction.execute("INSERT INTO files (filepath) VALUES (?1)", [&file.filepath])?;
            let file_id = transaction.last_insert_rowid();

            let mut statement = transaction.prepare(
                "INSERT INTO embeddings (embedding, node_name, file_id)
                 VALUES (vector_as_f32(?1), ?2, ?3)",
            )?;

            for item in &file.embeddings {
                ensure!(
                    item.embedding.len() == EMBEDDING_DIMENSIONS,
                    "embedding for {} has {} dimensions, expected {}",
                    item.node_name,
                    item.embedding.len(),
                    EMBEDDING_DIMENSIONS
                );
                let embedding_blob = item
                    .embedding
                    .iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect::<Vec<_>>();
                statement.execute(params![embedding_blob, item.node_name, file_id])?;
            }
        }

        transaction.commit()?;
        Ok(())
    }
}
