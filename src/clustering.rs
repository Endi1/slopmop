use std::{collections::HashMap, path::Path};

use anyhow::{Context, Result, ensure};

use crate::{database::Database, sqlite_vector};

pub fn list_largest_clusters(project_root: &Path, minimum_similarity: f32) -> Result<()> {
    ensure!(
        (-1.0..=1.0).contains(&minimum_similarity),
        "similarity threshold must be between -1 and 1"
    );

    let database_path = project_root.join(".slopmop");
    ensure!(
        database_path.is_file(),
        "{} does not exist; index the project first",
        database_path.display()
    );

    let extension_path = sqlite_vector::extension_path()?;
    let database = Database::open(&database_path, &extension_path)?;
    let embeddings = database.embeddings()?;
    if embeddings.is_empty() {
        println!("The index contains no embeddings.");
        return Ok(());
    }

    let indices = embeddings
        .iter()
        .enumerate()
        .map(|(index, embedding)| (embedding.id, index))
        .collect::<HashMap<_, _>>();
    let mut groups = UnionFind::new(embeddings.len());

    for (index, embedding) in embeddings.iter().enumerate() {
        let neighbors = database
            .similar_embedding_ids(&embedding.embedding, minimum_similarity)
            .with_context(|| format!("failed to find neighbors for {}", embedding.node_name))?;
        for id in neighbors {
            if let Some(&neighbor) = indices.get(&id)
                && neighbor > index
            {
                groups.union(index, neighbor);
            }
        }
    }

    let mut clusters = HashMap::<usize, Vec<usize>>::new();
    for index in 0..embeddings.len() {
        clusters.entry(groups.find(index)).or_default().push(index);
    }
    let mut clusters = clusters.into_values().collect::<Vec<_>>();
    clusters.sort_by(|left, right| {
        right
            .len()
            .cmp(&left.len())
            .then_with(|| left[0].cmp(&right[0]))
    });

    println!(
        "{} clusters at cosine similarity >= {:.3}; showing the 10 largest\n",
        clusters.len(),
        minimum_similarity
    );
    for (rank, cluster) in clusters.iter().take(10).enumerate() {
        println!("Cluster {} ({} nodes)", rank + 1, cluster.len());
        for &index in cluster {
            let item = &embeddings[index];
            println!("  {}:{}", item.filepath, item.node_name);
        }
        println!();
    }

    Ok(())
}

struct UnionFind {
    parents: Vec<usize>,
    sizes: Vec<usize>,
}

impl UnionFind {
    fn new(length: usize) -> Self {
        Self {
            parents: (0..length).collect(),
            sizes: vec![1; length],
        }
    }

    fn find(&mut self, item: usize) -> usize {
        if self.parents[item] != item {
            self.parents[item] = self.find(self.parents[item]);
        }
        self.parents[item]
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        if self.sizes[left_root] < self.sizes[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parents[right_root] = left_root;
        self.sizes[left_root] += self.sizes[right_root];
    }
}
