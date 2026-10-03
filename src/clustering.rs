use std::{collections::HashMap, path::Path};

use anyhow::{Context, Result, ensure};

use crate::{database::Database, sqlite_vector};

pub fn list_clusters(project_root: &Path, minimum_similarity: f32) -> Result<()> {
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
    let mut clusters = clusters
        .into_values()
        .filter(|members| members.len() > 1)
        .map(|members| {
            let score = average_similarity(
                members
                    .iter()
                    .map(|&index| embeddings[index].embedding.as_slice()),
            );
            (members, score)
        })
        .collect::<Vec<_>>();
    rank_clusters(&mut clusters);

    println!(
        "{} clusters at cosine similarity >= {:.3}; ranked by average pairwise similarity\n",
        clusters.len(),
        minimum_similarity
    );
    for (rank, (cluster, score)) in clusters.iter().enumerate() {
        let similarity = score.map_or_else(|| "N/A".to_owned(), |score| format!("{score:.3}"));
        println!(
            "Cluster {} ({} nodes, average similarity: {})",
            rank + 1,
            cluster.len(),
            similarity
        );
        for &index in cluster {
            let item = &embeddings[index];
            println!("  {}:{}", item.filepath, item.node_name);
        }
        println!();
    }

    Ok(())
}

// For unit vectors, sum of all distinct pairwise dot products is
// (||sum(v)||² - n) / 2. This avoids a quadratic pairwise scoring pass.
fn average_similarity<'a>(embeddings: impl Iterator<Item = &'a [u8]>) -> Option<f64> {
    let mut sum = Vec::<f64>::new();
    let mut count = 0;
    for bytes in embeddings {
        let vector = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()) as f64)
            .collect::<Vec<_>>();
        let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return None;
        }
        if count == 0 {
            sum.resize(vector.len(), 0.0);
        }
        for (total, value) in sum.iter_mut().zip(vector) {
            *total += value / norm;
        }
        count += 1;
    }
    if count < 2 {
        return None;
    }
    let n = count as f64;
    Some(
        ((sum.iter().map(|value| value * value).sum::<f64>() - n) / (n * (n - 1.0)))
            .clamp(-1.0, 1.0),
    )
}

fn rank_clusters(clusters: &mut [(Vec<usize>, Option<f64>)]) {
    clusters.sort_by(|(left, left_score), (right, right_score)| {
        match (left_score, right_score) {
            (Some(left), Some(right)) => right.total_cmp(left),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        }
        .then_with(|| left[0].cmp(&right[0]))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score(vectors: &[[f32; 2]]) -> Option<f64> {
        let blobs = vectors
            .iter()
            .map(|vector| {
                vector
                    .iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        average_similarity(blobs.iter().map(Vec::as_slice))
    }

    #[test]
    fn scores_all_distinct_pairs() {
        assert_eq!(score(&[[1.0, 0.0]]), None);
        assert_eq!(score(&[[1.0, 0.0], [2.0, 0.0]]), Some(1.0));
        assert_eq!(score(&[[1.0, 0.0], [0.0, 1.0]]), Some(0.0));
        assert_eq!(score(&[[1.0, 0.0], [-1.0, 0.0]]), Some(-1.0));
        let average = score(&[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]).unwrap();
        assert!((average + 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn ranks_by_similarity_not_size_and_keeps_all_clusters() {
        let mut clusters = vec![(vec![0, 1, 2], Some(0.8)), (vec![3, 4], Some(0.95))];
        clusters.extend((5..20).map(|index| (vec![index], None)));
        rank_clusters(&mut clusters);
        assert_eq!(clusters.len(), 17);
        assert_eq!(clusters[0].0, [3, 4]);
        assert_eq!(clusters[1].0, [0, 1, 2]);
        assert_eq!(clusters[2].0, [5]);
    }
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
