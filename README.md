# slopmop

Indexes a Go project with Tree-sitter, creates Jina code embeddings with Candle, and stores functions, methods, structs, and interfaces in SQLite using [SQLite-Vector](https://github.com/sqliteai/sqlite-vector).

## SQLite-Vector

On the first run, slopmop automatically downloads SQLite-Vector 1.1.0 for the current platform and stores it in the system cache. To use an existing or custom build instead, set `SQLITE_VECTOR_PATH` to its shared library:

```sh
export SQLITE_VECTOR_PATH=/absolute/path/to/vector.dylib # macOS
# export SQLITE_VECTOR_PATH=/absolute/path/to/vector.so  # Linux
# set SQLITE_VECTOR_PATH=C:\absolute\path\to\vector.dll # Windows
```

## Index a Go project

Build or install the executable:

```sh
cargo install --path .
```

Then run it from a Go project:

```sh
cd path/to/go-project
slopmop
```

Alternatively, provide the project directory while developing slopmop:

```sh
cargo run --release -- path/to/go-project
```

The index is created as `.slopmop` in the project root. Each run recursively finds `.go` files and atomically replaces the existing index. The first run downloads and caches `jinaai/jina-embeddings-v2-base-code` from Hugging Face.

### Ignore files and directories

Add a `.slopmopignore` file to the project root to exclude paths from indexing. It uses gitignore syntax, including comments, glob patterns, directory patterns, root-relative patterns, and `!` negation:

```gitignore
# Generated code
internal/generated/
*.generated.go

# Ignore one file only at the project root
/legacy.go

# Re-include an otherwise ignored file
!important.generated.go
```

Ignored directories are not traversed. As with `.gitignore`, a file inside an ignored directory cannot be re-included unless its parent directory is also re-included.

## Cluster similar nodes

From an indexed project, list the 10 largest similarity clusters:

```sh
slopmop cluster
```

The default minimum cosine similarity is `0.8`. It can be changed explicitly, and a project directory can be supplied:

```sh
slopmop cluster path/to/go-project --threshold 0.75
```

Nodes are connected when their cosine similarity meets the threshold. Transitive connected nodes belong to the same cluster.

The database contains two tables:

```text
files
  id        INTEGER PRIMARY KEY
  filepath  TEXT NOT NULL UNIQUE

embeddings
  id         INTEGER PRIMARY KEY
  embedding  BLOB NOT NULL
  node_name  TEXT NOT NULL
  file_id    INTEGER NOT NULL REFERENCES files(id)
```

Embeddings are normalized 768-dimensional `FLOAT32` vectors. The vector column uses cosine distance and can be queried with SQLite-Vector's `vector_full_scan` or, after quantization, `vector_quantize_scan`.
