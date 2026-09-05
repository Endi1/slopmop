use std::{env, fs, path::PathBuf};

use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use tar::Archive;

const VERSION: &str = "1.1.0";

pub fn extension_path() -> Result<PathBuf> {
    if let Some(path) = env::var_os("SQLITE_VECTOR_PATH") {
        return Ok(PathBuf::from(path));
    }

    let (asset, library) = platform_release()?;
    let cache_directory = dirs::cache_dir()
        .context("could not determine the system cache directory")?
        .join("slopmop")
        .join("sqlite-vector")
        .join(VERSION);
    let library_path = cache_directory.join(library);
    if library_path.is_file() {
        return Ok(library_path);
    }

    fs::create_dir_all(&cache_directory).with_context(|| {
        format!(
            "failed to create SQLite-Vector cache at {}",
            cache_directory.display()
        )
    })?;

    let url =
        format!("https://github.com/sqliteai/sqlite-vector/releases/download/{VERSION}/{asset}");
    eprintln!("Downloading SQLite-Vector {VERSION} for this platform...");
    let response = ureq::get(&url)
        .call()
        .with_context(|| format!("failed to download {url}"))?;
    let reader = response.into_parts().1.into_reader();
    Archive::new(GzDecoder::new(reader))
        .unpack(&cache_directory)
        .context("failed to extract SQLite-Vector")?;

    if !library_path.is_file() {
        bail!(
            "SQLite-Vector archive did not contain {}",
            library_path.display()
        );
    }
    Ok(library_path)
}

fn platform_release() -> Result<(&'static str, &'static str)> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Ok(("vector-macos-arm64-1.1.0.tar.gz", "vector.dylib"))
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        Ok(("vector-macos-x86_64-1.1.0.tar.gz", "vector.dylib"))
    } else if cfg!(all(
        target_os = "linux",
        target_arch = "aarch64",
        target_env = "musl"
    )) {
        Ok(("vector-linux-musl-arm64-1.1.0.tar.gz", "vector.so"))
    } else if cfg!(all(
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "musl"
    )) {
        Ok(("vector-linux-musl-x86_64-1.1.0.tar.gz", "vector.so"))
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        Ok(("vector-linux-arm64-1.1.0.tar.gz", "vector.so"))
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Ok(("vector-linux-x86_64-1.1.0.tar.gz", "vector.so"))
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Ok(("vector-windows-x86_64-1.1.0.tar.gz", "vector.dll"))
    } else {
        bail!(
            "SQLite-Vector has no prebuilt release for {}-{}; set SQLITE_VECTOR_PATH manually",
            env::consts::OS,
            env::consts::ARCH
        )
    }
}
