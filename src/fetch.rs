//! Downloads and archive extraction.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::index::check_rel_path;
use crate::{Result, err};

const UA: &str = concat!("aji-runtime/", env!("CARGO_PKG_VERSION"));

/// Fetches a small file (an index) into memory.
pub fn fetch_small(url: &str) -> Result<Vec<u8>> {
    let mut resp = ureq::get(url).header("User-Agent", UA).call().map_err(|e| err(format!("downloading {url}: {e}")))?;
    resp.body_mut().with_config().limit(64 << 20).read_to_vec().map_err(|e| err(format!("downloading {url}: {e}")))
}

/// Downloads `url` to `dest`, reporting (bytes so far, total) at most once per percent.
pub fn download(url: &str, dest: &Path, expected: u64, progress: &mut dyn FnMut(u64, u64)) -> Result<()> {
    let resp = ureq::get(url).header("User-Agent", UA).call().map_err(|e| err(format!("downloading {url}: {e}")))?;
    let total = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(expected);
    let mut reader = resp.into_body().into_reader();
    let mut out = std::io::BufWriter::new(std::fs::File::create(dest).map_err(|e| err(format!("{}: {e}", dest.display())))?);
    let (mut got, mut said) = (0u64, u64::MAX);
    progress(0, total);
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf).map_err(|e| err(format!("downloading {url}: {e}")))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| err(format!("{}: {e}", dest.display())))?;
        got += n as u64;
        if total > 0 && got * 100 / total != said {
            said = got * 100 / total;
            progress(got, total);
        }
    }
    out.flush().map_err(|e| err(format!("{}: {e}", dest.display())))?;
    Ok(())
}

/// Extracts a .7z or .zip archive into `dest`, keeping relative paths. Returns the files written.
pub fn extract(archive: &Path, dest: &Path) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dest).map_err(|e| err(format!("{}: {e}", dest.display())))?;
    let name = archive.to_string_lossy().to_ascii_lowercase();
    let mut files = Vec::new();
    if name.ends_with(".zip") {
        let f = std::fs::File::open(archive).map_err(|e| err(format!("{}: {e}", archive.display())))?;
        let mut zip = zip::ZipArchive::new(f).map_err(|e| err(format!("{}: {e}", archive.display())))?;
        for i in 0..zip.len() {
            let mut e = zip.by_index(i).map_err(|e| err(format!("{}: {e}", archive.display())))?;
            if e.is_dir() {
                continue;
            }
            let entry = e.name().map_err(|x| err(format!("{}: {x}", archive.display())))?.replace('\\', "/");
            check_rel_path(&entry)?;
            let out = dest.join(&entry);
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent).map_err(|x| err(format!("{}: {x}", parent.display())))?;
            }
            let mut w = std::fs::File::create(&out).map_err(|x| err(format!("{}: {x}", out.display())))?;
            std::io::copy(&mut e, &mut w).map_err(|x| err(format!("{}: {x}", out.display())))?;
            files.push(out);
        }
    } else if name.ends_with(".7z") {
        let mut bad = None;
        sevenz_rust2::decompress_file_with_extract_fn(archive, dest, |entry, reader, path| {
            if !entry.is_directory() {
                let n = entry.name().replace('\\', "/");
                if let Err(e) = check_rel_path(&n) {
                    bad = Some(e);
                    return Ok(false);
                }
                files.push(path.clone());
            }
            sevenz_rust2::default_entry_extract_fn(entry, reader, path)
        })
        .map_err(|e| err(format!("extracting {}: {e}", archive.display())))?;
        if let Some(e) = bad {
            return Err(e);
        }
    } else {
        return Err(err(format!("{}: not a .7z or .zip", archive.display())));
    }
    #[cfg(unix)]
    for f in &files {
        use std::os::unix::fs::PermissionsExt;
        if f.file_name().and_then(|n| n.to_str()) == Some("trtexec") {
            let _ = std::fs::set_permissions(f, std::fs::Permissions::from_mode(0o755));
        }
    }
    Ok(files)
}
