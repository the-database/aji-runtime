//! Downloads and archive extraction.
//!
//! Downloads run under the store lock, so one that hangs would block every other app's install and
//! uninstall. So connecting has a time limit ([`CONNECT_TIMEOUT`]), a download that receives nothing
//! for [`STALL_TIMEOUT`] fails (one that's slow but still receiving never does: there's no limit on
//! the whole body), and a transient failure is retried after [`RETRY_DELAYS`], resuming where it
//! stopped with an HTTP Range request (from the start if the server doesn't do ranges). The caller
//! checks the whole file's SHA-256 as before, so a resumed download is verified like any other.

use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::index::check_rel_path;
use crate::{Result, err};

const UA: &str = concat!("aji-runtime/", env!("CARGO_PKG_VERSION"));

/// Resolving the host and connecting (with the TLS handshake), each.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// A download (or a response's headers) that receives no bytes for this long fails, and is retried.
pub const STALL_TIMEOUT: Duration = Duration::from_secs(30);
/// The waits before each retry after a failure that received nothing new: 4 retries in about a
/// minute. A failure after new bytes starts this over (a long download on a flaky connection
/// keeps going while it gets somewhere).
pub const RETRY_DELAYS: [Duration; 4] =
    [Duration::from_secs(2), Duration::from_secs(5), Duration::from_secs(15), Duration::from_secs(30)];
/// Attempts per download in all, however much each one received.
pub const MAX_ATTEMPTS: u32 = 20;

/// The timeouts and retries of a download (tests use short ones).
#[derive(Clone, Debug)]
pub struct Policy {
    pub connect: Duration,
    pub stall: Duration,
    pub retry_delays: Vec<Duration>,
    pub max_attempts: u32,
}

impl Default for Policy {
    fn default() -> Policy {
        Policy { connect: CONNECT_TIMEOUT, stall: STALL_TIMEOUT, retry_delays: RETRY_DELAYS.to_vec(), max_attempts: MAX_ATTEMPTS }
    }
}

fn agent(policy: &Policy) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_resolve(Some(policy.connect))
        .timeout_connect(Some(policy.connect))
        // restarts with every read: a stall, not a slow download
        .timeout_per_read(Some(policy.stall))
        .timeout_per_write(Some(policy.stall))
        .build()
        .into()
}

fn secs(d: Duration) -> String {
    if d >= Duration::from_secs(1) { format!("{} s", d.as_secs()) } else { format!("{} ms", d.as_millis()) }
}

/// Why an attempt failed, and whether trying again may help.
enum Failure {
    Retry(String),
    Fatal(String),
}

/// A failed request or body read: timeouts, dropped connections and server errors are worth
/// another try; a missing file, a refused request or an unknown host (offline) aren't.
fn failure(e: ureq::Error, policy: &Policy) -> Failure {
    use ureq::Error as E;
    match e {
        E::Timeout(ureq::Timeout::Resolve | ureq::Timeout::Connect) => {
            Failure::Retry(format!("couldn't connect within {}", secs(policy.connect)))
        }
        E::Timeout(_) => Failure::Retry(format!("no data for {}", secs(policy.stall))),
        E::StatusCode(c) if c == 408 || c == 429 || c >= 500 => Failure::Retry(format!("the server answered {c}")),
        E::Io(io) if io.kind() == std::io::ErrorKind::UnexpectedEof => Failure::Retry("the connection closed early".into()),
        e @ (E::Io(_) | E::ConnectionFailed | E::Protocol(_)) => Failure::Retry(e.to_string()),
        e => Failure::Fatal(e.to_string()),
    }
}

/// Runs `attempt` until it succeeds, a failure isn't transient, or the retries run out. `progress`
/// says whether the attempt got anywhere (which starts the retry delays over).
fn with_retries<T>(
    url: &str,
    policy: &Policy,
    mut attempt: impl FnMut(&ureq::Agent) -> std::result::Result<T, Failure>,
    mut progress: impl FnMut() -> u64,
) -> Result<T> {
    let agent = agent(policy);
    let name = url.rsplit('/').next().unwrap_or(url);
    let (mut attempts, mut fails) = (0u32, 0usize);
    loop {
        let before = progress();
        attempts += 1;
        let why = match attempt(&agent) {
            Ok(v) => return Ok(v),
            Err(Failure::Fatal(why)) => return Err(err(format!("downloading {url}: {why}"))),
            Err(Failure::Retry(why)) => why,
        };
        let now = progress();
        if now > before {
            fails = 0;
        }
        if fails >= policy.retry_delays.len() || attempts >= policy.max_attempts {
            return Err(err(format!("downloading {url}: {why} (gave up after {attempts} attempts)")));
        }
        let wait = policy.retry_delays[fails];
        fails += 1;
        let from = if now > 0 { format!(", resuming at {:.0} MB", now as f64 / 1e6) } else { String::new() };
        eprintln!("  {name}: {why}; trying again in {}{from}", secs(wait));
        std::thread::sleep(wait);
    }
}

/// Fetches a small file (an index) into memory.
pub fn fetch_small(url: &str) -> Result<Vec<u8>> {
    fetch_small_with(url, &Policy::default())
}

pub fn fetch_small_with(url: &str, policy: &Policy) -> Result<Vec<u8>> {
    with_retries(
        url,
        policy,
        |agent| {
            let mut resp = agent.get(url).header("User-Agent", UA).call().map_err(|e| failure(e, policy))?;
            resp.body_mut().with_config().limit(64 << 20).read_to_vec().map_err(|e| failure(e, policy))
        },
        || 0,
    )
}

/// Downloads `url` to `dest`, reporting (bytes so far, total) at most once per percent. `expected` is
/// the index's size: a download that ends short of it is resumed.
pub fn download(url: &str, dest: &Path, expected: u64, progress: &mut dyn FnMut(u64, u64)) -> Result<()> {
    download_with(url, dest, expected, &Policy::default(), progress)
}

pub fn download_with(url: &str, dest: &Path, expected: u64, policy: &Policy, progress: &mut dyn FnMut(u64, u64)) -> Result<()> {
    let file = std::fs::File::create(dest).map_err(|e| err(format!("{}: {e}", dest.display())))?;
    let mut out = std::io::BufWriter::with_capacity(1 << 20, file);
    // bytes in `dest` so far: shared by the attempts and the retry loop
    let got = std::cell::Cell::new(0u64);
    with_retries(url, policy, |agent| attempt(agent, url, dest, expected, policy, &mut out, &got, progress), || got.get())?;
    out.flush().map_err(|e| err(format!("{}: {e}", dest.display())))?;
    Ok(())
}

/// One request: from the start, or from `got` with a Range request. Appends to `out`.
#[allow(clippy::too_many_arguments)]
fn attempt(
    agent: &ureq::Agent,
    url: &str,
    dest: &Path,
    expected: u64,
    policy: &Policy,
    out: &mut std::io::BufWriter<std::fs::File>,
    got: &std::cell::Cell<u64>,
    progress: &mut dyn FnMut(u64, u64),
) -> std::result::Result<(), Failure> {
    let disk = |e: std::io::Error| Failure::Fatal(format!("{}: {e}", dest.display()));
    let from = got.get();
    // Range offsets are of the bytes as sent: no compression
    let mut req = agent.get(url).header("User-Agent", UA).header("Accept-Encoding", "identity");
    if from > 0 {
        req = req.header("Range", format!("bytes={from}-"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        // what's there is already the whole file (resumed exactly at its end)
        Err(ureq::Error::StatusCode(416)) if from > 0 && from == expected => return Ok(()),
        Err(ureq::Error::StatusCode(416)) => {
            restart(out, got).map_err(disk)?;
            return Err(Failure::Retry("the server refused to resume; starting over".into()));
        }
        Err(e) => return Err(failure(e, policy)),
    };
    let header = |name: &str| resp.headers().get(name).and_then(|v| v.to_str().ok()).map(str::to_owned);
    let length = header("content-length").and_then(|v| v.parse::<u64>().ok());
    let resumed = from > 0 && resp.status().as_u16() == 206;
    if resumed && !header("content-range").is_some_and(|r| r.starts_with(&format!("bytes {from}-"))) {
        restart(out, got).map_err(disk)?;
        return Err(Failure::Retry("the server resumed at the wrong place; starting over".into()));
    }
    if from > 0 && !resumed {
        // the server sent the whole file: start over
        restart(out, got).map_err(disk)?;
    }
    let total = match length {
        Some(n) => got.get() + n,
        None => expected,
    };
    progress(got.get(), total);
    let mut said = got.get().checked_mul(100).and_then(|x| x.checked_div(total)).unwrap_or(0);
    let mut reader = resp.into_body().into_reader();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = match reader.read(&mut buf) {
            Ok(n) => n,
            Err(e) => return Err(failure(ureq::Error::from(e), policy)),
        };
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(disk)?;
        got.set(got.get() + n as u64);
        if total > 0 && got.get() * 100 / total != said {
            said = got.get() * 100 / total;
            progress(got.get(), total);
        }
    }
    // without a Content-Length, a dropped connection looks like the end: the index says how big
    if got.get() < expected {
        return Err(Failure::Retry(format!("the connection closed early ({} of {expected} bytes)", got.get())));
    }
    Ok(())
}

/// Empties the download to start it over.
fn restart(out: &mut std::io::BufWriter<std::fs::File>, got: &std::cell::Cell<u64>) -> std::io::Result<()> {
    out.flush()?;
    out.get_mut().set_len(0)?;
    out.seek(std::io::SeekFrom::Start(0))?;
    got.set(0);
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
