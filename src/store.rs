//! The per-user pack store:
//!
//! ```text
//! <root>/                      %LOCALAPPDATA%\the-database\aji  /  $XDG_DATA_HOME/the-database/aji
//!   store.json                 {"format": 1}
//!   lock                       held exclusively while anything changes the store
//!   tmp/                       downloads, extractions and deletions in progress; cleared under the lock
//!   packs/<name>/<version>/    one folder per exact pack version, never changed in place;
//!       .pack.json             complete only once this is there (written last)
//!   refs/<app-id>.json         one per installed app: the packs it uses
//! ```
//!
//! Rules (plans/shared-runtime-packs.md in the-database/animejan.ai):
//! - A pack is deleted only when no live ref lists it. A ref is live if it parses and its
//!   `app_path` exists; dead refs are removed by [`Store::gc`].
//! - Everything that changes the store takes [`Store::lock`] first; the functions that change it
//!   take the [`StoreLock`] as a parameter, so they can't be called without it. A waiter gives up
//!   after [`LOCK_WAIT`] with an error, never by breaking the lock.
//! - Every change is a rename (a pack folder in or out whole, a ref or `aji-runtime.json` replaced
//!   by [`write_atomic`]), so reading without the lock (`ensure --check`, `show`) sees each pack and
//!   ref as it was before a change or after it, never half.
//! - A pack is deleted by renaming it into `tmp/` first, so `packs/` only ever holds complete
//!   packs. On Windows, a pack with a file in use (a loaded DLL) is skipped and retried by the
//!   next gc, never half-deleted.
//! - A complete pack is installed once and never replaced: it counts as installed when its
//!   `.pack.json` lists the index's files, whatever archive they came in, and installing the same
//!   name/version with other files fails ([`Store::commit_pack`]). Only gc deletes complete packs.
//! - A store whose format is newer than [`FORMAT`] is read and added to, but nothing is deleted.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::{Error, Result, err};

/// The store format this version writes and fully understands.
pub const FORMAT: u32 = 1;

/// How long [`Store::lock`] waits for another app's install, update or move before giving up. Its
/// downloads can't hang (`fetch`'s timeouts), so this is only for an install that takes this long
/// (gigabytes on a slow connection) or a holder that hangs; giving up changes nothing.
pub const LOCK_WAIT: Duration = Duration::from_secs(60 * 60);

/// How often a waiting [`Store::lock`] tries again.
const LOCK_POLL: Duration = Duration::from_millis(250);

/// "5 minutes", "1 minute", or seconds below a minute (tests).
pub fn minutes(d: Duration) -> String {
    match d.as_secs() {
        s if s < 60 => format!("{s} s"),
        60..=119 => "1 minute".into(),
        s => format!("{} minutes", s / 60),
    }
}

#[derive(Serialize, Deserialize)]
struct StoreJson {
    format: u32,
}

/// One file of a pack, relative to the pack folder.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PackFile {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

/// `.pack.json`: what a complete pack folder holds.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PackMeta {
    pub format: u32,
    pub name: String,
    pub version: String,
    pub kind: String,
    pub platform: String,
    /// The archive it came from (URL, or `adopted from <folder>`).
    pub source: String,
    /// The SHA-256 of the archive it was installed from (the index's pin then). For information
    /// only: a later index may pin another archive of the same files, which is the same pack.
    pub sha256: String,
    pub files: Vec<PackFile>,
}

/// `refs/<app-id>.json`: the packs one installed app uses.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AppRef {
    pub app_id: String,
    #[serde(default)]
    pub app_version: String,
    /// The app's executable: the ref is live while it exists.
    pub app_path: PathBuf,
    /// `name/version` of each pack.
    pub packs: Vec<String>,
}

/// A ref file as found on disk.
#[derive(Debug)]
pub struct RefEntry {
    pub file: PathBuf,
    pub app_ref: Option<AppRef>,
    pub live: bool,
}

/// What a gc did.
#[derive(Debug, Default)]
pub struct GcReport {
    /// `name/version` of each pack deleted.
    pub deleted: Vec<String>,
    /// Unreferenced packs with a file in use: kept for now, retried next time.
    pub in_use: Vec<String>,
    /// Refs whose app is gone (or that didn't parse), removed.
    pub dead_refs: Vec<PathBuf>,
    /// Incomplete pack folders (no `.pack.json`) removed.
    pub incomplete: Vec<String>,
    /// True when the store's format is newer than this program's: nothing was deleted.
    pub newer_format: bool,
}

pub struct Store {
    root: PathBuf,
    format: u32,
}

/// The exclusive store lock, held until dropped.
pub struct StoreLock {
    _file: File,
}

/// `name/version` of a pack.
pub fn pack_key(name: &str, version: &str) -> String {
    format!("{name}/{version}")
}

/// Names, versions and app ids become folder and file names: keep them to a safe alphabet.
pub fn check_component(what: &str, s: &str) -> Result<()> {
    let ok = !s.is_empty()
        && s.len() <= 100
        && s != "."
        && s != ".."
        && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '+' | '-'));
    if ok { Ok(()) } else { Err(err(format!("invalid {what} {s:?}"))) }
}

/// Writes `bytes` to `path` through a temporary file and a rename, so readers see the old file or
/// the new one, never a partial one.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    {
        let mut f = File::create(&tmp).map_err(|e| err(format!("{}: {e}", tmp.display())))?;
        f.write_all(bytes).map_err(|e| err(format!("{}: {e}", tmp.display())))?;
        f.sync_all().map_err(|e| err(format!("{}: {e}", tmp.display())))?;
    }
    fs::rename(&tmp, path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        err(format!("{}: {e}", path.display()))
    })
}

impl Store {
    /// The per-user store's default location.
    pub fn default_root() -> Result<PathBuf> {
        if let Some(p) = std::env::var_os("AJI_RUNTIME_STORE").filter(|p| !p.is_empty()) {
            return Ok(PathBuf::from(p));
        }
        if cfg!(windows) {
            let base = std::env::var_os("LOCALAPPDATA").ok_or_else(|| err("LOCALAPPDATA is not set"))?;
            Ok(PathBuf::from(base).join("the-database").join("aji"))
        } else {
            let base = match std::env::var_os("XDG_DATA_HOME").filter(|p| !p.is_empty()) {
                Some(p) => PathBuf::from(p),
                None => PathBuf::from(std::env::var_os("HOME").ok_or_else(|| err("HOME is not set"))?)
                    .join(".local")
                    .join("share"),
            };
            Ok(base.join("the-database").join("aji"))
        }
    }

    /// Opens (creating if needed) the store at `root`.
    pub fn open(root: &Path) -> Result<Store> {
        for d in [root.to_path_buf(), root.join("tmp"), root.join("packs"), root.join("refs")] {
            fs::create_dir_all(&d).map_err(|e| err(format!("{}: {e}", d.display())))?;
        }
        let sj = root.join("store.json");
        let format = match fs::read(&sj) {
            Ok(b) => serde_json::from_slice::<StoreJson>(&b)
                .map_err(|e| err(format!("{}: {e}", sj.display())))?
                .format,
            Err(_) => {
                write_atomic(&sj, &serde_json::to_vec_pretty(&StoreJson { format: FORMAT }).unwrap())?;
                FORMAT
            }
        };
        Ok(Store { root: root.to_path_buf(), format })
    }

    /// Opens an existing store without creating anything (for lookups), or None.
    pub fn open_existing(root: &Path) -> Option<Store> {
        let b = fs::read(root.join("store.json")).ok()?;
        let format = serde_json::from_slice::<StoreJson>(&b).ok()?.format;
        Some(Store { root: root.to_path_buf(), format })
    }

    /// The store at `root` for reading only (`ensure --check`): creates nothing, and a store that
    /// isn't there yet reads as empty. An unreadable `store.json` is an error, as for [`Store::open`].
    pub fn open_read_only(root: &Path) -> Result<Store> {
        let sj = root.join("store.json");
        let format = match fs::read(&sj) {
            Ok(b) => serde_json::from_slice::<StoreJson>(&b).map_err(|e| err(format!("{}: {e}", sj.display())))?.format,
            Err(_) => FORMAT,
        };
        Ok(Store { root: root.to_path_buf(), format })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn format(&self) -> u32 {
        self.format
    }

    /// False when a newer program wrote this store: then nothing may be deleted.
    pub fn may_delete(&self) -> bool {
        self.format <= FORMAT
    }

    pub fn tmp_dir(&self) -> PathBuf {
        self.root.join("tmp")
    }

    pub fn pack_dir(&self, name: &str, version: &str) -> PathBuf {
        self.root.join("packs").join(name).join(version)
    }

    fn ref_file(&self, app_id: &str) -> PathBuf {
        self.root.join("refs").join(format!("{app_id}.json"))
    }

    /// Takes the store lock, waiting up to [`LOCK_WAIT`] for another installer; `waiting` is called
    /// once first if it has to wait.
    pub fn lock(&self, waiting: &mut dyn FnMut()) -> Result<StoreLock> {
        self.lock_within(LOCK_WAIT, waiting)
    }

    /// Takes the store lock, waiting up to `limit` for another installer; `waiting` is called once
    /// first if it has to wait. Giving up is an error that changes nothing: the lock is only ever
    /// taken, never broken (the OS releases it when its holder exits).
    pub fn lock_within(&self, limit: Duration, waiting: &mut dyn FnMut()) -> Result<StoreLock> {
        let p = self.root.join("lock");
        let f = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&p)
            .map_err(|e| err(format!("{}: {e}", p.display())))?;
        let start = Instant::now();
        let mut said = false;
        loop {
            match f.try_lock() {
                Ok(()) => return Ok(StoreLock { _file: f }),
                Err(fs::TryLockError::WouldBlock) => {}
                Err(fs::TryLockError::Error(e)) => return Err(err(format!("{}: {e}", p.display()))),
            }
            let waited = start.elapsed();
            if waited >= limit {
                return Err(err(format!(
                    "gave up waiting for the pack store after {}: another app's install or update has held it                      all that time ({}). Nothing was changed; try again once it has finished.",
                    minutes(limit),
                    p.display()
                )));
            }
            if !said {
                said = true;
                waiting();
            }
            std::thread::sleep(LOCK_POLL.min(limit - waited));
        }
    }

    /// The lock if it's free right now, else None.
    pub fn try_lock(&self) -> Result<Option<StoreLock>> {
        let p = self.root.join("lock");
        let f = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&p)
            .map_err(|e| err(format!("{}: {e}", p.display())))?;
        match f.try_lock() {
            Ok(()) => Ok(Some(StoreLock { _file: f })),
            Err(fs::TryLockError::WouldBlock) => Ok(None),
            Err(fs::TryLockError::Error(e)) => Err(err(format!("{}: {e}", p.display()))),
        }
    }

    /// The `.pack.json` of a complete pack, or None.
    pub fn installed(&self, name: &str, version: &str) -> Option<PackMeta> {
        let b = fs::read(self.pack_dir(name, version).join(".pack.json")).ok()?;
        serde_json::from_slice(&b).ok()
    }

    /// Whether `packs/<name>/<version>` is a complete pack (has its `.pack.json`), readable or not.
    pub fn is_complete(&self, name: &str, version: &str) -> bool {
        self.pack_dir(name, version).join(".pack.json").is_file()
    }

    /// Checks that the complete pack `name/version` holds exactly `files` (by its `.pack.json`).
    /// An error names the conflict: the folder holds other files, or its `.pack.json` can't be read.
    pub fn check_same_files(&self, name: &str, version: &str, files: &[PackFile]) -> Result<()> {
        let key = pack_key(name, version);
        match self.installed(name, version) {
            Some(m) if same_files(&m.files, files) => Ok(()),
            Some(_) => Err(err(format!(
                "the pack store already has {key} with other files than the runtime index lists for it; an \
                 installed pack is never replaced (the index reuses a version for different files: a runtime \
                 release error)"
            ))),
            None => Err(err(format!(
                "the pack store's {key} has a .pack.json that can't be read ({}); an installed pack is never \
                 replaced",
                self.pack_dir(name, version).join(".pack.json").display()
            ))),
        }
    }

    /// Every complete pack: (meta, folder, bytes on disk).
    pub fn packs(&self) -> Vec<(PackMeta, PathBuf, u64)> {
        let mut out = Vec::new();
        for (name, version) in self.pack_folders() {
            if let Some(m) = self.installed(&name, &version) {
                let bytes = m.files.iter().map(|f| f.bytes).sum();
                out.push((m, self.pack_dir(&name, &version), bytes));
            }
        }
        out
    }

    /// (name, version) of every folder under packs/, complete or not.
    fn pack_folders(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        let Ok(names) = fs::read_dir(self.root.join("packs")) else { return out };
        for n in names.flatten() {
            if !n.path().is_dir() {
                continue;
            }
            let Ok(versions) = fs::read_dir(n.path()) else { continue };
            for v in versions.flatten() {
                if v.path().is_dir() {
                    out.push((n.file_name().to_string_lossy().into_owned(), v.file_name().to_string_lossy().into_owned()));
                }
            }
        }
        out.sort();
        out
    }

    /// Every ref file, with whether it's live.
    pub fn refs(&self) -> Vec<RefEntry> {
        let mut out = Vec::new();
        let Ok(rd) = fs::read_dir(self.root.join("refs")) else { return out };
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }
            let app_ref = fs::read(&p).ok().and_then(|b| serde_json::from_slice::<AppRef>(&b).ok());
            let live = app_ref.as_ref().is_some_and(|r| r.app_path.exists());
            out.push(RefEntry { file: p, app_ref, live });
        }
        out.sort_by(|a, b| a.file.cmp(&b.file));
        out
    }

    pub fn read_ref(&self, app_id: &str) -> Option<AppRef> {
        let b = fs::read(self.ref_file(app_id)).ok()?;
        serde_json::from_slice(&b).ok()
    }

    /// Writes an app's ref whole, replacing any older one.
    pub fn write_ref(&self, _lock: &StoreLock, r: &AppRef) -> Result<()> {
        check_component("app id", &r.app_id)?;
        write_atomic(&self.ref_file(&r.app_id), &serde_json::to_vec_pretty(r).unwrap())
    }

    /// Removes an app's ref (no error if it has none). Returns whether there was one.
    pub fn remove_ref(&self, _lock: &StoreLock, app_id: &str) -> Result<bool> {
        check_component("app id", app_id)?;
        match fs::remove_file(self.ref_file(app_id)) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(err(format!("{}: {e}", self.ref_file(app_id).display()))),
        }
    }

    /// Empties tmp/ (what's left there is from an interrupted install or delete). Under the lock,
    /// nothing else is using it.
    pub fn clean_tmp(&self, _lock: &StoreLock) {
        if let Ok(rd) = fs::read_dir(self.tmp_dir()) {
            for e in rd.flatten() {
                let p = e.path();
                let _ = if p.is_dir() { fs::remove_dir_all(&p) } else { fs::remove_file(&p) };
            }
        }
    }

    /// A fresh, empty folder under tmp/.
    pub fn tmp_folder(&self, _lock: &StoreLock, label: &str) -> Result<PathBuf> {
        for i in 0u32.. {
            let p = self.tmp_dir().join(format!("{label}-{}-{i}", std::process::id()));
            match fs::create_dir(&p) {
                Ok(()) => return Ok(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(err(format!("{}: {e}", p.display()))),
            }
        }
        unreachable!()
    }

    /// Moves a complete pack folder (its `.pack.json` already written) from tmp/ into packs/.
    ///
    /// A complete pack already there is never replaced or deleted: other apps use it, and on Windows
    /// may have its DLLs loaded. If it holds the same files, it's kept and `staged` is dropped;
    /// otherwise (other files, or an unreadable `.pack.json`) this fails and changes nothing.
    pub fn commit_pack(&self, _lock: &StoreLock, staged: &Path, name: &str, version: &str) -> Result<PathBuf> {
        check_component("pack name", name)?;
        check_component("pack version", version)?;
        let meta = staged.join(".pack.json");
        let staged_meta: PackMeta = fs::read(&meta)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .ok_or_else(|| err(format!("{}: no .pack.json", staged.display())))?;
        let dest = self.pack_dir(name, version);
        if self.is_complete(name, version) {
            self.check_same_files(name, version, &staged_meta.files)?;
            let _ = fs::remove_dir_all(staged);
            return Ok(dest);
        }
        if dest.exists() {
            // An incomplete leftover (no .pack.json): out of the way.
            self.remove_pack_folder(&dest)?;
        }
        fs::create_dir_all(dest.parent().unwrap()).map_err(|e| err(format!("{}: {e}", dest.display())))?;
        fs::rename(staged, &dest).map_err(|e| err(format!("{}: {e}", dest.display())))?;
        Ok(dest)
    }

    /// Deletes a pack folder: first renamed into tmp/ (one step, so packs/ never holds half a
    /// pack), then removed; anything that can't be removed now is cleared from tmp/ later.
    fn remove_pack_folder(&self, dir: &Path) -> Result<()> {
        let trash = self.tmp_dir().join(format!(
            "del-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
        ));
        fs::rename(dir, &trash).map_err(|e| err(format!("{}: {e}", dir.display())))?;
        let _ = fs::remove_dir_all(&trash);
        Ok(())
    }

    /// Deletes every pack no live ref lists, the dead refs, and incomplete pack folders.
    pub fn gc(&self, lock: &StoreLock) -> Result<GcReport> {
        let mut report = GcReport::default();
        if !self.may_delete() {
            report.newer_format = true;
            return Ok(report);
        }
        self.clean_tmp(lock);
        let mut keep = BTreeSet::new();
        for r in self.refs() {
            match (&r.app_ref, r.live) {
                (Some(a), true) => keep.extend(a.packs.iter().cloned()),
                _ => {
                    if fs::remove_file(&r.file).is_ok() {
                        report.dead_refs.push(r.file.clone());
                    }
                }
            }
        }
        for (name, version) in self.pack_folders() {
            let key = pack_key(&name, &version);
            let dir = self.pack_dir(&name, &version);
            let complete = dir.join(".pack.json").is_file();
            if complete && keep.contains(&key) {
                continue;
            }
            if in_use(&dir) {
                report.in_use.push(key);
                continue;
            }
            if self.remove_pack_folder(&dir).is_ok() {
                if complete { report.deleted.push(key) } else { report.incomplete.push(key) }
            } else {
                report.in_use.push(key);
            }
        }
        // Empty packs/<name>/ folders.
        if let Ok(rd) = fs::read_dir(self.root.join("packs")) {
            for e in rd.flatten() {
                let _ = fs::remove_dir(e.path());
            }
        }
        Ok(report)
    }
}

/// Whether two file lists are the same files: the same paths with the same sizes and SHA-256s, in
/// any order.
pub fn same_files(a: &[PackFile], b: &[PackFile]) -> bool {
    let key = |l: &[PackFile]| {
        let mut v: Vec<(String, u64, String)> = l.iter().map(|f| (f.path.clone(), f.bytes, f.sha256.to_ascii_lowercase())).collect();
        v.sort();
        v
    };
    let (ka, kb) = (key(a), key(b));
    ka == kb && ka.windows(2).all(|w| w[0].0 != w[1].0)
}

/// Whether a file in `dir` is in use in a way that stops it from being deleted. On Windows a
/// loaded DLL (or a running exe) can't be opened for writing: such a pack is skipped, never
/// half-deleted. On Unix an open or mapped file can be unlinked, so nothing is ever in use.
fn in_use(dir: &Path) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        fn walk(d: &Path) -> bool {
            let Ok(rd) = fs::read_dir(d) else { return false };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    if walk(&p) {
                        return true;
                    }
                } else if fs::OpenOptions::new().write(true).share_mode(0).open(&p).is_err() {
                    return true;
                }
            }
            false
        }
        walk(dir)
    }
    #[cfg(not(windows))]
    {
        let _ = dir;
        false
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        err(e.to_string())
    }
}
