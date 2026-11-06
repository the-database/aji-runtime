//! aji-runtime: the per-user store of runtime and model packs that libaji apps share.
//!
//! Each pack (a TensorRT runtime, one GPU generation's builder resources, the TensorRT-RTX
//! runtime, a model set) is installed once per user at an exact version, and every installed app
//! records the packs it uses in its own ref. A pack is deleted when no installed app lists it.
//! Design: plans/shared-runtime-packs.md in the-database/animejan.ai; layout and rules: [`store`].
//!
//! The library is what MangaJaNaiCLI's `setup` uses; the `aji-runtime` program wraps it for the
//! other apps' installers (`ensure`, `adopt`, `unref`, `gc`, `list`, `path`).

pub mod fetch;
pub mod index;
pub mod nvml;
pub mod store;

use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

pub use index::{Index, IndexPack};
pub use store::{AppRef, GcReport, PackFile, PackMeta, Store, StoreLock};

#[derive(Debug, Clone)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn err(s: impl Into<String>) -> Error {
    Error(s.into())
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_bytes(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path).map_err(|e| err(format!("{}: {e}", path.display())))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| err(format!("{}: {e}", path.display())))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// How a pack will be provided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanState {
    /// Already in the store.
    Installed,
    /// Moved over from an app's old private files (`adopt_from`).
    Adopted,
    /// To download.
    Download,
}

/// Progress of [`ensure`].
pub enum Event<'a> {
    /// Another installer holds the store lock; waiting for it.
    Waiting,
    Plan { pack: &'a IndexPack, state: PlanState },
    /// Bytes that will be downloaded in all.
    Total(u64),
    Download { pack: &'a IndexPack, got: u64, total: u64 },
    Extract { pack: &'a IndexPack },
    Done,
}

pub struct EnsureOptions {
    pub app_id: String,
    pub app_version: String,
    /// The app's executable: its ref is live while this exists.
    pub app_path: PathBuf,
    /// The packs this app version uses ([`Index::select`]).
    pub packs: Vec<IndexPack>,
    /// A folder holding the archives by asset name, used instead of downloading (testing).
    pub mirror: Option<PathBuf>,
    /// Old private install folders whose files may be taken over instead of downloading them
    /// (each folder and its `models/` are searched; files are matched by SHA-256 and hard-linked,
    /// or copied across drives, so the old install keeps working until the app removes them).
    pub adopt_from: Vec<PathBuf>,
    /// Where the app's libaji is: aji-runtime.json is written there.
    pub aji_dir: Option<PathBuf>,
    /// Report the plan only; change nothing.
    pub check: bool,
}

pub struct EnsureResult {
    /// Each pack and its folder in the store, in the order asked for.
    pub packs: Vec<(IndexPack, PathBuf)>,
    pub downloaded_bytes: u64,
    pub adopted: Vec<String>,
    pub gc: GcReport,
    /// False after `check` when something would have to be installed.
    pub complete: bool,
}

/// Whether the store holds `p` exactly as the index pins it.
fn have(store: &Store, p: &IndexPack) -> bool {
    store.installed(&p.name, &p.version).is_some_and(|m| m.sha256 == p.sha256)
}

/// Installs the app's packs, writes its ref (exactly these packs) and aji-runtime.json, then
/// frees what no app uses any more. A failure leaves the store consistent and the app's old ref in
/// place.
pub fn ensure(store: &Store, opts: &EnsureOptions, on: &mut dyn FnMut(Event)) -> Result<EnsureResult> {
    store::check_component("app id", &opts.app_id)?;
    let lock = store.lock(&mut || on(Event::Waiting))?;
    if store.may_delete() {
        store.clean_tmp(&lock);
    }
    let mut adopted = Vec::new();
    let mut todo = Vec::new();
    for p in &opts.packs {
        let state = if have(store, p) {
            PlanState::Installed
        } else if !opts.check && !opts.adopt_from.is_empty() && adopt(store, &lock, p, &opts.adopt_from)? {
            adopted.push(p.key());
            PlanState::Adopted
        } else {
            todo.push(p);
            PlanState::Download
        };
        on(Event::Plan { pack: p, state });
    }
    let total: u64 = todo.iter().map(|p| p.bytes).sum();
    on(Event::Total(total));
    if opts.check {
        return Ok(EnsureResult {
            packs: opts.packs.iter().map(|p| (p.clone(), store.pack_dir(&p.name, &p.version))).collect(),
            downloaded_bytes: 0,
            adopted,
            gc: GcReport::default(),
            complete: todo.is_empty(),
        });
    }
    for p in &todo {
        install(store, &lock, p, opts.mirror.as_deref(), on)?;
    }
    store.write_ref(
        &lock,
        &AppRef {
            app_id: opts.app_id.clone(),
            app_version: opts.app_version.clone(),
            app_path: opts.app_path.clone(),
            packs: opts.packs.iter().map(IndexPack::key).collect(),
        },
    )?;
    let packs: Vec<(IndexPack, PathBuf)> =
        opts.packs.iter().map(|p| (p.clone(), store.pack_dir(&p.name, &p.version))).collect();
    if let Some(dir) = &opts.aji_dir {
        write_aji_runtime_json(dir, &packs)?;
    }
    let gc = store.gc(&lock)?;
    on(Event::Done);
    Ok(EnsureResult { packs, downloaded_bytes: total, adopted, gc, complete: true })
}

/// Downloads (or copies from `mirror`), verifies and installs one pack.
pub fn install(store: &Store, lock: &StoreLock, p: &IndexPack, mirror: Option<&Path>, on: &mut dyn FnMut(Event)) -> Result<PathBuf> {
    let work = store.tmp_folder(lock, &format!("dl-{}", p.name))?;
    let result = (|| {
        let archive = work.join(&p.asset);
        match mirror {
            Some(m) => {
                let src = m.join(&p.asset);
                std::fs::copy(&src, &archive).map_err(|e| err(format!("{}: {e}", src.display())))?;
                on(Event::Download { pack: p, got: p.bytes, total: p.bytes });
            }
            None => fetch::download(&p.url, &archive, p.bytes, &mut |got, total| on(Event::Download { pack: p, got, total }))?,
        }
        let got = sha256_file(&archive)?;
        if got != p.sha256 {
            return Err(err(format!("{} failed its SHA-256 check (got {got}, want {})", p.asset, p.sha256)));
        }
        on(Event::Extract { pack: p });
        let staged = work.join("pack");
        fetch::extract(&archive, &staged)?;
        let _ = std::fs::remove_file(&archive);
        verify_files(&staged, p)?;
        write_pack_meta(&staged, p, &p.url)?;
        store.commit_pack(lock, &staged, &p.name, &p.version)
    })();
    let _ = std::fs::remove_dir_all(&work);
    result
}

/// Checks that `dir` holds exactly the pack's files, with their SHA-256s.
fn verify_files(dir: &Path, p: &IndexPack) -> Result<()> {
    let mut found = Vec::new();
    list_files(dir, dir, &mut found);
    found.sort();
    let mut want: Vec<String> = p.files.iter().map(|f| f.path.clone()).collect();
    want.sort();
    if found != want {
        return Err(err(format!("{}: the archive's files don't match the index", p.asset)));
    }
    for f in &p.files {
        let path = dir.join(&f.path);
        let got = sha256_file(&path)?;
        if got != f.sha256 {
            return Err(err(format!("{}: {} failed its SHA-256 check", p.asset, f.path)));
        }
    }
    Ok(())
}

fn list_files(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            list_files(root, &p, out);
        } else if let Ok(rel) = p.strip_prefix(root) {
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
}

fn write_pack_meta(dir: &Path, p: &IndexPack, source: &str) -> Result<()> {
    let meta = PackMeta {
        format: store::FORMAT,
        name: p.name.clone(),
        version: p.version.clone(),
        kind: p.kind.clone(),
        platform: index::platform().into(),
        source: source.into(),
        sha256: p.sha256.clone(),
        files: p.files.clone(),
    };
    store::write_atomic(&dir.join(".pack.json"), &serde_json::to_vec_pretty(&meta).unwrap())
}

/// Installs `p` from an app's old private files if all of them are there with the pinned SHA-256s
/// (in one of `from` or its `models/`). Hard-links them (copies across drives), so the old files
/// stay until the app deletes them. Returns whether it did.
pub fn adopt(store: &Store, lock: &StoreLock, p: &IndexPack, from: &[PathBuf]) -> Result<bool> {
    let mut sources = Vec::new();
    for f in &p.files {
        let found = from.iter().flat_map(|d| [d.join(&f.path), d.join("models").join(&f.path)]).find(|c| {
            std::fs::metadata(c).is_ok_and(|m| m.is_file() && m.len() == f.bytes)
                && sha256_file(c).is_ok_and(|h| h == f.sha256)
        });
        match found {
            Some(c) => sources.push(c),
            None => return Ok(false),
        }
    }
    let work = store.tmp_folder(lock, &format!("adopt-{}", p.name))?;
    let result = (|| {
        let staged = work.join("pack");
        for (f, src) in p.files.iter().zip(&sources) {
            let dst = staged.join(&f.path);
            std::fs::create_dir_all(dst.parent().unwrap()).map_err(|e| err(format!("{}: {e}", dst.display())))?;
            if std::fs::hard_link(src, &dst).is_err() {
                std::fs::copy(src, &dst).map_err(|e| err(format!("{}: {e}", src.display())))?;
                if sha256_file(&dst)? != f.sha256 {
                    return Err(err(format!("{}: copy failed its SHA-256 check", dst.display())));
                }
            }
        }
        let from_desc = from.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", ");
        write_pack_meta(&staged, p, &format!("adopted from {from_desc}"))?;
        store.commit_pack(lock, &staged, &p.name, &p.version)
    })();
    let _ = std::fs::remove_dir_all(&work);
    result.map(|_| true)
}

/// aji-runtime.json beside libaji (animejanai-inference `core/runtime_packs.h`): the TensorRT,
/// builder-resource and TensorRT-RTX pack folders to load from.
pub fn aji_runtime_json(packs: &[(IndexPack, PathBuf)]) -> serde_json::Value {
    let s = |p: &PathBuf| serde_json::Value::String(p.to_string_lossy().into_owned());
    let mut v = serde_json::json!({ "format": 1 });
    if let Some((_, d)) = packs.iter().find(|(p, _)| p.kind == "trt") {
        v["trt"] = s(d);
    }
    let builders: Vec<serde_json::Value> = packs.iter().filter(|(p, _)| p.kind == "trt-builder").map(|(_, d)| s(d)).collect();
    if !builders.is_empty() {
        v["trt_builder"] = serde_json::Value::Array(builders);
    }
    if let Some((_, d)) = packs.iter().find(|(p, _)| p.kind == "trtrtx") {
        v["trtrtx"] = s(d);
    }
    v
}

pub fn write_aji_runtime_json(aji_dir: &Path, packs: &[(IndexPack, PathBuf)]) -> Result<()> {
    let mut b = serde_json::to_vec_pretty(&aji_runtime_json(packs)).unwrap();
    b.push(b'\n');
    store::write_atomic(&aji_dir.join("aji-runtime.json"), &b)
}

/// Drops an app's ref (its uninstaller calls this), then frees what no app uses.
pub fn unref(store: &Store, app_id: &str, on_wait: &mut dyn FnMut()) -> Result<(bool, GcReport)> {
    let lock = store.lock(on_wait)?;
    let had = store.remove_ref(&lock, app_id)?;
    let gc = store.gc(&lock)?;
    Ok((had, gc))
}

/// The installed packs an app's ref lists, with their folders (e.g. to find its model folders).
pub fn app_packs(store: &Store, app_id: &str) -> Vec<(PackMeta, PathBuf)> {
    let Some(r) = store.read_ref(app_id) else { return Vec::new() };
    r.packs
        .iter()
        .filter_map(|k| {
            let (name, version) = k.split_once('/')?;
            let m = store.installed(name, version)?;
            Some((m, store.pack_dir(name, version)))
        })
        .collect()
}

#[cfg(test)]
mod tests;
