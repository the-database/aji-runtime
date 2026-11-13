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
pub mod location;
pub mod nvml;
pub mod paths;
pub mod settings;
pub mod store;
pub mod switch;

use std::collections::BTreeSet;
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
    /// Moved over from an app's old private files (`adopt_from`), or copied from the per-user store
    /// ([`Store::with_shared`]).
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
    /// Report the plan only; change nothing. Doesn't take the store lock (so it never waits for
    /// another app's download), and is complete only if the app's ref and aji-runtime.json are
    /// what an ensure would write.
    pub check: bool,
}

pub struct EnsureResult {
    /// Each pack and its folder in the store, in the order asked for.
    pub packs: Vec<(IndexPack, PathBuf)>,
    pub downloaded_bytes: u64,
    pub adopted: Vec<String>,
    pub gc: GcReport,
    /// False after `check` when something would have to be installed, or `stale` isn't empty.
    pub complete: bool,
    /// After `check`: what else an ensure would put right (the app's ref is missing, lists other
    /// packs or names another app path; aji-runtime.json is missing or different).
    pub stale: Vec<String>,
}

/// Whether the store holds `p`: a complete pack of its name and version whose `.pack.json` lists
/// exactly the index's files (paths, sizes, SHA-256s). The archive's SHA-256 doesn't matter: a later
/// runtime release may repack the same files, and they're still the same pack. A complete pack with
/// other files is an error, since pack folders are never replaced ([`Store::commit_pack`]).
fn have(store: &Store, p: &IndexPack) -> Result<bool> {
    if !store.is_complete(&p.name, &p.version) {
        return Ok(false);
    }
    store.check_same_files(&p.name, &p.version, &p.files)?;
    Ok(true)
}

/// The ref id of one copy of an app (`--per-copy`): `<app_id>-<8 hex>`, the start of the SHA-256 of
/// the copy's app path in the form the store's refs keep it ([`Store::path_form`], normalized by
/// [`paths::path_key`]). In relative mode that's relative to the store, so it survives moving the
/// whole folder.
pub fn copy_ref_id(store: &Store, app_id: &str, app_path: &Path) -> Result<String> {
    store::check_component("app id", app_id)?;
    let key = paths::path_key(&store.path_form(app_path));
    let id = format!("{app_id}-{}", &sha256_bytes(key.as_bytes())[..8]);
    store::check_component("app id", &id)?;
    Ok(id)
}

/// The ref an app (or, with `per_copy`, this copy of it at `app_path`) has, and its id. With
/// `per_copy`, a plain `app_id` ref that names this same app path counts too (made before the app
/// used per-copy refs); one naming another path never does.
pub fn find_ref(store: &Store, app_id: &str, app_path: Option<&Path>, per_copy: bool) -> Result<Option<(String, AppRef)>> {
    store::check_component("app id", app_id)?;
    if !per_copy {
        return Ok(store.read_ref(app_id).map(|r| (app_id.to_string(), r)));
    }
    let p = app_path.ok_or_else(|| err("--per-copy needs --app-path"))?;
    let id = copy_ref_id(store, app_id, p)?;
    if let Some(r) = store.read_ref(&id) {
        return Ok(Some((id, r)));
    }
    Ok(plain_ref_for(store, app_id, p).map(|r| (app_id.to_string(), r)))
}

/// The plain `app_id` ref if it names `app_path`.
pub(crate) fn plain_ref_for(store: &Store, app_id: &str, app_path: &Path) -> Option<AppRef> {
    store.read_ref(app_id).filter(|r| store.names_path(&r.app_path, app_path))
}

/// Installs the app's packs, writes its ref (exactly these packs) and aji-runtime.json, then
/// frees what no app uses any more. A failure leaves the store consistent and the app's old ref in
/// place.
pub fn ensure(store: &Store, opts: &EnsureOptions, on: &mut dyn FnMut(Event)) -> Result<EnsureResult> {
    ensure_as(store, opts, false, on)
}

/// [`ensure`] for one copy of an app that can have several on one PC: the ref is
/// [`copy_ref_id`]'s (`opts.app_id` is the app's plain id). A plain `app_id` ref naming this same
/// app path is removed once the per-copy ref is written; one naming another path is left alone.
pub fn ensure_per_copy(store: &Store, opts: &EnsureOptions, on: &mut dyn FnMut(Event)) -> Result<EnsureResult> {
    ensure_as(store, opts, true, on)
}

fn ensure_as(store: &Store, opts: &EnsureOptions, per_copy: bool, on: &mut dyn FnMut(Event)) -> Result<EnsureResult> {
    store::check_component("app id", &opts.app_id)?;
    // A ref is live only while its app_path exists: a wrong one would have this very run's gc free
    // everything it just installed.
    if !opts.app_path.exists() {
        return Err(err(format!("--app-path {} does not exist (the app's executable)", opts.app_path.display())));
    }
    let ref_id = if per_copy { copy_ref_id(store, &opts.app_id, &opts.app_path)? } else { opts.app_id.clone() };
    if opts.check {
        return check(store, opts, &ref_id, per_copy, on);
    }
    let shared = shared_store(store);
    let lock = store.lock(&mut || on(Event::Waiting))?;
    if store.may_delete() {
        store.clean_tmp(&lock);
    }
    let mut adopted = Vec::new();
    let mut todo = Vec::new();
    // A pack installed with other files stops everything before anything is changed or downloaded.
    let mut present = Vec::with_capacity(opts.packs.len());
    for p in &opts.packs {
        present.push(have(store, p)?);
    }
    for (p, present) in opts.packs.iter().zip(present) {
        let state = if present {
            PlanState::Installed
        } else if shared.as_ref().is_some_and(|s| shared_has(s, p) && take_shared(store, &lock, s, p))
            || (!opts.adopt_from.is_empty() && adopt(store, &lock, p, &opts.adopt_from)?)
        {
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
    for p in &todo {
        install(store, &lock, p, opts.mirror.as_deref(), on)?;
    }
    store.write_ref(
        &lock,
        &AppRef {
            app_id: ref_id.clone(),
            app_version: opts.app_version.clone(),
            app_path: store.path_form(&opts.app_path),
            packs: opts.packs.iter().map(IndexPack::key).collect(),
        },
    )?;
    // the plain ref this copy had before it used a per-copy one
    if per_copy && plain_ref_for(store, &opts.app_id, &opts.app_path).is_some() {
        store.remove_ref(&lock, &opts.app_id)?;
    }
    let packs: Vec<(IndexPack, PathBuf)> =
        opts.packs.iter().map(|p| (p.clone(), store.pack_dir(&p.name, &p.version))).collect();
    if let Some(dir) = &opts.aji_dir {
        write_aji_runtime_json(dir, &json_form(store, dir, &packs))?;
    }
    let gc = store.gc(&lock)?;
    on(Event::Done);
    Ok(EnsureResult { packs, downloaded_bytes: total, adopted, gc, complete: true, stale: Vec::new() })
}

/// `ensure --check`: the plan, changing nothing and without the store lock, so an updater's check
/// never waits for another app's long download. That's safe because the store changes only by
/// renames (a pack folder in or out whole, refs and aji-runtime.json replaced whole): this sees each
/// pack and ref as it was before a change or after it. A pack that another app's gc renames out
/// while this looks at it reads as not installed.
///
/// Complete only when nothing would be installed and the app's ref lists exactly these packs for
/// this `app_path` (and, with `aji_dir`, aji-runtime.json is what an ensure writes). A ref that names
/// the app's old path (a moved or renamed AppImage) is dead for gc, which then frees the packs: such
/// a check is incomplete, so the updater's ensure writes the ref again.
fn check(store: &Store, opts: &EnsureOptions, ref_id: &str, per_copy: bool, on: &mut dyn FnMut(Event)) -> Result<EnsureResult> {
    let shared = shared_store(store);
    let mut present = Vec::with_capacity(opts.packs.len());
    for p in &opts.packs {
        present.push(match have(store, p) {
            Ok(b) => b,
            // gone meanwhile (another app's gc): not installed
            Err(_) if !store.is_complete(&p.name, &p.version) => false,
            Err(e) => return Err(e),
        });
    }
    let (mut total, mut todo) = (0, false);
    for (p, present) in opts.packs.iter().zip(present) {
        // an adoption is planned without doing it: the old files are there with the pinned SHA-256s
        let state = if present {
            PlanState::Installed
        } else if shared.as_ref().is_some_and(|s| shared_has(s, p))
            || (!opts.adopt_from.is_empty() && adoptable(p, &opts.adopt_from).is_some())
        {
            todo = true;
            PlanState::Adopted
        } else {
            todo = true;
            total += p.bytes;
            PlanState::Download
        };
        on(Event::Plan { pack: p, state });
    }
    on(Event::Total(total));
    let packs: Vec<(IndexPack, PathBuf)> =
        opts.packs.iter().map(|p| (p.clone(), store.pack_dir(&p.name, &p.version))).collect();
    let stale = stale(store, opts, ref_id, per_copy, &packs);
    Ok(EnsureResult {
        packs,
        downloaded_bytes: 0,
        adopted: Vec::new(),
        gc: GcReport::default(),
        complete: !todo && stale.is_empty(),
        stale,
    })
}

/// What an ensure would write differently from what's there: the app's ref (it must list exactly
/// these packs and name this app_path in the form this store writes it), aji-runtime.json, and with
/// `per_copy` the plain ref this copy had before.
fn stale(store: &Store, opts: &EnsureOptions, ref_id: &str, per_copy: bool, packs: &[(IndexPack, PathBuf)]) -> Vec<String> {
    let mut out = Vec::new();
    match store.read_ref(ref_id) {
        None => out.push(format!("{ref_id} has no ref in the store")),
        Some(r) => {
            let want: Vec<String> = opts.packs.iter().map(IndexPack::key).collect();
            let have: BTreeSet<&str> = r.packs.iter().map(String::as_str).collect();
            if have != want.iter().map(String::as_str).collect() {
                out.push(format!("{ref_id}'s ref lists other packs ({})", r.packs.join(", ")));
            }
            if !same_form(&r.app_path, &store.path_form(&opts.app_path)) {
                out.push(format!("{ref_id}'s ref names another app path ({})", r.app_path.display()));
            }
        }
    }
    if per_copy && plain_ref_for(store, &opts.app_id, &opts.app_path).is_some() {
        out.push(format!("{}'s plain ref for this copy is still there", opts.app_id));
    }
    if let Some(dir) = &opts.aji_dir {
        let file = dir.join("aji-runtime.json");
        let current = std::fs::read(&file).ok().and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
        if current.as_ref() != Some(&aji_runtime_json(&json_form(store, dir, packs))) {
            out.push(format!("{} is missing or names other packs", file.display()));
        }
    }
    out
}

/// Whether a ref's app path is `want` in the same form: both absolute and the same file, or both
/// relative and the same relative path (case-insensitively on Windows).
fn same_form(have: &Path, want: &Path) -> bool {
    if have.is_absolute() != want.is_absolute() {
        return false;
    }
    if want.is_absolute() { paths::same_path(have, want) } else { paths::path_key(have) == paths::path_key(want) }
}

/// The pack folders as aji-runtime.json in `aji_dir` names them: as they are, or in relative mode
/// relative to `aji_dir` (absolute when there's no relative path, on another drive).
pub fn json_form(store: &Store, aji_dir: &Path, packs: &[(IndexPack, PathBuf)]) -> Vec<(IndexPack, PathBuf)> {
    packs.iter().map(|(p, d)| (p.clone(), json_dir(store, aji_dir, d))).collect()
}

/// One pack folder as aji-runtime.json in `aji_dir` names it ([`json_form`]).
pub fn json_dir(store: &Store, aji_dir: &Path, dir: &Path) -> PathBuf {
    if store.is_relative() {
        paths::relative_to(aji_dir, dir).unwrap_or_else(|| dir.to_path_buf())
    } else {
        dir.to_path_buf()
    }
}

/// The per-user store to take packs from, if this store has one ([`Store::with_shared`]) and it's
/// another store.
fn shared_store(store: &Store) -> Option<Store> {
    let root = store.shared_root()?;
    if paths::same_dir(root, store.root()) {
        return None;
    }
    Store::open_existing(root)
}

/// Whether `shared` holds `p` complete, with the index's files (as [`have`] checks).
fn shared_has(shared: &Store, p: &IndexPack) -> bool {
    shared.installed(&p.name, &p.version).is_some_and(|m| store::same_files(&m.files, &p.files))
}

/// Copies `p` in from the per-user store; false (and nothing changed) on any error, to download it
/// instead.
fn take_shared(store: &Store, lock: &StoreLock, shared: &Store, p: &IndexPack) -> bool {
    copy_pack(shared, store, lock, &p.name, &p.version, Some(&p.files), &mut |_| {}).is_ok()
}

/// Copies the complete pack `name/version` from the store `from` into `to` (holding `to`'s lock,
/// never `from`'s): each file hard-linked, or copied when that fails (another drive), staged in
/// `to`'s tmp/ and renamed in, with the `.pack.json` written last. The copy is checked against the
/// `.pack.json` (the file count and sizes), and with `expect`, that lists exactly those files.
/// `from` is only read. `progress` gets each file's bytes as it's done.
pub fn copy_pack(
    from: &Store,
    to: &Store,
    lock: &StoreLock,
    name: &str,
    version: &str,
    expect: Option<&[PackFile]>,
    progress: &mut dyn FnMut(u64),
) -> Result<PathBuf> {
    store::check_component("pack name", name)?;
    store::check_component("pack version", version)?;
    let key = store::pack_key(name, version);
    let meta = from.installed(name, version).ok_or_else(|| err(format!("{key} isn't complete in {}", from.root().display())))?;
    if expect.is_some_and(|e| !store::same_files(&meta.files, e)) {
        return Err(err(format!("{key} in {} has other files", from.root().display())));
    }
    let src_dir = from.pack_dir(name, version);
    let work = to.tmp_folder(lock, &format!("copy-{name}"))?;
    let result = (|| {
        let staged = work.join("pack");
        std::fs::create_dir_all(&staged).map_err(|e| err(format!("{}: {e}", staged.display())))?;
        for f in &meta.files {
            index::check_rel_path(&f.path)?;
            let (src, dst) = (src_dir.join(&f.path), staged.join(&f.path));
            std::fs::create_dir_all(dst.parent().unwrap()).map_err(|e| err(format!("{}: {e}", dst.display())))?;
            if std::fs::hard_link(&src, &dst).is_err() {
                std::fs::copy(&src, &dst).map_err(|e| err(format!("copying {} to {}: {e}", src.display(), dst.display())))?;
            }
            progress(f.bytes);
        }
        let mut found = Vec::new();
        list_files(&staged, &staged, &mut found);
        let sizes_ok = meta.files.iter().all(|f| std::fs::metadata(staged.join(&f.path)).is_ok_and(|m| m.len() == f.bytes));
        if found.len() != meta.files.len() || !sizes_ok {
            return Err(err(format!("the copy of {key} doesn't match its .pack.json")));
        }
        store::write_atomic(&staged.join(".pack.json"), &serde_json::to_vec_pretty(&meta).unwrap())?;
        to.commit_pack(lock, &staged, name, version)
    })();
    let _ = std::fs::remove_dir_all(&work);
    result
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
    let Some(sources) = adoptable(p, from) else {
        return Ok(false);
    };
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

/// The old private file for each of `p`'s files, if all of them are there with the pinned SHA-256s
/// (in one of `from` or its `models/`).
pub fn adoptable(p: &IndexPack, from: &[PathBuf]) -> Option<Vec<PathBuf>> {
    p.files
        .iter()
        .map(|f| {
            from.iter().flat_map(|d| [d.join(&f.path), d.join("models").join(&f.path)]).find(|c| {
                std::fs::metadata(c).is_ok_and(|m| m.is_file() && m.len() == f.bytes)
                    && sha256_file(c).is_ok_and(|h| h == f.sha256)
            })
        })
        .collect()
}

/// aji-runtime.json beside libaji (animejanai-inference `core/runtime_packs.h`): the TensorRT,
/// builder-resource and TensorRT-RTX pack folders to load from, and the app's model pack folders.
/// A relative folder (relative mode) is relative to the file's folder.
pub fn aji_runtime_json(packs: &[(IndexPack, PathBuf)]) -> serde_json::Value {
    let kinds: Vec<(&str, &Path)> = packs.iter().map(|(p, d)| (p.kind.as_str(), d.as_path())).collect();
    runtime_json_of_kinds(&kinds)
}

/// [`aji_runtime_json`] from (kind, folder) pairs.
pub fn runtime_json_of_kinds(packs: &[(&str, &Path)]) -> serde_json::Value {
    let s = |p: &Path| serde_json::Value::String(p.to_string_lossy().into_owned());
    let mut v = serde_json::json!({ "format": 1 });
    if let Some((_, d)) = packs.iter().find(|(k, _)| *k == "trt") {
        v["trt"] = s(d);
    }
    let builders: Vec<serde_json::Value> = packs.iter().filter(|(k, _)| *k == "trt-builder").map(|(_, d)| s(d)).collect();
    if !builders.is_empty() {
        v["trt_builder"] = serde_json::Value::Array(builders);
    }
    if let Some((_, d)) = packs.iter().find(|(k, _)| *k == "trtrtx") {
        v["trtrtx"] = s(d);
    }
    // For the app, not libaji (which skips keys it doesn't know): its model pack folders, so an app
    // that only knows its libaji folder (yosh's add-on) finds the models too.
    let models: Vec<serde_json::Value> = packs.iter().filter(|(k, _)| *k == "models").map(|(_, d)| s(d)).collect();
    if !models.is_empty() {
        v["models"] = serde_json::Value::Array(models);
    }
    v
}

pub fn write_aji_runtime_json(aji_dir: &Path, packs: &[(IndexPack, PathBuf)]) -> Result<()> {
    write_runtime_json_value(aji_dir, &aji_runtime_json(packs))
}

pub(crate) fn write_runtime_json_value(aji_dir: &Path, v: &serde_json::Value) -> Result<()> {
    let mut b = serde_json::to_vec_pretty(v).unwrap();
    b.push(b'\n');
    store::write_atomic(&aji_dir.join("aji-runtime.json"), &b)
}

/// How long [`unref`] waits for another app's install: less than an install waits, since an
/// uninstaller is blocked meanwhile, and giving up is safe (the ref is dead once the app's files are
/// gone, and the next gc frees its packs).
pub const UNREF_LOCK_WAIT: std::time::Duration = std::time::Duration::from_secs(5 * 60);

/// Drops an app's ref (its uninstaller calls this), then frees what no app uses. Waits up to
/// [`UNREF_LOCK_WAIT`] for the store lock.
pub fn unref(store: &Store, app_id: &str, on_wait: &mut dyn FnMut()) -> Result<(bool, GcReport)> {
    let lock = store.lock_within(UNREF_LOCK_WAIT, on_wait)?;
    let had = store.remove_ref(&lock, app_id)?;
    let gc = store.gc(&lock)?;
    Ok((had, gc))
}

/// [`unref`] that drops the plain `app_id` ref only when it names `app_path` (this copy's exe,
/// which needn't exist any more; a relative ref path is resolved against the store root), then gc.
/// A ref naming another path is another copy's (an older portable copy that took the plain id
/// over) and is left as it is: returns `(false, Some(its path), gc)`. Otherwise `(had, None, gc)`.
pub fn unref_for_path(
    store: &Store,
    app_id: &str,
    app_path: &Path,
    on_wait: &mut dyn FnMut(),
) -> Result<(bool, Option<PathBuf>, GcReport)> {
    store::check_component("app id", app_id)?;
    let lock = store.lock_within(UNREF_LOCK_WAIT, on_wait)?;
    let (had, other) = match store.read_ref(app_id) {
        Some(r) if !store.names_path(&r.app_path, app_path) => (false, Some(paths::absolute(&store.resolve_path(&r.app_path)))),
        // this copy's ref; or none, or an unreadable one, dropped as `unref` does (gc frees it anyway)
        _ => (store.remove_ref(&lock, app_id)?, None),
    };
    let gc = store.gc(&lock)?;
    Ok((had, other, gc))
}

/// [`unref`] for one copy of an app ([`ensure_per_copy`]): drops its per-copy ref, and the plain
/// `app_id` ref if that names this same app path (never one naming another path), then gc.
/// `app_path` needn't exist any more.
pub fn unref_per_copy(store: &Store, app_id: &str, app_path: &Path, on_wait: &mut dyn FnMut()) -> Result<(bool, GcReport)> {
    let id = copy_ref_id(store, app_id, app_path)?;
    let lock = store.lock_within(UNREF_LOCK_WAIT, on_wait)?;
    let mut had = store.remove_ref(&lock, &id)?;
    if plain_ref_for(store, app_id, app_path).is_some() {
        had |= store.remove_ref(&lock, app_id)?;
    }
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
