//! `aji-runtime use`: switches which store a copy of an app uses (its settings file), taking the
//! app's packs along without downloading them.
//!
//! The order keeps the copy working on one store or the other whatever fails:
//! 1. in the new store, under its lock: copy in the packs it lacks (hard links, or copies; staged
//!    and renamed in like any install), write the app's ref, rewrite aji-runtime.json, and write the
//!    settings file. A failure undoes the ref, aji-runtime.json and settings file; packs copied in
//!    without a ref are harmless (the next gc frees them).
//! 2. the new store's lock is released, then in the old store, under its lock: the app's ref is
//!    dropped and gc frees what no app uses. An old store that isn't the per-user one and is left
//!    empty is removed. The two locks are never held together, so two opposite `use` runs can't
//!    deadlock. A failure here leaves the copy on the new store, and is reported as a warning.

use std::path::{Path, PathBuf};

use crate::settings::{self, Resolved};
use crate::store::{self, AppRef, GcReport, Store};
use crate::{Result, copy_ref_id, err, find_ref, paths, plain_ref_for};

/// Where to switch to.
#[derive(Debug, Clone)]
pub enum Target {
    /// A folder as typed: relative to the settings file's folder (then relative mode), or absolute.
    Path(String),
    /// The per-user store.
    Shared,
}

/// The app whose packs go along.
#[derive(Debug, Clone)]
pub struct UseApp {
    pub app_id: String,
    pub per_copy: bool,
    /// The copy's executable.
    pub app_path: PathBuf,
    /// Where its libaji is: aji-runtime.json is rewritten there.
    pub aji_dir: Option<PathBuf>,
}

#[derive(Debug, Default)]
pub struct UseReport {
    /// The new store.
    pub root: PathBuf,
    /// The `store` value written to the settings file ("" for the per-user store).
    pub setting: String,
    pub relative: bool,
    /// The ref written in the new store (None: the app had no ref, or no app was given).
    pub ref_id: Option<String>,
    /// Packs copied into the new store.
    pub copied: Vec<String>,
    /// The old store's gc, after its ref was dropped.
    pub old_gc: Option<GcReport>,
    /// The old store's folder was removed (it was left empty).
    pub removed_old: bool,
    /// What couldn't be cleaned up in the old store (the copy works on the new one regardless).
    pub warnings: Vec<String>,
}

/// Whether `dir` may become a store: missing, empty, a store already, or what an emptied store
/// leaves on systems where its lock file stays.
fn usable_folder(dir: &Path) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    if !dir.is_dir() {
        return Err(err(format!("{} is a file, not a folder", dir.display())));
    }
    if dir.join("store.json").is_file() {
        return Ok(());
    }
    let names: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| err(format!("{}: {e}", dir.display())))?
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    if names.iter().all(|n| n == "lock") {
        return Ok(());
    }
    Err(err(format!("{} isn't empty and isn't a runtime store: pick a new or empty folder", dir.display())))
}

/// Files to put back as they were when a later step fails.
#[derive(Default)]
struct Undo(Vec<(PathBuf, Option<Vec<u8>>)>);

impl Undo {
    fn save(&mut self, p: PathBuf) {
        let before = std::fs::read(&p).ok();
        self.0.push((p, before));
    }
    fn restore(self) {
        for (p, before) in self.0.into_iter().rev() {
            let _ = match before {
                Some(b) => store::write_atomic(&p, &b),
                None => std::fs::remove_file(&p).or(Ok(())),
            };
        }
    }
}

/// The ref this copy has in `store`: with `per_copy` as [`find_ref`]; else the plain ref, when it
/// names this copy's path (or its app is gone). A plain ref for another copy that's there is
/// another copy's, and stays where it is.
fn ref_of(s: &Store, app: &UseApp) -> Result<Option<(String, AppRef)>> {
    if app.per_copy {
        return find_ref(s, &app.app_id, Some(&app.app_path), true);
    }
    Ok(s.read_ref(&app.app_id)
        .filter(|r| s.names_path(&r.app_path, &app.app_path) || !s.resolve_path(&r.app_path).exists())
        .map(|r| (app.app_id.clone(), r)))
}

/// Switches the copy whose settings file and current store `from` describes to `target`, taking the
/// app's packs along. `waiting` is called when a store lock is busy; `progress` gets (copied, total)
/// bytes of the packs copied.
pub fn use_store(
    from: &Resolved,
    target: &Target,
    app: Option<&UseApp>,
    waiting: &mut dyn FnMut(),
    progress: &mut dyn FnMut(u64, u64),
) -> Result<UseReport> {
    let conf = &from.settings_file;
    let (root, setting, relative) = match target {
        Target::Shared => (from.default_root.clone(), String::new(), false),
        Target::Path(p) => {
            let v = p.trim();
            if v.is_empty() {
                return Err(err("use needs a folder, or --shared"));
            }
            let root = settings::store_dir(v, conf);
            if paths::same_dir(&root, &from.default_root) {
                (from.default_root.clone(), String::new(), false)
            } else if settings::is_relative_value(v) {
                (root, v.to_string(), true)
            } else {
                (root.clone(), root.to_string_lossy().into_owned(), false)
            }
        }
    };
    if paths::is_network(&root) {
        return Err(err(format!("{}: a network folder can't hold the store; use a local drive", root.display())));
    }
    let shared = setting.is_empty();
    if !shared {
        usable_folder(&root)?;
    }
    let same = paths::same_dir(&root, &from.root);
    if !same && (paths::inside(&root, &from.root) || paths::inside(&from.root, &root)) {
        return Err(err(format!("{} is inside the current store {}, or the store is inside it", root.display(), from.root.display())));
    }
    // the settings file's folder must be writable before anything is copied
    let probe = conf.with_extension(format!("probe{}", std::process::id()));
    std::fs::write(&probe, b"").map_err(|e| err(format!("can't write the settings file {}: {e}", conf.display())))?;
    let _ = std::fs::remove_file(&probe);

    let mut report = UseReport { root: root.clone(), setting: setting.clone(), relative, ..Default::default() };
    let old = Store::open_existing(&from.root).map(|s| s.with_relative(from.relative));
    let found = match (app, &old) {
        (Some(a), Some(s)) => ref_of(s, a)?.map(|f| (a, f)),
        _ => None,
    };
    let Some((app, (old_id, old_ref))) = found else {
        settings::write(conf, &setting)?;
        return Ok(report);
    };
    let old = old.unwrap();

    // 1. the new store
    let new = if same { Store::open(&from.root)? } else { Store::open(&root)? }.with_relative(relative);
    let new_id = if app.per_copy { copy_ref_id(&new, &app.app_id, &app.app_path)? } else { app.app_id.clone() };
    let new_ref = AppRef {
        app_id: new_id.clone(),
        app_version: old_ref.app_version.clone(),
        app_path: new.path_form(&app.app_path),
        packs: old_ref.packs.clone(),
    };
    let lock = new.lock(waiting)?;
    if !same {
        let mut todo = Vec::new();
        for k in &old_ref.packs {
            let (n, v) = k.split_once('/').ok_or_else(|| err(format!("{old_id}'s ref lists {k:?}, not name/version")))?;
            let meta = old.installed(n, v).ok_or_else(|| {
                err(format!("{k} isn't complete in {}: repair the app's runtime there first. Nothing was changed.", old.root().display()))
            })?;
            if new.is_complete(n, v) {
                new.check_same_files(n, v, &meta.files)?;
            } else {
                todo.push((n, v, meta.files.iter().map(|f| f.bytes).sum::<u64>()));
            }
        }
        let total: u64 = todo.iter().map(|t| t.2).sum();
        let mut done = 0;
        progress(0, total);
        for (n, v, _) in todo {
            crate::copy_pack(&old, &new, &lock, n, v, None, &mut |b| {
                done += b;
                progress(done, total);
            })?;
            report.copied.push(store::pack_key(n, v));
        }
    }
    let mut undo = Undo::default();
    let written = (|| -> Result<()> {
        undo.save(new.ref_file(&new_id));
        new.write_ref(&lock, &new_ref)?;
        if let Some(dir) = &app.aji_dir {
            let mut kinds = Vec::new();
            for k in &new_ref.packs {
                let (n, v) = k.split_once('/').unwrap();
                let m = new.installed(n, v).ok_or_else(|| err(format!("{k} isn't complete in {}", new.root().display())))?;
                kinds.push((m.kind, crate::json_dir(&new, dir, &new.pack_dir(n, v))));
            }
            let pairs: Vec<(&str, &Path)> = kinds.iter().map(|(k, d)| (k.as_str(), d.as_path())).collect();
            undo.save(dir.join("aji-runtime.json"));
            crate::write_runtime_json_value(dir, &crate::runtime_json_of_kinds(&pairs))?;
        }
        undo.save(conf.clone());
        settings::write(conf, &setting)
    })();
    if let Err(e) = written {
        undo.restore();
        return Err(e);
    }
    report.ref_id = Some(new_id.clone());
    if same {
        // the same store in another form: the ref under its old id goes
        if old_id != new_id {
            new.remove_ref(&lock, &old_id)?;
        }
        if app.per_copy && app.app_id != new_id && plain_ref_for(&new, &app.app_id, &app.app_path).is_some() {
            new.remove_ref(&lock, &app.app_id)?;
        }
        return Ok(report);
    }
    drop(lock);

    // 2. the old store
    let r = (|| -> Result<()> {
        let lock = old.lock(waiting)?;
        old.remove_ref(&lock, &old_id)?;
        if app.per_copy && plain_ref_for(&old, &app.app_id, &app.app_path).is_some() {
            old.remove_ref(&lock, &app.app_id)?;
        }
        report.old_gc = Some(old.gc(&lock)?);
        if !from.is_default_store() {
            report.removed_old = old.remove_if_empty(lock);
        }
        Ok(())
    })();
    if let Err(e) = r {
        report.warnings.push(format!("the old store {} still has this app's ref: {e}", from.root.display()));
    }
    Ok(report)
}
