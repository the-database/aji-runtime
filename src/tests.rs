//! Store tests against temporary stores. A gc bug deletes runtimes another app needs, breaking an
//! app the user never touched, so every rule has a test here.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::nvml::Gpu;
use crate::store::{AppRef, Store, pack_key};
use crate::*;

struct Tmp(PathBuf);

impl Tmp {
    fn new(name: &str) -> Tmp {
        let p = std::env::temp_dir().join(format!("aji-runtime-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Tmp(p)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A fake app executable (refs are live while it exists).
fn app(t: &Tmp, name: &str) -> PathBuf {
    let p = t.path().join("apps").join(name).join("app.exe");
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(&p, b"exe").unwrap();
    p
}

/// Builds a .zip pack in `mirror` and its index entry.
fn make_pack(mirror: &Path, name: &str, version: &str, kind: &str, sm: Option<&str>, files: &[(&str, &[u8])]) -> IndexPack {
    std::fs::create_dir_all(mirror).unwrap();
    let asset = format!("{name}-{version}.zip");
    let path = mirror.join(&asset);
    let mut z = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
    for (n, data) in files {
        z.start_file(*n, zip::write::SimpleFileOptions::default()).unwrap();
        z.write_all(data).unwrap();
    }
    z.finish().unwrap();
    IndexPack {
        name: name.into(),
        version: version.into(),
        kind: kind.into(),
        sm: sm.map(String::from),
        asset: asset.clone(),
        url: format!("https://invalid.example/{asset}"),
        sha256: sha256_file(&path).unwrap(),
        bytes: std::fs::metadata(&path).unwrap().len(),
        installed_bytes: files.iter().map(|(_, d)| d.len() as u64).sum(),
        files: files
            .iter()
            .map(|(n, d)| PackFile { path: n.to_string(), sha256: sha256_bytes(d), bytes: d.len() as u64 })
            .collect(),
    }
}

fn opts(app_id: &str, app_path: &Path, packs: Vec<IndexPack>, mirror: &Path) -> EnsureOptions {
    EnsureOptions {
        app_id: app_id.into(),
        app_version: "1.0".into(),
        app_path: app_path.to_path_buf(),
        packs,
        mirror: Some(mirror.to_path_buf()),
        adopt_from: Vec::new(),
        aji_dir: None,
        check: false,
    }
}

fn run(store: &Store, o: &EnsureOptions) -> Result<EnsureResult> {
    ensure(store, o, &mut |_| {})
}

fn installed(store: &Store) -> Vec<String> {
    store.packs().into_iter().map(|(m, _, _)| pack_key(&m.name, &m.version)).collect()
}

struct Fixture {
    t: Tmp,
    store: Store,
    mirror: PathBuf,
    rt1: IndexPack,
    rt2: IndexPack,
    sm120: IndexPack,
    models: IndexPack,
}

fn fixture(name: &str) -> Fixture {
    let t = Tmp::new(name);
    let store = Store::open(&t.path().join("store")).unwrap();
    let mirror = t.path().join("mirror");
    let rt1 = make_pack(&mirror, "trt-runtime", "11.4", "trt", None, &[("nvinfer_11.dll", b"nvinfer 11.4"), ("trtexec.exe", b"trtexec")]);
    let rt2 = make_pack(&mirror, "trt-runtime", "11.5", "trt", None, &[("nvinfer_11.dll", b"nvinfer 11.5"), ("trtexec.exe", b"trtexec 2")]);
    let sm120 = make_pack(&mirror, "trt-sm120", "11.4", "trt-builder", Some("sm120"), &[("builder_sm120.dll", b"sm120")]);
    let models = make_pack(&mirror, "models-mangajanai", "1.0.0", "models", None, &[("a.onnx", b"model a"), ("b.onnx", b"model b")]);
    Fixture { t, store, mirror, rt1, rt2, sm120, models }
}

#[test]
fn ensure_installs_writes_ref_and_runtime_json() {
    let f = fixture("ensure");
    let exe = app(&f.t, "gui");
    let aji = f.t.path().join("apps/gui/inference");
    std::fs::create_dir_all(&aji).unwrap();
    let mut o = opts("mangajanai-gui", &exe, vec![f.rt1.clone(), f.sm120.clone(), f.models.clone()], &f.mirror);
    o.aji_dir = Some(aji.clone());
    let r = run(&f.store, &o).unwrap();
    assert!(r.complete);
    assert_eq!(installed(&f.store), vec!["models-mangajanai/1.0.0", "trt-runtime/11.4", "trt-sm120/11.4"]);
    let dir = f.store.pack_dir("trt-runtime", "11.4");
    assert_eq!(std::fs::read(dir.join("nvinfer_11.dll")).unwrap(), b"nvinfer 11.4");
    let meta = f.store.installed("trt-runtime", "11.4").unwrap();
    assert_eq!(meta.sha256, f.rt1.sha256);
    let r = f.store.read_ref("mangajanai-gui").unwrap();
    assert_eq!(r.packs, vec!["trt-runtime/11.4", "trt-sm120/11.4", "models-mangajanai/1.0.0"]);
    let j: serde_json::Value = serde_json::from_slice(&std::fs::read(aji.join("aji-runtime.json")).unwrap()).unwrap();
    assert_eq!(j["format"], 1);
    assert_eq!(j["trt"], dir.to_string_lossy().as_ref());
    assert_eq!(j["trt_builder"][0], f.store.pack_dir("trt-sm120", "11.4").to_string_lossy().as_ref());
    assert!(j.get("trtrtx").is_none());
    assert_eq!(j["models"][0], f.store.pack_dir("models-mangajanai", "1.0.0").to_string_lossy().as_ref());
    // app_packs finds the model folder through the ref
    let models: Vec<PathBuf> =
        app_packs(&f.store, "mangajanai-gui").into_iter().filter(|(m, _)| m.kind == "models").map(|(_, d)| d).collect();
    assert_eq!(models, vec![f.store.pack_dir("models-mangajanai", "1.0.0")]);
    // nothing left in tmp
    assert_eq!(std::fs::read_dir(f.store.tmp_dir()).unwrap().count(), 0);
}

#[test]
fn second_app_reuses_installed_packs() {
    let f = fixture("reuse");
    let (a, b) = (app(&f.t, "a"), app(&f.t, "b"));
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone(), f.models.clone()], &f.mirror)).unwrap();
    // B installs with an empty mirror: everything must come from the store
    let empty = f.t.path().join("empty-mirror");
    std::fs::create_dir_all(&empty).unwrap();
    let mut states = Vec::new();
    ensure(&f.store, &opts("app-b", &b, vec![f.rt1.clone(), f.models.clone()], &empty), &mut |e| {
        if let Event::Plan { state, .. } = e {
            states.push(state)
        }
    })
    .unwrap();
    assert_eq!(states, vec![PlanState::Installed, PlanState::Installed]);
}

#[test]
fn uninstall_in_both_orders() {
    for first in ["app-a", "app-b"] {
        let f = fixture(&format!("unref-{first}"));
        let (a, b) = (app(&f.t, "a"), app(&f.t, "b"));
        run(&f.store, &opts("app-a", &a, vec![f.rt1.clone(), f.models.clone()], &f.mirror)).unwrap();
        run(&f.store, &opts("app-b", &b, vec![f.rt1.clone(), f.sm120.clone()], &f.mirror)).unwrap();
        let second = if first == "app-a" { "app-b" } else { "app-a" };
        let (had, gc) = unref(&f.store, first, &mut || {}).unwrap();
        assert!(had);
        // the shared runtime stays; only the first app's own pack goes
        assert!(f.store.installed("trt-runtime", "11.4").is_some());
        let own = if first == "app-a" { "models-mangajanai/1.0.0" } else { "trt-sm120/11.4" };
        assert_eq!(gc.deleted, vec![own.to_string()]);
        unref(&f.store, second, &mut || {}).unwrap();
        assert!(installed(&f.store).is_empty(), "{first} first: {:?}", installed(&f.store));
        assert!(f.store.refs().is_empty());
    }
}

#[test]
fn unref_of_an_unknown_app_is_harmless() {
    let f = fixture("unref-unknown");
    let a = app(&f.t, "a");
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
    let (had, gc) = unref(&f.store, "never-installed", &mut || {}).unwrap();
    assert!(!had);
    assert!(gc.deleted.is_empty());
    assert_eq!(installed(&f.store), vec!["trt-runtime/11.4"]);
}

#[test]
fn different_versions_coexist_until_the_last_user_moves() {
    let f = fixture("versions");
    let (a, b) = (app(&f.t, "a"), app(&f.t, "b"));
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
    // B updates to the new TensorRT; A stays on the old one: both kept
    run(&f.store, &opts("app-b", &b, vec![f.rt2.clone()], &f.mirror)).unwrap();
    assert_eq!(installed(&f.store), vec!["trt-runtime/11.4", "trt-runtime/11.5"]);
    // A updates too: the old one is freed
    let r = run(&f.store, &opts("app-a", &a, vec![f.rt2.clone()], &f.mirror)).unwrap();
    assert_eq!(r.gc.deleted, vec!["trt-runtime/11.4".to_string()]);
    assert_eq!(installed(&f.store), vec!["trt-runtime/11.5"]);
}

#[test]
fn app_deleted_by_hand_has_a_dead_ref() {
    let f = fixture("dead-ref");
    let (a, b) = (app(&f.t, "a"), app(&f.t, "b"));
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone(), f.models.clone()], &f.mirror)).unwrap();
    run(&f.store, &opts("app-b", &b, vec![f.rt1.clone()], &f.mirror)).unwrap();
    std::fs::remove_dir_all(a.parent().unwrap()).unwrap();
    let lock = f.store.lock(&mut || {}).unwrap();
    let gc = f.store.gc(&lock).unwrap();
    assert_eq!(gc.deleted, vec!["models-mangajanai/1.0.0".to_string()]);
    assert_eq!(gc.dead_refs.len(), 1);
    assert!(f.store.read_ref("app-a").is_none());
    assert_eq!(installed(&f.store), vec!["trt-runtime/11.4"]);
}

#[test]
fn unreadable_ref_counts_as_dead() {
    let f = fixture("bad-ref");
    let a = app(&f.t, "a");
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
    std::fs::write(f.store.root().join("refs").join("app-a.json"), b"{ not json").unwrap();
    let lock = f.store.lock(&mut || {}).unwrap();
    let gc = f.store.gc(&lock).unwrap();
    assert_eq!(gc.deleted, vec!["trt-runtime/11.4".to_string()]);
}

#[test]
fn update_replaces_the_ref_whole() {
    let f = fixture("ref-whole");
    let a = app(&f.t, "a");
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone(), f.models.clone()], &f.mirror)).unwrap();
    // the new version no longer uses the models
    let r = run(&f.store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
    assert_eq!(r.gc.deleted, vec!["models-mangajanai/1.0.0".to_string()]);
    assert_eq!(f.store.read_ref("app-a").unwrap().packs, vec!["trt-runtime/11.4"]);
}

#[test]
fn interrupted_install_leaves_nothing_behind() {
    let f = fixture("interrupted");
    let a = app(&f.t, "a");
    // what a killed install leaves: a half download in tmp/, and a pack folder without .pack.json
    std::fs::write(f.store.tmp_dir().join("trt-runtime.7z.part"), b"half").unwrap();
    std::fs::create_dir_all(f.store.tmp_dir().join("dl-x-1/pack")).unwrap();
    let half = f.store.pack_dir("trt-runtime", "11.4");
    std::fs::create_dir_all(&half).unwrap();
    std::fs::write(half.join("nvinfer_11.dll"), b"trunc").unwrap();
    assert!(f.store.installed("trt-runtime", "11.4").is_none(), "no .pack.json: not installed");
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
    assert_eq!(std::fs::read(half.join("nvinfer_11.dll")).unwrap(), b"nvinfer 11.4");
    assert_eq!(std::fs::read_dir(f.store.tmp_dir()).unwrap().count(), 0);
}

#[test]
fn incomplete_pack_folders_are_removed_by_gc() {
    let f = fixture("incomplete");
    let half = f.store.pack_dir("trt-sm86", "11.4");
    std::fs::create_dir_all(&half).unwrap();
    std::fs::write(half.join("x.dll"), b"x").unwrap();
    let lock = f.store.lock(&mut || {}).unwrap();
    let gc = f.store.gc(&lock).unwrap();
    assert_eq!(gc.incomplete, vec!["trt-sm86/11.4".to_string()]);
    assert!(!half.exists());
    assert!(!half.parent().unwrap().exists(), "the empty packs/<name>/ goes too");
}

#[test]
fn bad_archive_fails_without_touching_the_store() {
    let f = fixture("bad-archive");
    let a = app(&f.t, "a");
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
    // the archive changed under its pin
    let mut bad = f.models.clone();
    std::fs::write(f.mirror.join(&bad.asset), b"corrupt").unwrap();
    bad.bytes = 7;
    let e = run(&f.store, &opts("app-a", &a, vec![f.rt1.clone(), bad], &f.mirror)).err().unwrap();
    assert!(e.0.contains("SHA-256"), "{e}");
    // the old ref and its pack are untouched, nothing half-installed
    assert_eq!(f.store.read_ref("app-a").unwrap().packs, vec!["trt-runtime/11.4"]);
    assert_eq!(installed(&f.store), vec!["trt-runtime/11.4"]);
    assert!(!f.store.pack_dir("models-mangajanai", "1.0.0").exists());
    assert_eq!(std::fs::read_dir(f.store.tmp_dir()).unwrap().count(), 0);
}

#[test]
fn archive_must_hold_exactly_the_indexed_files() {
    let f = fixture("files-mismatch");
    let a = app(&f.t, "a");
    // an archive whose pin matches but whose contents don't match the file list
    let mut p = make_pack(&f.mirror, "models-x", "1", "models", None, &[("a.onnx", b"A"), ("extra.onnx", b"E")]);
    p.files.retain(|x| x.path == "a.onnx");
    let e = run(&f.store, &opts("app-a", &a, vec![p.clone()], &f.mirror)).err().unwrap();
    assert!(e.0.contains("don't match"), "{e}");
    // a file whose hash doesn't match its entry
    let mut q = make_pack(&f.mirror, "models-y", "1", "models", None, &[("a.onnx", b"A")]);
    q.files[0].sha256 = sha256_bytes(b"B");
    let e = run(&f.store, &opts("app-a", &a, vec![q], &f.mirror)).err().unwrap();
    assert!(e.0.contains("a.onnx failed its SHA-256"), "{e}");
    assert!(installed(&f.store).is_empty());
}

#[test]
fn newer_store_format_is_added_to_but_never_deleted_from() {
    let f = fixture("newer-format");
    let a = app(&f.t, "a");
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone(), f.models.clone()], &f.mirror)).unwrap();
    std::fs::write(f.store.root().join("store.json"), br#"{"format": 2}"#).unwrap();
    let store = Store::open(f.store.root()).unwrap();
    assert!(!store.may_delete());
    std::fs::write(store.tmp_dir().join("leftover"), b"x").unwrap();
    // an update that drops the models, and a dead ref: nothing may be deleted
    let r = run(&store, &opts("app-a", &a, vec![f.rt2.clone()], &f.mirror)).unwrap();
    assert!(r.gc.newer_format);
    assert!(r.gc.deleted.is_empty());
    assert_eq!(installed(&store), vec!["models-mangajanai/1.0.0", "trt-runtime/11.4", "trt-runtime/11.5"]);
    std::fs::remove_dir_all(a.parent().unwrap()).unwrap();
    let (_, gc) = unref(&store, "app-a", &mut || {}).unwrap();
    assert!(gc.deleted.is_empty());
    assert!(store.tmp_dir().join("leftover").exists(), "tmp isn't cleared either");
    assert_eq!(installed(&store).len(), 3);
}

#[test]
fn open_existing_creates_nothing() {
    let t = Tmp::new("open-existing");
    let root = t.path().join("nowhere");
    assert!(Store::open_existing(&root).is_none());
    assert!(!root.exists());
    Store::open(&root).unwrap();
    assert_eq!(Store::open_existing(&root).unwrap().format(), store::FORMAT);
}

#[test]
fn the_lock_is_exclusive() {
    let f = fixture("lock");
    let held = f.store.lock(&mut || {}).unwrap();
    assert!(f.store.try_lock().unwrap().is_none());
    let other = Store::open(f.store.root()).unwrap();
    assert!(other.try_lock().unwrap().is_none(), "another handle on the same store");
    drop(held);
    assert!(f.store.try_lock().unwrap().is_some());
}

#[test]
fn concurrent_installers_wait_for_each_other() {
    let f = fixture("concurrent");
    let (a, b) = (app(&f.t, "a"), app(&f.t, "b"));
    let root = f.store.root().to_path_buf();
    let held = f.store.lock(&mut || {}).unwrap();
    let mut handles = Vec::new();
    for (id, exe) in [("app-a", a), ("app-b", b)] {
        let (root, mirror, packs) = (root.clone(), f.mirror.clone(), vec![f.rt1.clone(), f.models.clone()]);
        handles.push(std::thread::spawn(move || {
            let store = Store::open(&root).unwrap();
            let mut waited = false;
            ensure(&store, &opts(id, &exe, packs, &mirror), &mut |e| {
                if let Event::Waiting = e {
                    waited = true
                }
            })
            .unwrap();
            waited
        }));
    }
    std::thread::sleep(std::time::Duration::from_millis(300));
    drop(held);
    for h in handles {
        assert!(h.join().unwrap(), "both installers had to wait for the lock");
    }
    assert_eq!(installed(&f.store), vec!["models-mangajanai/1.0.0", "trt-runtime/11.4"]);
    assert_eq!(f.store.refs().len(), 2);
}

#[test]
fn a_missing_app_path_is_refused() {
    let f = fixture("missing-app");
    let e = run(&f.store, &opts("app-a", &f.t.path().join("no/such.exe"), vec![f.rt1.clone()], &f.mirror)).err().unwrap();
    assert!(e.0.contains("does not exist"), "{e}");
    assert!(installed(&f.store).is_empty());
}

#[test]
fn check_changes_nothing() {
    let f = fixture("check");
    let a = app(&f.t, "a");
    let mut o = opts("app-a", &a, vec![f.rt1.clone()], &f.mirror);
    o.check = true;
    let r = run(&f.store, &o).unwrap();
    assert!(!r.complete);
    assert!(installed(&f.store).is_empty());
    assert!(f.store.read_ref("app-a").is_none());
}

#[test]
fn check_plans_an_adoption_without_doing_it() {
    let f = fixture("check-adopt");
    let a = app(&f.t, "a");
    let old = f.t.path().join("old-install");
    std::fs::create_dir_all(&old).unwrap();
    std::fs::write(old.join("nvinfer_11.dll"), b"nvinfer 11.4").unwrap();
    std::fs::write(old.join("trtexec.exe"), b"trtexec").unwrap();
    let mut o = opts("app-a", &a, vec![f.rt1.clone(), f.models.clone()], &f.mirror);
    o.adopt_from = vec![old.clone()];
    o.check = true;
    let mut states = Vec::new();
    let mut total = None;
    let r = ensure(&f.store, &o, &mut |e| match e {
        Event::Plan { state, .. } => states.push(state),
        Event::Total(t) => total = Some(t),
        _ => {}
    })
    .unwrap();
    // the runtime would be adopted (no download), the models downloaded; nothing changed yet
    assert_eq!(states, vec![PlanState::Adopted, PlanState::Download]);
    assert_eq!(total, Some(f.models.bytes));
    assert!(!r.complete);
    assert!(installed(&f.store).is_empty());
    // with only adoptable packs, there's still work to do
    o.packs = vec![f.rt1.clone()];
    assert!(!run(&f.store, &o).unwrap().complete);
    assert!(installed(&f.store).is_empty());
}

#[test]
fn adopt_takes_matching_old_files_and_leaves_them_in_place() {
    let f = fixture("adopt");
    let a = app(&f.t, "a");
    // an old private install: runtime files flat, models in models/
    let old = f.t.path().join("old-install");
    std::fs::create_dir_all(old.join("models")).unwrap();
    std::fs::write(old.join("nvinfer_11.dll"), b"nvinfer 11.4").unwrap();
    std::fs::write(old.join("trtexec.exe"), b"trtexec").unwrap();
    std::fs::write(old.join("models/a.onnx"), b"model a").unwrap();
    std::fs::write(old.join("models/b.onnx"), b"model b (a different version)").unwrap();
    let mut o = opts("app-a", &a, vec![f.rt1.clone(), f.models.clone()], &f.mirror);
    o.adopt_from = vec![old.clone()];
    // the runtime matches, the models don't: runtime adopted, models downloaded
    std::fs::remove_file(f.mirror.join(&f.rt1.asset)).unwrap();
    let mut states = Vec::new();
    let r = ensure(&f.store, &o, &mut |e| {
        if let Event::Plan { state, .. } = e {
            states.push(state)
        }
    })
    .unwrap();
    assert_eq!(states, vec![PlanState::Adopted, PlanState::Download]);
    assert_eq!(r.adopted, vec!["trt-runtime/11.4".to_string()]);
    assert_eq!(std::fs::read(f.store.pack_dir("trt-runtime", "11.4").join("nvinfer_11.dll")).unwrap(), b"nvinfer 11.4");
    assert!(f.store.installed("trt-runtime", "11.4").unwrap().source.starts_with("adopted from"));
    // the old install still works until the app removes its private copies
    assert_eq!(std::fs::read(old.join("nvinfer_11.dll")).unwrap(), b"nvinfer 11.4");
    // and removing them doesn't touch the store's copy (hard link)
    std::fs::remove_file(old.join("nvinfer_11.dll")).unwrap();
    assert_eq!(std::fs::read(f.store.pack_dir("trt-runtime", "11.4").join("nvinfer_11.dll")).unwrap(), b"nvinfer 11.4");
}

#[test]
fn adopt_finds_models_in_the_models_folder() {
    let f = fixture("adopt-models");
    let a = app(&f.t, "a");
    let old = f.t.path().join("old");
    std::fs::create_dir_all(old.join("models")).unwrap();
    std::fs::write(old.join("models/a.onnx"), b"model a").unwrap();
    std::fs::write(old.join("models/b.onnx"), b"model b").unwrap();
    let lock = f.store.lock(&mut || {}).unwrap();
    assert!(adopt(&f.store, &lock, &f.models, std::slice::from_ref(&old)).unwrap());
    assert!(f.store.installed("models-mangajanai", "1.0.0").is_some());
    drop(lock);
    // adopted packs without a ref are freed by the next gc: ensure writes the ref in the same run
    let r = run(&f.store, &opts("app-a", &a, vec![f.models.clone()], &f.mirror)).unwrap();
    assert!(r.gc.deleted.is_empty());
}

#[test]
fn a_missing_or_wrong_file_means_no_adopt() {
    let f = fixture("adopt-partial");
    let old = f.t.path().join("old");
    std::fs::create_dir_all(&old).unwrap();
    std::fs::write(old.join("nvinfer_11.dll"), b"nvinfer 11.4").unwrap(); // trtexec.exe missing
    let lock = f.store.lock(&mut || {}).unwrap();
    assert!(!adopt(&f.store, &lock, &f.rt1, std::slice::from_ref(&old)).unwrap());
    std::fs::write(old.join("trtexec.exe"), b"other trtexec").unwrap(); // wrong content, same name
    assert!(!adopt(&f.store, &lock, &f.rt1, std::slice::from_ref(&old)).unwrap());
    assert!(installed(&f.store).is_empty());
}

#[test]
fn portable_apps_are_invisible_to_gc() {
    // a portable app keeps private files and never writes a ref: gc only ever looks inside the store
    let f = fixture("portable");
    let a = app(&f.t, "a");
    let portable = f.t.path().join("portable-app");
    std::fs::create_dir_all(&portable).unwrap();
    std::fs::write(portable.join("nvinfer_11.dll"), b"nvinfer 11.4").unwrap();
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
    unref(&f.store, "app-a", &mut || {}).unwrap();
    assert!(installed(&f.store).is_empty());
    assert!(portable.join("nvinfer_11.dll").exists());
}

#[cfg(windows)]
#[test]
fn a_pack_in_use_is_skipped_whole_and_freed_later() {
    use std::os::windows::fs::OpenOptionsExt;
    let f = fixture("in-use");
    let (a, b) = (app(&f.t, "a"), app(&f.t, "b"));
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
    // app B runs a DLL from app A's pack (its ref was removed, e.g. uninstalled while running)
    let dll = f.store.pack_dir("trt-runtime", "11.4").join("trtexec.exe");
    let held = std::fs::OpenOptions::new().read(true).share_mode(1 /* FILE_SHARE_READ */).open(&dll).unwrap();
    let _ = b;
    let (_, gc) = unref(&f.store, "app-a", &mut || {}).unwrap();
    assert_eq!(gc.in_use, vec!["trt-runtime/11.4".to_string()]);
    assert!(gc.deleted.is_empty());
    let meta = f.store.installed("trt-runtime", "11.4").expect("kept whole, still complete");
    assert_eq!(meta.files.len(), 2);
    assert!(f.store.pack_dir("trt-runtime", "11.4").join("nvinfer_11.dll").exists());
    drop(held);
    let lock = f.store.lock(&mut || {}).unwrap();
    assert_eq!(f.store.gc(&lock).unwrap().deleted, vec!["trt-runtime/11.4".to_string()]);
}

#[cfg(windows)]
#[test]
fn a_loaded_dll_blocks_deletion() {
    // the real case: a DLL loaded from a pack. Use a system DLL copied into a pack.
    let f = fixture("loaded-dll");
    let a = app(&f.t, "a");
    let sys = PathBuf::from(std::env::var("SystemRoot").unwrap()).join("System32").join("version.dll");
    let data = std::fs::read(&sys).unwrap();
    let p = make_pack(&f.mirror, "dll-pack", "1", "trt", None, &[("verx.dll", &data)]);
    run(&f.store, &opts("app-a", &a, vec![p], &f.mirror)).unwrap();
    let lib = unsafe { libloading::Library::new(f.store.pack_dir("dll-pack", "1").join("verx.dll")) }.unwrap();
    let (_, gc) = unref(&f.store, "app-a", &mut || {}).unwrap();
    assert_eq!(gc.in_use, vec!["dll-pack/1".to_string()]);
    assert!(f.store.installed("dll-pack", "1").is_some());
    drop(lib);
    let lock = f.store.lock(&mut || {}).unwrap();
    assert_eq!(f.store.gc(&lock).unwrap().deleted, vec!["dll-pack/1".to_string()]);
}

#[test]
fn invalid_names_are_rejected() {
    let f = fixture("names");
    let a = app(&f.t, "a");
    let lock = f.store.lock(&mut || {}).unwrap();
    for bad in ["", "..", "a/b", "a\\b", "x y", "c:"] {
        let r = AppRef { app_id: bad.into(), app_version: String::new(), app_path: a.clone(), packs: vec![] };
        assert!(f.store.write_ref(&lock, &r).is_err(), "{bad:?}");
    }
    assert!(index::check_rel_path("../x").is_err());
    assert!(index::check_rel_path("a/../../x").is_err());
    assert!(index::check_rel_path("/etc/x").is_err());
    assert!(index::check_rel_path("C:/x").is_err());
    assert!(index::check_rel_path("sub/ok.onnx").is_ok());
}

fn index_with(packs: Vec<IndexPack>) -> Index {
    Index { format: 1, tag: "runtime-test".into(), platform: index::platform().into(), packs }
}

fn gpu(major: i32, minor: i32) -> Gpu {
    Gpu { name: format!("sm{major}{minor}"), major, minor }
}

#[test]
fn select_picks_builders_per_gpu_generation() {
    let m = Path::new("unused");
    let pk = |name: &str, kind: &str, sm: Option<&str>| IndexPack {
        name: name.into(),
        version: "1".into(),
        kind: kind.into(),
        sm: sm.map(String::from),
        asset: format!("{name}.7z"),
        url: String::new(),
        sha256: String::new(),
        bytes: 1,
        installed_bytes: 1,
        files: vec![],
    };
    let _ = m;
    let idx = index_with(vec![
        pk("trt-runtime", "trt", None),
        pk("trt-sm120", "trt-builder", Some("sm120")),
        pk("trt-sm86", "trt-builder", Some("sm86")),
        pk("trt-ptx", "trt-builder", Some("ptx")),
        pk("trtrtx-runtime", "trtrtx", None),
        pk("dml-runtime", "private", None),
        pk("models-mangajanai", "models", None),
    ]);
    let names = |v: Vec<IndexPack>| v.into_iter().map(|p| p.name).collect::<Vec<_>>();
    let want = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    // two GPUs of one generation, one of another, one without its own pack (sm130 -> ptx)
    let got = idx.select(&want(&["trt", "trtrtx", "models-mangajanai"]), &[gpu(12, 0), gpu(12, 0), gpu(8, 6), gpu(13, 0)]).unwrap();
    assert_eq!(names(got), vec!["trt-runtime", "trt-sm120", "trt-sm86", "trt-ptx", "trtrtx-runtime", "models-mangajanai"]);
    assert!(idx.select(&want(&["trt"]), &[]).is_err(), "no GPU, no builder packs");
    assert!(idx.select(&want(&["dml-runtime"]), &[]).is_err(), "private packs aren't for the store");
    assert!(idx.select(&want(&["models-nope"]), &[]).is_err());
    let no_ptx = index_with(vec![pk("trt-runtime", "trt", None), pk("trt-sm120", "trt-builder", Some("sm120"))]);
    assert!(no_ptx.select(&want(&["trt"]), &[gpu(7, 5)]).is_err());
}

#[test]
fn index_is_checked_against_its_pin() {
    let idx = index_with(vec![]);
    let bytes = serde_json::to_vec(&idx).unwrap();
    let pin = sha256_bytes(&bytes);
    assert!(Index::parse(&bytes, Some(&pin)).is_ok());
    assert!(Index::parse(&bytes, Some(&pin.to_uppercase())).is_ok());
    assert!(Index::parse(&bytes, Some(&sha256_bytes(b"other"))).is_err());
    let mut v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    v["format"] = 2.into();
    assert!(Index::parse(&serde_json::to_vec(&v).unwrap(), None).is_err(), "unknown index format");
    let mut v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    v["packs"] = serde_json::json!([{ "name": "x", "version": "1", "kind": "models", "asset": "x.7z", "url": "",
        "sha256": "", "bytes": 1, "files": [{ "path": "../evil.dll", "sha256": "", "bytes": 1 }] }]);
    assert!(Index::parse(&serde_json::to_vec(&v).unwrap(), None).is_err(), "path escaping the pack folder");
}

// ---- moving the store (location.rs) ---------------------------------------------------------
// The default path becomes a link to the new folder. Every app reaches the store through that path,
// old versions included, so after a move everything must keep working through it.

use crate::location::{Location, location, move_back, move_to, reset};

fn no_wait() {}

fn moved(f: &Fixture) -> (PathBuf, PathBuf) {
    let a = app(&f.t, "a");
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
    (f.store.root().to_path_buf(), a)
}

#[test]
fn a_moved_store_is_reached_through_the_default_path() {
    let f = fixture("move");
    let (root, a) = moved(&f);
    let dest = f.t.path().join("other-drive").join("aji");
    let mut seen = Vec::new();
    move_to(&root, &dest, &mut no_wait, &mut |d, t| seen.push((d, t))).unwrap();

    assert_eq!(location(&root), Location::Moved { target: dest.clone(), available: true });
    assert!(dest.join("packs/trt-runtime/11.4/nvinfer_11.dll").is_file(), "the packs are at the new place");
    assert_eq!(std::fs::read(root.join("packs/trt-runtime/11.4/nvinfer_11.dll")).unwrap(), b"nvinfer 11.4", "and read through the old path");
    assert!(!PathBuf::from(format!("{}.moving", root.display())).exists(), "the old copy is gone");
    assert_eq!(seen.last().map(|&(d, t)| d == t), Some(true), "progress reached the total");

    // an app that knows nothing of the move: opens, installs, refs and gc through the default path
    let store = Store::open(&root).unwrap();
    assert_eq!(installed(&store), ["trt-runtime/11.4"]);
    let b = app(&f.t, "b");
    run(&store, &opts("app-b", &b, vec![f.models.clone()], &f.mirror)).unwrap();
    assert!(dest.join("packs/models-mangajanai/1.0.0/a.onnx").is_file(), "new packs land at the new place");
    assert!(dest.join("refs/app-b.json").is_file());
    let (_, gc) = unref(&store, "app-a", &mut || {}).unwrap();
    assert_eq!(gc.deleted, ["trt-runtime/11.4"]);
    assert!(!dest.join("packs/trt-runtime/11.4").exists());
    let _ = a;
}

#[test]
fn the_store_moves_again_and_back() {
    let f = fixture("move-again");
    let (root, _) = moved(&f);
    let (d1, d2) = (f.t.path().join("d1"), f.t.path().join("d2"));
    move_to(&root, &d1, &mut no_wait, &mut |_, _| {}).unwrap();
    move_to(&root, &d2, &mut no_wait, &mut |_, _| {}).unwrap();
    assert_eq!(location(&root), Location::Moved { target: d2.clone(), available: true });
    assert!(!d1.exists(), "the previous place is cleared");

    move_back(&root, &mut no_wait, &mut |_, _| {}).unwrap();
    assert_eq!(location(&root), Location::Default);
    assert!(!d2.exists());
    assert!(std::fs::symlink_metadata(&root).unwrap().is_dir(), "a real folder again");
    assert_eq!(installed(&Store::open(&root).unwrap()), ["trt-runtime/11.4"]);
    move_back(&root, &mut no_wait, &mut |_, _| {}).unwrap(); // already there: nothing to do
}

#[test]
fn moving_to_the_default_path_moves_back() {
    let f = fixture("move-to-default");
    let (root, _) = moved(&f);
    move_to(&root, &f.t.path().join("d1"), &mut no_wait, &mut |_, _| {}).unwrap();
    move_to(&root, &root, &mut no_wait, &mut |_, _| {}).unwrap();
    assert_eq!(location(&root), Location::Default);
}

#[test]
fn a_move_needs_a_new_or_empty_folder_outside_the_store() {
    let f = fixture("move-refused");
    let (root, _) = moved(&f);
    let full = f.t.path().join("full");
    std::fs::create_dir_all(&full).unwrap();
    std::fs::write(full.join("my-file.txt"), b"mine").unwrap();
    assert!(move_to(&root, &full, &mut no_wait, &mut |_, _| {}).is_err(), "a folder with files in it");
    assert!(full.join("my-file.txt").is_file());
    assert!(move_to(&root, &root.join("packs").join("x"), &mut no_wait, &mut |_, _| {}).is_err(), "inside the store");
    assert!(move_to(&root, Path::new("relative/dir"), &mut no_wait, &mut |_, _| {}).is_err(), "a relative path");
    assert_eq!(location(&root), Location::Default, "nothing changed");
    assert_eq!(installed(&Store::open(&root).unwrap()), ["trt-runtime/11.4"]);

    let empty = f.t.path().join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    move_to(&root, &empty, &mut no_wait, &mut |_, _| {}).unwrap();
    assert_eq!(location(&root), Location::Moved { target: empty, available: true });
}

#[test]
fn a_store_not_installed_yet_is_just_linked() {
    let t = Tmp::new("move-nothing");
    let root = t.path().join("store");
    let dest = t.path().join("there");
    move_to(&root, &dest, &mut no_wait, &mut |_, _| {}).unwrap();
    assert_eq!(location(&root), Location::Moved { target: dest.clone(), available: true });
    Store::open(&root).unwrap();
    assert!(dest.join("store.json").is_file(), "the first install creates the store there");
}

// A moved store's drive that's gone (unplugged): nothing may quietly start a new store on the default
// drive and download gigabytes there. Reset is the explicit way back.
#[test]
fn a_missing_target_fails_safely_until_reset() {
    let f = fixture("move-missing");
    let (root, _) = moved(&f);
    let dest = f.t.path().join("usb");
    move_to(&root, &dest, &mut no_wait, &mut |_, _| {}).unwrap();
    let unplugged = f.t.path().join("usb-unplugged");
    std::fs::rename(&dest, &unplugged).unwrap();

    assert_eq!(location(&root), Location::Moved { target: dest.clone(), available: false });
    assert!(Store::open(&root).is_err(), "no new store through a broken link");
    assert!(move_back(&root, &mut no_wait, &mut |_, _| {}).is_err());
    assert!(move_to(&root, &f.t.path().join("d2"), &mut no_wait, &mut |_, _| {}).is_err());

    reset(&root).unwrap();
    assert_eq!(location(&root), Location::Default);
    let fresh = Store::open(&root).unwrap();
    assert!(installed(&fresh).is_empty(), "an empty store at the default path");
    assert!(unplugged.join("packs/trt-runtime/11.4/nvinfer_11.dll").is_file(), "the old copy is never touched");
}

#[test]
fn reset_is_refused_while_the_target_is_there() {
    let f = fixture("reset-refused");
    let (root, _) = moved(&f);
    let dest = f.t.path().join("d1");
    move_to(&root, &dest, &mut no_wait, &mut |_, _| {}).unwrap();
    assert!(reset(&root).is_err(), "move back instead, which keeps what's installed");
    assert_eq!(location(&root), Location::Moved { target: dest, available: true });
    reset(&Tmp::new("reset-default").path().join("store")).unwrap(); // a default store: nothing to do
}

#[cfg(windows)]
#[test]
fn a_store_in_use_doesnt_move() {
    use std::os::windows::fs::OpenOptionsExt;
    let f = fixture("move-in-use");
    let (root, _) = moved(&f);
    let dll = root.join("packs/trt-runtime/11.4/nvinfer_11.dll");
    // a running app with a DLL loaded from the store
    let held = std::fs::OpenOptions::new().read(true).share_mode(1 /* FILE_SHARE_READ */).open(&dll).unwrap();
    let dest = f.t.path().join("d1");
    let e = move_to(&root, &dest, &mut no_wait, &mut |_, _| {}).unwrap_err().to_string();
    assert!(e.contains("in use") && e.contains("Nothing was changed"), "{e}");
    assert_eq!(location(&root), Location::Default);
    assert!(!dest.exists(), "the copy is removed");
    assert!(dll.is_file());
    drop(held);
    move_to(&root, &dest, &mut no_wait, &mut |_, _| {}).unwrap();

    // and moving back while it's loaded from the new place
    let held = std::fs::OpenOptions::new().read(true).share_mode(1).open(dest.join("packs/trt-runtime/11.4/nvinfer_11.dll")).unwrap();
    assert!(move_back(&root, &mut no_wait, &mut |_, _| {}).is_err());
    assert_eq!(location(&root), Location::Moved { target: dest.clone(), available: true });
    assert!(!PathBuf::from(format!("{}.new", root.display())).exists());
    drop(held);
    move_back(&root, &mut no_wait, &mut |_, _| {}).unwrap();
    assert_eq!(location(&root), Location::Default);
}

// ---- pre-Turing GPUs (compute capability below 7.5) -----------------------------------------
// TensorRT doesn't run on them, and the PTX pack can't help (PTX only compiles forward).

fn pre_turing_index() -> Index {
    let pk = |name: &str, kind: &str, sm: Option<&str>| IndexPack {
        name: name.into(),
        version: "1".into(),
        kind: kind.into(),
        sm: sm.map(String::from),
        asset: format!("{name}.7z"),
        url: String::new(),
        sha256: String::new(),
        bytes: 1,
        installed_bytes: 1,
        files: vec![],
    };
    index_with(vec![
        pk("trt-runtime", "trt", None),
        pk("trt-sm75", "trt-builder", Some("sm75")),
        pk("trt-sm86", "trt-builder", Some("sm86")),
        pk("trt-ptx", "trt-builder", Some("ptx")),
        pk("models-mangajanai", "models", None),
    ])
}

#[test]
fn trt_threshold_is_turing() {
    assert_eq!(crate::nvml::TRT_MIN_COMPUTE, (7, 5));
    for (major, minor) in [(5, 2), (6, 1), (7, 0), (7, 2)] {
        assert!(!gpu(major, minor).trt_capable(), "sm{major}{minor}");
    }
    for (major, minor) in [(7, 5), (8, 0), (8, 6), (12, 0), (13, 0)] {
        assert!(gpu(major, minor).trt_capable(), "sm{major}{minor}");
    }
}

#[test]
fn select_skips_pre_turing_gpus() {
    let idx = pre_turing_index();
    let names = |v: Vec<IndexPack>| v.into_iter().map(|p| p.name).collect::<Vec<_>>();
    let want = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    // a GTX 1080 (sm61) next to an RTX 3080 (sm86): only the 3080's pack, no PTX for the 1080
    let got = idx.select(&want(&["trt"]), &[gpu(6, 1), gpu(8, 6)]).unwrap();
    assert_eq!(names(got), vec!["trt-runtime", "trt-sm86"]);
    // Volta (sm70) next to Turing (sm75)
    let got = idx.select(&want(&["trt"]), &[gpu(7, 0), gpu(7, 5)]).unwrap();
    assert_eq!(names(got), vec!["trt-runtime", "trt-sm75"]);
    // only old GPUs: TensorRT can't run, and the error says why
    for gpus in [vec![gpu(6, 1)], vec![gpu(6, 1), gpu(7, 0)]] {
        let e = idx.select(&want(&["trt"]), &gpus).unwrap_err().to_string();
        assert!(e.contains("TensorRT needs an NVIDIA GeForce RTX 20 / GTX 16 series or newer GPU (compute capability 7.5+)"), "{e}");
        assert!(e.contains("sm61 is compute 6.1"), "{e}");
    }
    // packs other than trt don't depend on the GPU
    let got = idx.select(&want(&["models-mangajanai"]), &[gpu(6, 1)]).unwrap();
    assert_eq!(names(got), vec!["models-mangajanai"]);
}

// ---- a pack is the same pack whatever archive it came in (lib.rs `have`, store.rs `commit_pack`) ---
// Each runtime-<n> release may pack unchanged files into an archive with another SHA-256. An installed
// pack with the index's files must count as installed (or every app on a newer release downloads it
// again and swaps it under the apps using it), and a complete pack folder is never replaced.

/// The same files as `p` in another archive (stored, not deflated, and in reverse order): another
/// asset, SHA-256 and size, the same file list.
fn repacked(mirror: &Path, p: &IndexPack, files: &[(&str, &[u8])]) -> IndexPack {
    let asset = format!("{}-{}-repacked.zip", p.name, p.version);
    let path = mirror.join(&asset);
    let mut z = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
    let o = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (n, data) in files.iter().rev() {
        z.start_file(*n, o).unwrap();
        z.write_all(data).unwrap();
    }
    z.finish().unwrap();
    let q = IndexPack {
        asset: asset.clone(),
        url: format!("https://invalid.example/{asset}"),
        sha256: sha256_file(&path).unwrap(),
        bytes: std::fs::metadata(&path).unwrap().len(),
        files: p.files.iter().rev().cloned().collect(),
        ..p.clone()
    };
    assert_ne!(q.sha256, p.sha256);
    q
}

const RT1_FILES: &[(&str, &[u8])] = &[("nvinfer_11.dll", b"nvinfer 11.4"), ("trtexec.exe", b"trtexec")];

fn plan_states(store: &Store, o: &EnsureOptions) -> (Vec<PlanState>, Option<u64>, Result<EnsureResult>) {
    let (mut states, mut total) = (Vec::new(), None);
    let r = ensure(store, o, &mut |e| match e {
        Event::Plan { state, .. } => states.push(state),
        Event::Total(t) => total = Some(t),
        _ => {}
    });
    (states, total, r)
}

#[test]
fn a_repacked_pack_with_the_same_files_is_installed() {
    let f = fixture("repacked");
    let (a, b) = (app(&f.t, "a"), app(&f.t, "b"));
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
    let rt1b = repacked(&f.mirror, &f.rt1, RT1_FILES);
    let pack_json = f.store.pack_dir("trt-runtime", "11.4").join(".pack.json");
    let before = std::fs::read(&pack_json).unwrap();
    // B is on a newer release that repacked the same files: nothing to download (an empty mirror)
    let empty = f.t.path().join("empty-mirror");
    std::fs::create_dir_all(&empty).unwrap();
    let mut o = opts("app-b", &b, vec![rt1b.clone()], &empty);
    o.check = true;
    let (states, total, r) = plan_states(&f.store, &o);
    assert!(r.unwrap().complete);
    assert_eq!((states, total), (vec![PlanState::Installed], Some(0)));
    o.check = false;
    let (states, total, r) = plan_states(&f.store, &o);
    let r = r.unwrap();
    assert_eq!((states, total, r.downloaded_bytes), (vec![PlanState::Installed], Some(0), 0));
    assert_eq!(r.packs[0].1, f.store.pack_dir("trt-runtime", "11.4"));
    assert_eq!(f.store.read_ref("app-b").unwrap().packs, vec!["trt-runtime/11.4"]);
    // the folder is untouched (its .pack.json still names the first archive), and A moving to the
    // newer release keeps it too
    run(&f.store, &opts("app-a", &a, vec![rt1b.clone()], &empty)).unwrap();
    assert_eq!(std::fs::read(&pack_json).unwrap(), before);
    // and apps on the two releases switching back and forth never reinstall it
    run(&f.store, &opts("app-b", &b, vec![f.rt1.clone()], &empty)).unwrap();
    run(&f.store, &opts("app-b", &b, vec![rt1b], &empty)).unwrap();
    assert_eq!(installed(&f.store), vec!["trt-runtime/11.4"]);
    assert_eq!(std::fs::read(&pack_json).unwrap(), before);
}

#[test]
fn installed_means_the_same_file_list() {
    let f = fixture("same-files");
    let a = app(&f.t, "a");
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
    let m = f.store.installed("trt-runtime", "11.4").unwrap();
    assert!(store::same_files(&m.files, &f.rt1.files));
    assert!(f.store.check_same_files("trt-runtime", "11.4", &f.rt1.files).is_ok());
    let mut rev = f.rt1.files.clone();
    rev.reverse();
    rev[0].sha256 = rev[0].sha256.to_ascii_uppercase();
    assert!(store::same_files(&m.files, &rev), "order and hex case don't matter");
    for change in 0..5 {
        let mut other = f.rt1.files.clone();
        match change {
            0 => other[0].sha256 = sha256_bytes(b"other"),
            1 => other[0].bytes += 1,
            2 => other[0].path = "renamed.dll".into(),
            3 => {
                other.pop();
            }
            _ => other.push(PackFile { path: "extra.dll".into(), sha256: sha256_bytes(b"x"), bytes: 1 }),
        }
        assert!(!store::same_files(&m.files, &other), "change {change}");
        assert!(f.store.check_same_files("trt-runtime", "11.4", &other).is_err(), "change {change}");
    }
    let dup = vec![f.rt1.files[0].clone(), f.rt1.files[0].clone()];
    assert!(!store::same_files(&dup, &dup), "a list with a path twice is never the same");
}

#[test]
fn a_version_installed_with_other_files_is_refused_and_left_alone() {
    let f = fixture("other-files");
    let (a, b) = (app(&f.t, "a"), app(&f.t, "b"));
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
    let dir = f.store.pack_dir("trt-runtime", "11.4");
    let before = std::fs::read(dir.join(".pack.json")).unwrap();
    // a (broken) index that pins other files under the installed name/version
    let mirror2 = f.t.path().join("mirror2");
    let other = make_pack(&mirror2, "trt-runtime", "11.4", "trt", None, &[("nvinfer_11.dll", b"nvinfer 11.4 rebuilt"), ("trtexec.exe", b"trtexec")]);
    let models = make_pack(&mirror2, "models-mangajanai", "1.0.0", "models", None, &[("a.onnx", b"model a"), ("b.onnx", b"model b")]);
    for check in [true, false] {
        let mut o = opts("app-b", &b, vec![models.clone(), other.clone()], &mirror2);
        o.check = check;
        let (states, _, r) = plan_states(&f.store, &o);
        let e = r.err().unwrap();
        assert!(e.0.contains("trt-runtime/11.4") && e.0.contains("other files") && e.0.contains("never replaced"), "{e}");
        assert!(states.is_empty(), "refused before planning, downloading or adopting anything");
    }
    // A's pack and the store are as they were; B has no ref and nothing of B's was installed
    assert_eq!(std::fs::read(dir.join(".pack.json")).unwrap(), before);
    assert_eq!(std::fs::read(dir.join("nvinfer_11.dll")).unwrap(), b"nvinfer 11.4");
    assert!(f.store.read_ref("app-b").is_none());
    assert_eq!(installed(&f.store), vec!["trt-runtime/11.4"]);
    assert_eq!(std::fs::read_dir(f.store.tmp_dir()).unwrap().count(), 0);
    // A keeps working with its own index
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
}

#[test]
fn an_unreadable_pack_json_is_never_replaced() {
    let f = fixture("bad-pack-json");
    let (a, b) = (app(&f.t, "a"), app(&f.t, "b"));
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
    let dir = f.store.pack_dir("trt-runtime", "11.4");
    std::fs::write(dir.join(".pack.json"), b"{ not json").unwrap();
    let e = run(&f.store, &opts("app-b", &b, vec![f.rt1.clone()], &f.mirror)).err().unwrap();
    assert!(e.0.contains("can't be read") && e.0.contains("never replaced"), "{e}");
    assert_eq!(std::fs::read(dir.join(".pack.json")).unwrap(), b"{ not json");
    assert_eq!(std::fs::read(dir.join("nvinfer_11.dll")).unwrap(), b"nvinfer 11.4");
}

#[test]
fn commit_pack_never_replaces_a_complete_pack() {
    let f = fixture("commit-complete");
    let a = app(&f.t, "a");
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
    let dir = f.store.pack_dir("trt-runtime", "11.4");
    let before = std::fs::read(dir.join(".pack.json")).unwrap();
    let lock = f.store.lock(&mut || {}).unwrap();
    let stage = |name: &str, files: &[PackFile], data: &[u8]| {
        let s = f.store.tmp_folder(&lock, "stage").unwrap();
        for p in files {
            std::fs::write(s.join(&p.path), data).unwrap();
        }
        let meta = PackMeta {
            format: store::FORMAT,
            name: name.into(),
            version: "11.4".into(),
            kind: "trt".into(),
            platform: index::platform().into(),
            source: "test".into(),
            sha256: "0".repeat(64),
            files: files.to_vec(),
        };
        std::fs::write(s.join(".pack.json"), serde_json::to_vec(&meta).unwrap()).unwrap();
        s
    };
    // other files: refused, and nothing in the store changes
    let mut other = f.rt1.files.clone();
    other[0].sha256 = sha256_bytes(b"other");
    let s = stage("trt-runtime", &other, b"other");
    let e = f.store.commit_pack(&lock, &s, "trt-runtime", "11.4").err().unwrap();
    assert!(e.0.contains("other files"), "{e}");
    assert_eq!(std::fs::read(dir.join(".pack.json")).unwrap(), before);
    assert_eq!(std::fs::read(dir.join("nvinfer_11.dll")).unwrap(), b"nvinfer 11.4");
    // the same files: the installed folder is kept as it is, the staged copy dropped
    let s = stage("trt-runtime", &f.rt1.files, b"unused");
    assert_eq!(f.store.commit_pack(&lock, &s, "trt-runtime", "11.4").unwrap(), dir);
    assert!(!s.exists());
    assert_eq!(std::fs::read(dir.join(".pack.json")).unwrap(), before);
    assert_eq!(std::fs::read(dir.join("nvinfer_11.dll")).unwrap(), b"nvinfer 11.4");
    // a staged folder without a readable .pack.json is refused
    let s = stage("trt-runtime", &f.rt1.files, b"x");
    std::fs::write(s.join(".pack.json"), b"{").unwrap();
    assert!(f.store.commit_pack(&lock, &s, "trt-runtime", "11.4").is_err());
    // an incomplete leftover (no .pack.json) is still replaced
    let half = f.store.pack_dir("trt-sm86", "11.4");
    std::fs::create_dir_all(&half).unwrap();
    std::fs::write(half.join("x.dll"), b"trunc").unwrap();
    let s = stage("trt-sm86", &f.sm120.files, b"sm86");
    assert_eq!(f.store.commit_pack(&lock, &s, "trt-sm86", "11.4").unwrap(), half);
    assert!(f.store.is_complete("trt-sm86", "11.4"));
    assert!(!half.join("x.dll").exists());
}

#[cfg(windows)]
#[test]
fn a_pack_in_use_is_not_touched_by_a_newer_release() {
    use std::os::windows::fs::OpenOptionsExt;
    // the case that broke: app A has the pack's DLLs loaded while app B, on a newer runtime release
    // with the same files in another archive, installs
    let f = fixture("repacked-in-use");
    let (a, b) = (app(&f.t, "a"), app(&f.t, "b"));
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
    let dll = f.store.pack_dir("trt-runtime", "11.4").join("nvinfer_11.dll");
    let held = std::fs::OpenOptions::new().read(true).share_mode(1 /* FILE_SHARE_READ */).open(&dll).unwrap();
    let rt1b = repacked(&f.mirror, &f.rt1, RT1_FILES);
    let r = run(&f.store, &opts("app-b", &b, vec![rt1b], &f.mirror)).unwrap();
    assert_eq!(r.downloaded_bytes, 0);
    assert!(r.gc.in_use.is_empty());
    assert!(dll.is_file());
    drop(held);
}
