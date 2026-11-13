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
    // nothing to install, but B has no ref yet: an ensure must still write it
    assert!(!r.unwrap().complete);
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

// ---- `ensure --check` and the app's ref (lib.rs `check`) ----
// Updaters and AppImage launchers trust a complete check. A ref that names the app's old path (an
// AppImage moved to ~/Applications) is dead: the next gc by any app frees the packs, and the app has
// no TensorRT until something ensures again. So check is complete only when the ref is what an
// ensure writes: there, listing exactly these packs, for this app_path.

fn check_of(store: &Store, o: &EnsureOptions) -> EnsureResult {
    run(store, &EnsureOptions { check: true, ..clone_opts(o) }).unwrap()
}

fn clone_opts(o: &EnsureOptions) -> EnsureOptions {
    EnsureOptions {
        app_id: o.app_id.clone(),
        app_version: o.app_version.clone(),
        app_path: o.app_path.clone(),
        packs: o.packs.clone(),
        mirror: o.mirror.clone(),
        adopt_from: o.adopt_from.clone(),
        aji_dir: o.aji_dir.clone(),
        check: o.check,
    }
}

#[test]
fn check_is_complete_with_the_right_ref() {
    let f = fixture("check-complete");
    let a = app(&f.t, "a");
    let o = opts("app-a", &a, vec![f.rt1.clone(), f.models.clone()], &f.mirror);
    run(&f.store, &o).unwrap();
    let r = check_of(&f.store, &o);
    assert!(r.complete, "{:?}", r.stale);
    assert!(r.stale.is_empty());
    // the ref's order doesn't matter, only which packs
    let reordered = opts("app-a", &a, vec![f.models.clone(), f.rt1.clone()], &f.mirror);
    assert!(check_of(&f.store, &reordered).complete);
}

#[test]
fn check_is_incomplete_without_a_ref() {
    let f = fixture("check-no-ref");
    let (a, b) = (app(&f.t, "a"), app(&f.t, "b"));
    // the packs are there (another app installed them), but this app has no ref
    run(&f.store, &opts("app-b", &b, vec![f.rt1.clone()], &f.mirror)).unwrap();
    let o = opts("app-a", &a, vec![f.rt1.clone()], &f.mirror);
    let (states, total, r) = plan_states(&f.store, &EnsureOptions { check: true, ..clone_opts(&o) });
    let r = r.unwrap();
    assert_eq!((states, total), (vec![PlanState::Installed], Some(0)), "nothing to download");
    assert!(!r.complete);
    assert!(r.stale[0].contains("no ref"), "{:?}", r.stale);
    // an unreadable ref is no ref
    run(&f.store, &o).unwrap();
    std::fs::write(f.store.root().join("refs").join("app-a.json"), b"{ not json").unwrap();
    assert!(!check_of(&f.store, &o).complete);
}

#[test]
fn check_is_incomplete_with_a_stale_ref() {
    // the AppImage was moved: the ref names its old path, which is gone
    let f = fixture("check-moved");
    let (old, b) = (app(&f.t, "old"), app(&f.t, "b"));
    run(&f.store, &opts("app-a", &old, vec![f.rt1.clone(), f.models.clone()], &f.mirror)).unwrap();
    run(&f.store, &opts("app-b", &b, vec![f.rt1.clone()], &f.mirror)).unwrap();
    std::fs::remove_dir_all(old.parent().unwrap()).unwrap();
    let new = app(&f.t, "Applications");
    let o = opts("app-a", &new, vec![f.rt1.clone(), f.models.clone()], &f.mirror);
    let r = check_of(&f.store, &o);
    assert!(!r.complete);
    assert_eq!(r.stale.len(), 1, "{:?}", r.stale);
    assert!(r.stale[0].contains("another app path"), "{:?}", r.stale);
    // what the check prevents: a gc now (app B's next ensure) frees A's models
    assert!(!f.store.refs().iter().find(|e| e.app_ref.as_ref().is_some_and(|x| x.app_id == "app-a")).unwrap().live);
    // the updater's ensure, run because the check was incomplete, writes the ref for the new path
    // (nothing to download), and B's gc keeps everything
    let r = run(&f.store, &EnsureOptions { mirror: None, ..clone_opts(&o) }).unwrap();
    assert_eq!(r.downloaded_bytes, 0);
    assert_eq!(f.store.read_ref("app-a").unwrap().app_path, new);
    let r = run(&f.store, &opts("app-b", &b, vec![f.rt1.clone()], &f.mirror)).unwrap();
    assert!(r.gc.deleted.is_empty(), "{:?}", r.gc.deleted);
    assert!(check_of(&f.store, &o).complete);
}

#[test]
fn check_is_incomplete_when_the_ref_names_another_app_path() {
    // both paths exist (two copies of the AppImage): the ref keeps the packs for the other one only
    let f = fixture("check-other-path");
    let (one, two) = (app(&f.t, "one"), app(&f.t, "two"));
    run(&f.store, &opts("app-a", &one, vec![f.rt1.clone()], &f.mirror)).unwrap();
    let o = opts("app-a", &two, vec![f.rt1.clone()], &f.mirror);
    let r = check_of(&f.store, &o);
    assert!(!r.complete);
    assert!(r.stale.iter().any(|s| s.contains("another app path")), "{:?}", r.stale);
    run(&f.store, &o).unwrap();
    assert_eq!(f.store.read_ref("app-a").unwrap().app_path, two);
    assert!(check_of(&f.store, &o).complete);
}

#[cfg(windows)]
#[test]
fn check_compares_app_paths_case_insensitively_on_windows() {
    let f = fixture("check-path-case");
    let a = app(&f.t, "a");
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
    let upper = PathBuf::from(a.to_string_lossy().to_uppercase().replace('\\', "/"));
    assert!(check_of(&f.store, &opts("app-a", &upper, vec![f.rt1.clone()], &f.mirror)).complete);
}

#[test]
fn check_is_incomplete_when_the_ref_lists_other_packs() {
    let f = fixture("check-other-packs");
    let (a, b) = (app(&f.t, "a"), app(&f.t, "b"));
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone(), f.models.clone()], &f.mirror)).unwrap();
    run(&f.store, &opts("app-b", &b, vec![f.sm120.clone()], &f.mirror)).unwrap();
    // fewer packs (the models were dropped), more (a builder pack another app installed), others
    for packs in [vec![f.rt1.clone()], vec![f.rt1.clone(), f.models.clone(), f.sm120.clone()], vec![f.sm120.clone()]] {
        let o = opts("app-a", &a, packs, &f.mirror);
        let r = check_of(&f.store, &o);
        assert!(!r.complete);
        assert!(r.stale.iter().any(|s| s.contains("other packs")), "{:?}", r.stale);
    }
    // and an ensure (non-check) writes the ref whole
    let o = opts("app-a", &a, vec![f.rt1.clone(), f.sm120.clone()], &f.mirror);
    run(&f.store, &o).unwrap();
    let r = f.store.read_ref("app-a").unwrap();
    assert_eq!(r.packs, vec!["trt-runtime/11.4", "trt-sm120/11.4"]);
    assert_eq!(r.app_path, a);
    assert!(check_of(&f.store, &o).complete);
}

#[test]
fn check_is_incomplete_without_the_right_aji_runtime_json() {
    let f = fixture("check-json");
    let a = app(&f.t, "a");
    let aji = f.t.path().join("apps/a/inference");
    std::fs::create_dir_all(&aji).unwrap();
    let mut o = opts("app-a", &a, vec![f.rt1.clone()], &f.mirror);
    o.aji_dir = Some(aji.clone());
    run(&f.store, &o).unwrap();
    assert!(check_of(&f.store, &o).complete);
    let json = aji.join("aji-runtime.json");
    let good = std::fs::read(&json).unwrap();
    std::fs::remove_file(&json).unwrap();
    let r = check_of(&f.store, &o);
    assert!(!r.complete);
    assert!(r.stale[0].contains("aji-runtime.json"), "{:?}", r.stale);
    // one naming another pack (an older version's)
    std::fs::write(&json, String::from_utf8(good.clone()).unwrap().replace("11.4", "11.3")).unwrap();
    assert!(!check_of(&f.store, &o).complete);
    // the same content, formatted differently, is fine
    let v: serde_json::Value = serde_json::from_slice(&good).unwrap();
    std::fs::write(&json, serde_json::to_vec(&v).unwrap()).unwrap();
    assert!(check_of(&f.store, &o).complete);
}

#[test]
fn check_doesnt_wait_for_the_lock() {
    let f = fixture("check-unlocked");
    let a = app(&f.t, "a");
    let o = opts("app-a", &a, vec![f.rt1.clone()], &f.mirror);
    run(&f.store, &o).unwrap();
    // another app is installing (it holds the lock, maybe for a long download)
    let held = f.store.lock(&mut || {}).unwrap();
    let start = std::time::Instant::now();
    let mut waited = false;
    let r = ensure(&f.store, &EnsureOptions { check: true, ..clone_opts(&o) }, &mut |e| {
        if let Event::Waiting = e {
            waited = true
        }
    })
    .unwrap();
    assert!(r.complete);
    assert!(!waited);
    assert!(start.elapsed() < std::time::Duration::from_secs(2));
    // and it leaves the other installer's tmp/ alone
    std::fs::write(f.store.tmp_dir().join("dl-other-download"), b"part").unwrap();
    check_of(&f.store, &o);
    assert!(f.store.tmp_dir().join("dl-other-download").exists());
    drop(held);
}

#[test]
fn check_of_a_store_not_there_yet_creates_nothing() {
    let f = fixture("check-no-store");
    let a = app(&f.t, "a");
    let root = f.t.path().join("no-store-yet");
    let store = Store::open_read_only(&root).unwrap();
    let r = check_of(&store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror));
    assert!(!r.complete);
    assert!(!root.exists());
    // an unreadable store.json is an error, as for an install
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("store.json"), b"{").unwrap();
    assert!(Store::open_read_only(&root).is_err());
}

// ---- the store lock's wait (store.rs `lock_within`) ----

#[test]
fn a_lock_wait_gives_up_without_breaking_the_lock() {
    let f = fixture("lock-wait");
    let a = app(&f.t, "a");
    run(&f.store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
    let held = f.store.lock(&mut || {}).unwrap();
    let mut waits = 0;
    let start = std::time::Instant::now();
    let e = f.store.lock_within(std::time::Duration::from_millis(600), &mut || waits += 1).err().unwrap();
    assert!(start.elapsed() >= std::time::Duration::from_millis(600));
    assert_eq!(waits, 1, "the waiting note is said once");
    assert!(e.0.contains("gave up waiting") && e.0.contains("Nothing was changed"), "{e}");
    // still held by its owner, and the store is as it was
    assert!(f.store.try_lock().unwrap().is_none());
    assert_eq!(installed(&f.store), vec!["trt-runtime/11.4"]);
    // once it's released within the limit, a waiter gets it
    let root = f.store.root().to_path_buf();
    let waiter = std::thread::spawn(move || {
        let s = Store::open(&root).unwrap();
        s.lock_within(std::time::Duration::from_secs(30), &mut || {}).map(|_| ()).is_ok()
    });
    std::thread::sleep(std::time::Duration::from_millis(300));
    drop(held);
    assert!(waiter.join().unwrap());
}

#[test]
fn lock_wait_limits() {
    assert_eq!(store::minutes(store::LOCK_WAIT), "60 minutes");
    assert_eq!(store::minutes(UNREF_LOCK_WAIT), "5 minutes");
    assert!(UNREF_LOCK_WAIT < store::LOCK_WAIT);
}

// ---- downloads: timeouts, retries and resume (fetch.rs) ----
// A download runs under the store lock, so a stalled one blocked every other app's install and
// uninstall for good. These run against a local server that misbehaves on purpose.

#[derive(Clone, Copy, Debug)]
enum Serve {
    /// The whole body (or from the Range asked for).
    Whole,
    /// Headers with the full Content-Length, then this many bytes, then the connection is closed.
    DropAfter(usize),
    /// Headers, this many bytes, then nothing (the connection stays open).
    StallAfter(usize),
    /// The whole body with 200, whatever Range was asked for.
    IgnoreRange,
    Status(u16),
}

struct TestServer {
    url: String,
    /// The Range header of each request, in order.
    ranges: std::sync::Arc<std::sync::Mutex<Vec<Option<String>>>>,
}

/// Serves `body` at /pack.zip; the n-th connection is answered as `script[n]` (the last entry for
/// the rest).
fn serve(body: Vec<u8>, script: Vec<Serve>) -> TestServer {
    use std::io::Read as _;
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/pack.zip", l.local_addr().unwrap());
    let ranges = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen = ranges.clone();
    let body = std::sync::Arc::new(body);
    std::thread::spawn(move || {
        for (i, conn) in l.incoming().enumerate() {
            let Ok(mut s) = conn else { continue };
            let how = *script.get(i).or(script.last()).unwrap();
            let (body, seen) = (body.clone(), seen.clone());
            std::thread::spawn(move || {
                let mut req = Vec::new();
                let mut b = [0u8; 1];
                while !req.ends_with(b"\r\n\r\n") && s.read(&mut b).unwrap_or(0) == 1 {
                    req.push(b[0]);
                }
                let req = String::from_utf8_lossy(&req).to_ascii_lowercase();
                let range = req.lines().find_map(|l| l.strip_prefix("range: ")).map(|r| r.trim().to_string());
                seen.lock().unwrap().push(range.clone());
                let from = range
                    .as_deref()
                    .and_then(|r| r.strip_prefix("bytes="))
                    .and_then(|r| r.strip_suffix('-'))
                    .and_then(|n| n.parse::<usize>().ok())
                    .filter(|_| !matches!(how, Serve::IgnoreRange))
                    .unwrap_or(0);
                if let Serve::Status(c) = how {
                    let _ = write!(s, "HTTP/1.1 {c} Nope\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                    return;
                }
                let part = &body[from..];
                let head = if from > 0 {
                    format!("HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {from}-{}/{}\r\n", body.len() - 1, body.len())
                } else {
                    "HTTP/1.1 200 OK\r\n".to_string()
                };
                let _ = write!(s, "{head}Content-Length: {}\r\nConnection: close\r\n\r\n", part.len());
                match how {
                    Serve::DropAfter(n) => {
                        let _ = s.write_all(&part[..n.min(part.len())]);
                    }
                    Serve::StallAfter(n) => {
                        let _ = s.write_all(&part[..n.min(part.len())]);
                        let _ = s.flush();
                        std::thread::sleep(std::time::Duration::from_secs(5));
                    }
                    _ => {
                        let _ = s.write_all(part);
                    }
                }
            });
        }
    });
    TestServer { url, ranges }
}

fn quick() -> fetch::Policy {
    fetch::Policy {
        connect: std::time::Duration::from_secs(5),
        stall: std::time::Duration::from_millis(300),
        retry_delays: vec![std::time::Duration::from_millis(20); 4],
        max_attempts: 20,
    }
}

fn test_body() -> Vec<u8> {
    (0..200_000u32).map(|i| (i * 7 % 251) as u8).collect()
}

fn fetch_to(t: &Tmp, srv: &TestServer, expected: u64) -> (Result<()>, Vec<u8>, Vec<(u64, u64)>) {
    let dest = t.path().join("download.part");
    let mut events = Vec::new();
    let r = fetch::download_with(&srv.url, &dest, expected, &quick(), &mut |got, total| events.push((got, total)));
    (r, std::fs::read(&dest).unwrap_or_default(), events)
}

#[test]
fn a_dropped_download_resumes_where_it_stopped() {
    let t = Tmp::new("dl-drop");
    let body = test_body();
    let srv = serve(body.clone(), vec![Serve::DropAfter(70_000), Serve::Whole]);
    let (r, got, events) = fetch_to(&t, &srv, body.len() as u64);
    r.unwrap();
    assert!(got == body, "the resumed file is the whole file");
    assert_eq!(*srv.ranges.lock().unwrap(), vec![None, Some("bytes=70000-".to_string())]);
    assert_eq!(events.last(), Some(&(body.len() as u64, body.len() as u64)));
}

#[test]
fn a_stalled_download_times_out_and_resumes() {
    let t = Tmp::new("dl-stall");
    let body = test_body();
    let srv = serve(body.clone(), vec![Serve::StallAfter(50_000), Serve::Whole]);
    let start = std::time::Instant::now();
    let (r, got, _) = fetch_to(&t, &srv, body.len() as u64);
    r.unwrap();
    assert!(got == body);
    assert_eq!(*srv.ranges.lock().unwrap(), vec![None, Some("bytes=50000-".to_string())]);
    assert!(start.elapsed() < std::time::Duration::from_secs(4), "it gave up on the stall, not the server");
}

#[test]
fn a_slow_download_that_keeps_coming_doesnt_time_out() {
    // a chunk every 200 ms for 2 s, under the 300 ms stall limit: no retry
    let t = Tmp::new("dl-slow");
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/pack.zip", l.local_addr().unwrap());
    std::thread::spawn(move || {
        use std::io::Read as _;
        let (mut s, _) = l.accept().unwrap();
        let mut b = [0u8; 4096];
        let _ = s.read(&mut b);
        let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Length: 10000\r\nConnection: close\r\n\r\n");
        for _ in 0..10 {
            std::thread::sleep(std::time::Duration::from_millis(200));
            let _ = s.write_all(&[7u8; 1000]);
            let _ = s.flush();
        }
    });
    let dest = t.path().join("slow.part");
    let p = fetch::Policy { retry_delays: Vec::new(), ..quick() };
    fetch::download_with(&url, &dest, 10_000, &p, &mut |_, _| {}).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), vec![7u8; 10_000]);
}

#[test]
fn a_server_without_ranges_starts_over() {
    let t = Tmp::new("dl-norange");
    let body = test_body();
    let srv = serve(body.clone(), vec![Serve::DropAfter(70_000), Serve::IgnoreRange]);
    let (r, got, _) = fetch_to(&t, &srv, body.len() as u64);
    r.unwrap();
    assert!(got == body, "not the first part twice");
    assert_eq!(srv.ranges.lock().unwrap()[1], Some("bytes=70000-".to_string()));
}

#[test]
fn a_download_that_never_comes_gives_up() {
    let t = Tmp::new("dl-dead");
    let srv = serve(test_body(), vec![Serve::StallAfter(0)]);
    let start = std::time::Instant::now();
    let (r, _, _) = fetch_to(&t, &srv, 200_000);
    let e = r.err().unwrap();
    assert!(e.0.contains("no data for") && e.0.contains("gave up after 5 attempts"), "{e}");
    assert_eq!(srv.ranges.lock().unwrap().len(), 5);
    assert!(start.elapsed() < std::time::Duration::from_secs(5));
}

#[test]
fn a_download_making_progress_keeps_retrying() {
    // dropped every 30 kB: more failures than the retries, but each gets further
    let t = Tmp::new("dl-flaky");
    let body = test_body();
    let srv = serve(body.clone(), vec![Serve::DropAfter(30_000)]);
    let (r, got, _) = fetch_to(&t, &srv, body.len() as u64);
    r.unwrap();
    assert!(got == body);
    assert_eq!(srv.ranges.lock().unwrap().len(), 7);
}

#[test]
fn missing_files_and_server_errors() {
    let t = Tmp::new("dl-status");
    // a 404 isn't retried
    let srv = serve(test_body(), vec![Serve::Status(404)]);
    let e = fetch_to(&t, &srv, 200_000).0.err().unwrap();
    assert!(e.0.contains("404"), "{e}");
    assert_eq!(srv.ranges.lock().unwrap().len(), 1);
    // a 503 is, and so is the index fetch
    let body = test_body();
    let srv = serve(body.clone(), vec![Serve::Status(503), Serve::Whole]);
    assert!(fetch_to(&t, &srv, body.len() as u64).1 == body);
    let srv = serve(b"{\"index\": 1}".to_vec(), vec![Serve::Status(502), Serve::DropAfter(3), Serve::Whole]);
    assert_eq!(fetch::fetch_small_with(&srv.url, &quick()).unwrap(), b"{\"index\": 1}");
    assert_eq!(srv.ranges.lock().unwrap().len(), 3);
}

#[test]
fn ensure_installs_a_resumed_download() {
    // the whole path: download (dropped once, resumed), SHA-256 check, extract, install
    let f = fixture("ensure-resume");
    let a = app(&f.t, "a");
    let mut p = f.models.clone();
    let archive = std::fs::read(f.mirror.join(&p.asset)).unwrap();
    let srv = serve(archive.clone(), vec![Serve::DropAfter(archive.len() / 2), Serve::Whole]);
    p.url = srv.url.clone();
    let mut o = opts("app-a", &a, vec![p], &f.mirror);
    o.mirror = None;
    let r = run(&f.store, &o).unwrap();
    assert_eq!(r.downloaded_bytes, archive.len() as u64);
    assert_eq!(installed(&f.store), vec!["models-mangajanai/1.0.0"]);
    assert_eq!(srv.ranges.lock().unwrap()[1], Some(format!("bytes={}-", archive.len() / 2)));
}

// ---- the settings file (settings.rs) ----
// Edited by hand by people who aren't programmers: forgiving where it can be, and an error that
// says where when it can't.

use crate::settings::{self as conf, Source, resolve_with};

fn parse_conf(text: &str) -> Result<conf::Settings> {
    conf::parse(text, Path::new("C:/App/aji-runtime.conf"))
}

#[test]
fn settings_parse_forgivingly() {
    let s = |t: &str| parse_conf(t).unwrap().store;
    assert_eq!(s(""), None);
    assert_eq!(s("store = runtime"), Some("runtime".into()));
    assert_eq!(s("  store=runtime  \r\n"), Some("runtime".into()));
    assert_eq!(s("\u{feff}store = runtime\n"), Some("runtime".into()), "a BOM");
    assert_eq!(s("# a comment\n; another\n\n   # indented\nstore = D:\\aji\n"), Some("D:\\aji".into()));
    assert_eq!(s("store = \"C:\\My Apps\\runtime\""), Some("C:\\My Apps\\runtime".into()), "Copy as path");
    assert_eq!(s("store = \"\"x\"\""), Some("\"x\"".into()), "one pair of quotes only");
    assert_eq!(s("store = \"half"), Some("\"half".into()));
    assert_eq!(s("store = a=b"), Some("a=b".into()), "the value is taken as typed");
    assert_eq!(s("store = C:\\tmp\\new"), Some("C:\\tmp\\new".into()), "no escapes");
    assert_eq!(s("store ="), None, "empty: the per-user store");
    assert_eq!(s("store = \"\""), None);
    assert_eq!(s("Store = x"), Some("x".into()));
    assert_eq!(s("store = a\nstore = b"), Some("b".into()), "the last one counts");
}

#[test]
fn settings_errors_name_the_file_and_line() {
    let e = parse_conf("# mine\n\nstor = runtime\n").unwrap_err().to_string();
    assert!(e.contains("aji-runtime.conf line 3: unknown setting \"stor\" (did you mean \"store\"?)"), "{e}");
    let e = parse_conf("store = x\nrunt1me\n").unwrap_err().to_string();
    assert!(e.contains("aji-runtime.conf line 2") && e.contains("store = <folder>"), "{e}");
    // a missing file is the default; a file that isn't text is an error
    let t = Tmp::new("conf-read");
    assert_eq!(conf::read(&t.path().join("none.conf")).unwrap(), conf::Settings::default());
    std::fs::write(t.path().join("bad.conf"), [0xff, 0xfe, 0x00]).unwrap();
    assert!(conf::read(&t.path().join("bad.conf")).is_err());
}

#[test]
fn written_settings_read_back() {
    let t = Tmp::new("conf-write");
    let f = t.path().join("aji-runtime.conf");
    for v in ["runtime", "..\\shared runtime", "D:\\aji", " padded ", ""] {
        conf::write(&f, v).unwrap();
        let text = std::fs::read_to_string(&f).unwrap();
        assert!(text.lines().filter(|l| l.starts_with('#')).count() >= 2, "a commented file");
        assert_eq!(conf::read(&f).unwrap().store, (!v.is_empty()).then(|| v.to_string()), "{text}");
    }
    assert!(std::fs::read_to_string(&f).unwrap().contains("\nstore = \n"), "--shared keeps the file with `store =`");
}

#[test]
fn the_store_comes_from_flag_env_settings_default_in_that_order() {
    let t = Tmp::new("resolve");
    let app = t.path().join("App");
    std::fs::create_dir_all(&app).unwrap();
    let file = app.join("aji-runtime.conf");
    let user = t.path().join("user-store");
    let (flag, env) = (t.path().join("flag"), t.path().join("env"));
    let r = resolve_with(None, None, &file, &user).unwrap();
    assert_eq!((r.source, &r.root, r.relative, &r.setting), (Source::Default, &user, false, &None), "no file");
    conf::write(&file, "").unwrap();
    assert_eq!(resolve_with(None, None, &file, &user).unwrap().source, Source::Default, "an empty setting");
    conf::write(&file, "runtime").unwrap();
    let r = resolve_with(None, None, &file, &user).unwrap();
    assert_eq!((r.source, r.relative), (Source::Settings, true));
    assert_eq!(r.root, paths::absolute(&app.join("runtime")), "relative to the settings file's folder");
    conf::write(&file, &Path::new("..").join("beside").to_string_lossy()).unwrap();
    assert_eq!(resolve_with(None, None, &file, &user).unwrap().root, paths::absolute(&t.path().join("beside")));
    let abs = t.path().join("custom");
    conf::write(&file, &abs.to_string_lossy()).unwrap();
    let r = resolve_with(None, None, &file, &user).unwrap();
    assert_eq!((r.source, r.relative, r.root.clone()), (Source::Settings, false, abs.clone()));
    let r = resolve_with(None, Some(env.clone().into()), &file, &user).unwrap();
    assert_eq!((r.source, &r.root, r.relative), (Source::Env, &env, false));
    assert_eq!(r.setting.as_deref(), Some(abs.to_string_lossy().as_ref()), "the setting is still reported");
    let r = resolve_with(Some(&flag), Some(env.into()), &file, &user).unwrap();
    assert_eq!((r.source, &r.root), (Source::Flag, &flag));
    assert_eq!(resolve_with(None, Some("".into()), &file, &user).unwrap().source, Source::Settings, "an empty env is unset");
    // a broken settings file stops only what it decides
    std::fs::write(&file, "stor = x\n").unwrap();
    assert!(resolve_with(None, None, &file, &user).unwrap_err().to_string().contains("line 1"));
    assert_eq!(resolve_with(Some(&flag), None, &file, &user).unwrap().source, Source::Flag);
}

#[test]
fn location_json_says_where_the_store_comes_from() {
    let t = Tmp::new("location-json");
    let app = t.path().join("App");
    std::fs::create_dir_all(&app).unwrap();
    let file = app.join("aji-runtime.conf");
    let user = t.path().join("user-store");
    conf::write(&file, "runtime").unwrap();
    let r = resolve_with(None, None, &file, &user).unwrap();
    let j = r.location_json();
    assert_eq!(j["root"], paths::absolute(&app.join("runtime")).to_string_lossy().as_ref());
    assert_eq!((j["moved"].clone(), j["target"].clone()), (false.into(), serde_json::Value::Null));
    assert_eq!(j["available"], true, "not there yet, on a drive that is: created by the first install");
    assert_eq!((j["source"].as_str(), j["setting"].as_str(), j["relative"].as_bool()), (Some("settings"), Some("runtime"), Some(true)));
    assert_eq!(j["settings_file"], file.to_string_lossy().as_ref());
    assert_eq!(j["default_root"], user.to_string_lossy().as_ref());
    let keys: Vec<&String> = j.as_object().unwrap().keys().collect();
    assert_eq!(keys.len(), 9, "{keys:?}");
    let j = resolve_with(None, None, &app.join("none.conf"), &user).unwrap().location_json();
    assert_eq!((j["source"].as_str(), j["setting"].clone(), j["relative"].as_bool()), (Some("default"), serde_json::Value::Null, Some(false)));
    assert_eq!(j["available"], true);
}

#[cfg(windows)]
#[test]
fn a_store_on_an_unplugged_drive_is_unavailable() {
    let t = Tmp::new("location-drive");
    let free = (b'D'..=b'Z').rev().map(|c| format!("{}:\\", c as char)).find(|d| !Path::new(d).exists()).unwrap();
    let file = t.path().join("aji-runtime.conf");
    conf::write(&file, &format!("{free}aji")).unwrap();
    let r = resolve_with(None, None, &file, &t.path().join("user")).unwrap();
    assert_eq!(r.location_json()["available"], false);
    assert!(r.open().is_err(), "and nothing is started elsewhere");
}

#[test]
fn move_and_reset_are_refused_for_a_store_set_in_the_settings_file() {
    let t = Tmp::new("move-settings");
    let file = t.path().join("aji-runtime.conf");
    let user = t.path().join("user-store");
    conf::write(&file, "runtime").unwrap();
    let e = resolve_with(None, None, &file, &user).unwrap().check_movable().unwrap_err().to_string();
    assert!(e.contains("this copy keeps its runtime in") && e.contains("aji-runtime.conf") && e.contains("aji-runtime use"), "{e}");
    for (flag, env) in [(Some(t.path().join("f")), None), (None, Some(t.path().join("e").into_os_string()))] {
        assert!(resolve_with(flag.as_deref(), env, &file, &user).unwrap().check_movable().is_ok());
    }
    assert!(resolve_with(None, None, &t.path().join("none.conf"), &user).unwrap().check_movable().is_ok());
}

// ---- relative mode: the store inside or beside the app's folder ----
// The point: the whole folder can be moved or copied to another PC and keep working. The rules that
// must hold everywhere: a relative app_path is resolved against the store root, by every store.

/// An app folder with a settings file, its exe and libaji folder, and the per-user store beside.
struct Portable {
    f: Fixture,
    app_dir: PathBuf,
    exe: PathBuf,
    aji: PathBuf,
    conf: PathBuf,
    user: PathBuf,
}

fn portable(name: &str) -> Portable {
    let f = fixture(name);
    let app_dir = f.t.path().join("Portable").join("VideoJaNai");
    std::fs::create_dir_all(app_dir.join("inference")).unwrap();
    let exe = app_dir.join("VideoJaNai.exe");
    std::fs::write(&exe, b"exe").unwrap();
    let user = f.store.root().to_path_buf();
    Portable { aji: app_dir.join("inference"), conf: app_dir.join("aji-runtime.conf"), app_dir, exe, user, f }
}

impl Portable {
    fn resolved(&self) -> conf::Resolved {
        resolve_with(None, None, &self.conf, &self.user).unwrap()
    }
    fn opts(&self, packs: Vec<IndexPack>) -> EnsureOptions {
        let mut o = opts("videojanai", &self.exe, packs, &self.f.mirror);
        o.aji_dir = Some(self.aji.clone());
        o
    }
}

fn json_of(dir: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(dir.join("aji-runtime.json")).unwrap()).unwrap()
}

#[test]
fn relative_mode_writes_relative_paths() {
    let p = portable("rel-write");
    conf::write(&p.conf, "runtime").unwrap();
    let store = p.resolved().open().unwrap();
    assert!(store.is_relative());
    run(&store, &p.opts(vec![p.f.rt1.clone(), p.f.models.clone()])).unwrap();
    let r = store.read_ref("videojanai").unwrap();
    assert_eq!(r.app_path, PathBuf::from("..").join("VideoJaNai.exe"));
    assert!(store.refs()[0].live);
    let j = json_of(&p.aji);
    let trt = PathBuf::from("..").join("runtime").join("packs").join("trt-runtime").join("11.4");
    assert_eq!(j["trt"], trt.to_string_lossy().as_ref(), "native separators, relative to the json's folder");
    assert!(p.aji.join(j["models"][0].as_str().unwrap()).join("a.onnx").is_file());
    assert!(check_of(&store, &p.opts(vec![p.f.rt1.clone(), p.f.models.clone()])).complete);
    // outside relative mode: absolute, as before
    let abs = Store::open(store.root()).unwrap();
    run(&abs, &p.opts(vec![p.f.rt1.clone()])).unwrap();
    assert_eq!(abs.read_ref("videojanai").unwrap().app_path, p.exe);
    assert_eq!(json_of(&p.aji)["trt"], abs.pack_dir("trt-runtime", "11.4").to_string_lossy().as_ref());
}

#[test]
fn a_relative_ref_keeps_its_packs_after_the_whole_tree_moves() {
    let p = portable("rel-move");
    conf::write(&p.conf, "runtime").unwrap();
    let o = p.opts(vec![p.f.rt1.clone(), p.f.models.clone()]);
    run(&p.resolved().open().unwrap(), &o).unwrap();
    // the whole folder moved (or copied to another PC)
    let moved = p.f.t.path().join("Elsewhere").join("VJ");
    std::fs::create_dir_all(moved.parent().unwrap()).unwrap();
    std::fs::rename(&p.app_dir, &moved).unwrap();
    let res = resolve_with(None, None, &moved.join("aji-runtime.conf"), &p.user).unwrap();
    let store = res.open().unwrap();
    // gc in that store, opened in any mode, keeps everything
    for s in [Store::open(store.root()).unwrap(), Store::open(store.root()).unwrap().with_relative(true)] {
        let lock = s.lock(&mut || {}).unwrap();
        let gc = s.gc(&lock).unwrap();
        assert!(gc.deleted.is_empty() && gc.dead_refs.is_empty(), "{gc:?}");
    }
    assert_eq!(installed(&store), vec!["models-mangajanai/1.0.0", "trt-runtime/11.4"]);
    // and nothing needs fixing: the check is complete (ref and aji-runtime.json still right)
    let mut o2 = opts("videojanai", &moved.join("VideoJaNai.exe"), o.packs.clone(), &p.f.t.path().join("no-mirror"));
    o2.aji_dir = Some(moved.join("inference"));
    let r = check_of(&store, &o2);
    assert!(r.complete, "{:?}", r.stale);
    assert!(moved.join("inference").join(json_of(&moved.join("inference"))["trt"].as_str().unwrap()).join("nvinfer_11.dll").is_file());
}

#[test]
fn a_relative_ref_whose_exe_is_gone_is_dead() {
    let p = portable("rel-dead");
    conf::write(&p.conf, "runtime").unwrap();
    let store = p.resolved().open().unwrap();
    run(&store, &p.opts(vec![p.f.rt1.clone()])).unwrap();
    std::fs::remove_file(&p.exe).unwrap();
    // even a store opened without relative mode resolves it against its root
    let plain = Store::open(store.root()).unwrap();
    assert!(!plain.refs()[0].live);
    let lock = plain.lock(&mut || {}).unwrap();
    let gc = plain.gc(&lock).unwrap();
    assert_eq!((gc.deleted.clone(), gc.dead_refs.len()), (vec!["trt-runtime/11.4".to_string()], 1));
}

#[test]
fn a_relative_ref_is_live_against_the_store_root_not_the_current_folder() {
    // a ref with a relative path that only exists relative to the current folder is dead
    let f = fixture("rel-cwd");
    let lock = f.store.lock(&mut || {}).unwrap();
    let cwd_file = PathBuf::from("Cargo.toml");
    assert!(cwd_file.exists());
    let r = AppRef { app_id: "x".into(), app_version: String::new(), app_path: cwd_file, packs: vec![] };
    f.store.write_ref(&lock, &r).unwrap();
    assert!(!f.store.refs()[0].live);
}

#[test]
fn check_wants_the_form_this_mode_writes() {
    let p = portable("rel-form");
    conf::write(&p.conf, "runtime").unwrap();
    let rel = p.resolved().open().unwrap();
    let abs = Store::open(rel.root()).unwrap();
    let o = opts("videojanai", &p.exe, vec![p.f.rt1.clone()], &p.f.mirror);
    // an absolute ref for the same exe: an ensure in relative mode would rewrite it
    run(&abs, &o).unwrap();
    assert!(check_of(&abs, &o).complete);
    let r = check_of(&rel, &o);
    assert!(!r.complete && r.stale[0].contains("another app path"), "{:?}", r.stale);
    run(&rel, &o).unwrap();
    assert!(check_of(&rel, &o).complete);
    assert!(!check_of(&abs, &o).complete, "and the other way round");
    // a relative aji-runtime.json is stale outside relative mode, and an absolute one inside it
    let mut o = p.opts(vec![p.f.rt1.clone()]);
    run(&rel, &o).unwrap();
    assert!(check_of(&rel, &o).complete);
    o.app_path = p.exe.clone();
    let mut a = clone_opts(&o);
    a.check = true;
    assert!(run(&abs, &a).unwrap().stale.iter().any(|s| s.contains("aji-runtime.json")));
}

#[test]
fn a_store_on_another_drive_gets_absolute_paths_in_relative_mode() {
    assert_eq!(paths::relative_to(Path::new("/a/b"), Path::new("/a/c/d")), Some(PathBuf::from("..").join("c").join("d")));
    assert_eq!(paths::relative_to(Path::new("/a/b"), Path::new("/a/b")), Some(PathBuf::from(".")));
    #[cfg(windows)]
    {
        assert_eq!(paths::relative_to(Path::new("C:\\a"), Path::new("D:\\a\\x.exe")), None);
        assert_eq!(paths::relative_to(Path::new("C:\\Apps\\VJ"), Path::new("c:\\apps\\vj\\x.exe")), Some(PathBuf::from("x.exe")));
        let s = Store::open_read_only(Path::new("C:\\nowhere\\store")).unwrap().with_relative(true);
        assert_eq!(s.path_form(Path::new("D:\\VJ\\x.exe")), PathBuf::from("D:\\VJ\\x.exe"));
        assert_eq!(s.path_form(Path::new("C:\\nowhere\\x.exe")), PathBuf::from("..\\x.exe"));
    }
}

// ---- per-copy refs (two copies of one app on one PC) ----

#[test]
fn per_copy_ids_differ_per_path() {
    let f = fixture("per-copy-ids");
    let (one, two) = (app(&f.t, "one"), app(&f.t, "two"));
    let a = copy_ref_id(&f.store, "videojanai", &one).unwrap();
    let b = copy_ref_id(&f.store, "videojanai", &two).unwrap();
    assert_ne!(a, b);
    assert!(a.starts_with("videojanai-") && a.len() == "videojanai-".len() + 8, "{a}");
    assert!(a["videojanai-".len()..].chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    // normalized: separators, `.`, a trailing separator (and case on Windows) don't matter
    let messy = PathBuf::from(format!("{}/./app.exe", one.parent().unwrap().display()));
    assert_eq!(copy_ref_id(&f.store, "videojanai", &messy).unwrap(), a);
    #[cfg(windows)]
    assert_eq!(copy_ref_id(&f.store, "videojanai", &PathBuf::from(one.to_string_lossy().to_uppercase())).unwrap(), a);
    // the id is the hash of the path as the ref stores it
    let key = paths::path_key(&one);
    assert_eq!(a, format!("videojanai-{}", &sha256_bytes(key.as_bytes())[..8]));
    // both copies install side by side; neither's ref is the other's
    run(&f.store, &opts("videojanai", &one, vec![f.rt1.clone()], &f.mirror)).unwrap();
    let mut seen = Vec::new();
    for exe in [&one, &two] {
        let r = ensure_per_copy(&f.store, &opts("videojanai", exe, vec![f.rt1.clone()], &f.mirror), &mut |_| {}).unwrap();
        assert!(r.complete);
        seen.push(copy_ref_id(&f.store, "videojanai", exe).unwrap());
    }
    for (id, exe) in seen.iter().zip([&one, &two]) {
        assert_eq!(&f.store.read_ref(id).unwrap().app_path, exe);
        assert_eq!(&f.store.read_ref(id).unwrap().app_id, id);
    }
}

#[test]
fn per_copy_ids_survive_moving_the_tree_in_relative_mode() {
    let p = portable("per-copy-move");
    conf::write(&p.conf, "runtime").unwrap();
    let store = p.resolved().open().unwrap();
    let id = copy_ref_id(&store, "videojanai", &p.exe).unwrap();
    ensure_per_copy(&store, &p.opts(vec![p.f.rt1.clone()]), &mut |_| {}).unwrap();
    let moved = p.f.t.path().join("USB").join("VJ");
    std::fs::create_dir_all(moved.parent().unwrap()).unwrap();
    std::fs::rename(&p.app_dir, &moved).unwrap();
    let store = resolve_with(None, None, &moved.join("aji-runtime.conf"), &p.user).unwrap().open().unwrap();
    let exe = moved.join("VideoJaNai.exe");
    assert_eq!(copy_ref_id(&store, "videojanai", &exe).unwrap(), id);
    let mut o = opts("videojanai", &exe, vec![p.f.rt1.clone()], &p.f.t.path().join("no-mirror"));
    o.aji_dir = Some(moved.join("inference"));
    o.check = true;
    assert!(ensure_per_copy(&store, &o, &mut |_| {}).unwrap().complete);
    // outside relative mode the id is of the absolute path, which a move changes
    let abs = Store::open(store.root()).unwrap();
    assert_ne!(copy_ref_id(&abs, "videojanai", &exe).unwrap(), copy_ref_id(&abs, "videojanai", &p.exe).unwrap());
}

#[test]
fn per_copy_takes_over_the_plain_ref_only_for_the_same_path() {
    let f = fixture("per-copy-migrate");
    let (one, two) = (app(&f.t, "one"), app(&f.t, "two"));
    let o1 = opts("videojanai", &one, vec![f.rt1.clone(), f.models.clone()], &f.mirror);
    // the plain ref, made before the app used per-copy refs, for copy one
    run(&f.store, &o1).unwrap();
    // copy one finds it (show, path, use); copy two doesn't
    let (rid, r) = find_ref(&f.store, "videojanai", Some(&one), true).unwrap().unwrap();
    assert_eq!((rid.as_str(), &r.app_path), ("videojanai", &one));
    assert!(find_ref(&f.store, "videojanai", Some(&two), true).unwrap().is_none());
    assert!(find_ref(&f.store, "videojanai", None, true).is_err(), "--per-copy needs --app-path");
    // a check with --per-copy is incomplete while the plain ref is there
    let r = run(&f.store, &EnsureOptions { check: true, ..clone_opts(&o1) }).unwrap();
    assert!(r.complete, "the plain check is fine");
    let r = ensure_per_copy(&f.store, &EnsureOptions { check: true, ..clone_opts(&o1) }, &mut |_| {}).unwrap();
    assert!(!r.complete && r.stale.iter().any(|s| s.contains("plain ref")), "{:?}", r.stale);
    // copy two's per-copy ensure leaves copy one's plain ref alone
    let o2 = opts("videojanai", &two, vec![f.rt1.clone()], &f.mirror);
    ensure_per_copy(&f.store, &o2, &mut |_| {}).unwrap();
    assert_eq!(f.store.read_ref("videojanai").unwrap().app_path, one);
    assert!(f.store.installed("models-mangajanai", "1.0.0").is_some(), "copy one's packs are kept");
    // copy one's per-copy ensure replaces it: no plain ref, nothing freed
    let r = ensure_per_copy(&f.store, &o1, &mut |_| {}).unwrap();
    assert!(r.gc.deleted.is_empty(), "{:?}", r.gc.deleted);
    assert!(f.store.read_ref("videojanai").is_none());
    let id1 = copy_ref_id(&f.store, "videojanai", &one).unwrap();
    assert_eq!(f.store.read_ref(&id1).unwrap().packs, vec!["trt-runtime/11.4", "models-mangajanai/1.0.0"]);
    assert!(ensure_per_copy(&f.store, &EnsureOptions { check: true, ..clone_opts(&o1) }, &mut |_| {}).unwrap().complete);
    assert_eq!(find_ref(&f.store, "videojanai", Some(&one), true).unwrap().unwrap().0, id1);
}

#[test]
fn unref_per_copy_drops_the_plain_ref_only_for_the_same_path() {
    let f = fixture("per-copy-unref");
    let (one, two) = (app(&f.t, "one"), app(&f.t, "two"));
    // copy two has a per-copy ref; the plain ref is copy one's
    run(&f.store, &opts("videojanai", &one, vec![f.rt1.clone()], &f.mirror)).unwrap();
    ensure_per_copy(&f.store, &opts("videojanai", &two, vec![f.models.clone()], &f.mirror), &mut |_| {}).unwrap();
    let (had, gc) = unref_per_copy(&f.store, "videojanai", &two, &mut || {}).unwrap();
    assert!(had);
    assert_eq!(gc.deleted, vec!["models-mangajanai/1.0.0".to_string()]);
    assert_eq!(f.store.read_ref("videojanai").unwrap().app_path, one, "another path's plain ref is never touched");
    // copy one, uninstalled (its exe already gone) by an uninstaller that passes --per-copy
    std::fs::remove_file(&one).unwrap();
    let (had, gc) = unref_per_copy(&f.store, "videojanai", &one, &mut || {}).unwrap();
    assert!(had);
    assert_eq!(gc.deleted, vec!["trt-runtime/11.4".to_string()]);
    assert!(f.store.refs().is_empty());
}

// ---- taking packs from the per-user store instead of downloading ----

/// Every file under `dir` with its bytes, to see that a store wasn't changed.
fn snapshot(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut files = Vec::new();
    crate::list_files(dir, dir, &mut files);
    files.sort();
    files.into_iter().filter(|f| f != "lock").map(|f| (f.clone(), std::fs::read(dir.join(&f)).unwrap())).collect()
}

#[test]
fn a_custom_store_takes_packs_from_the_per_user_store() {
    let p = portable("take-shared");
    let other = app(&p.f.t, "other");
    run(&p.f.store, &opts("other-app", &other, vec![p.f.rt1.clone(), p.f.models.clone()], &p.f.mirror)).unwrap();
    let before = snapshot(&p.user);
    conf::write(&p.conf, "runtime").unwrap();
    let store = p.resolved().open().unwrap();
    assert_eq!(store.shared_root(), Some(p.user.as_path()));
    // nothing to download (an empty mirror): the runtime comes from the per-user store
    let empty = p.f.t.path().join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let mut o = opts("videojanai", &p.exe, vec![p.f.rt1.clone(), p.f.sm120.clone()], &empty);
    o.check = true;
    let (states, total, r) = plan_states(&store, &o);
    assert_eq!((states, total), (vec![PlanState::Adopted, PlanState::Download], Some(p.f.sm120.bytes)));
    assert!(!r.unwrap().complete);
    o.check = false;
    o.packs = vec![p.f.rt1.clone()];
    let (states, total, r) = plan_states(&store, &o);
    let r = r.unwrap();
    assert_eq!((states, total, r.adopted), (vec![PlanState::Adopted], Some(0), vec!["trt-runtime/11.4".to_string()]));
    let dir = store.pack_dir("trt-runtime", "11.4");
    assert_eq!(std::fs::read(dir.join("nvinfer_11.dll")).unwrap(), b"nvinfer 11.4");
    assert!(store::same_files(&store.installed("trt-runtime", "11.4").unwrap().files, &p.f.rt1.files));
    assert_eq!(std::fs::read_dir(store.tmp_dir()).unwrap().count(), 0);
    // the per-user store is as it was, and its own gc doesn't touch the copy
    assert_eq!(snapshot(&p.user), before);
    unref(&p.f.store, "other-app", &mut || {}).unwrap();
    assert!(installed(&p.f.store).is_empty());
    assert_eq!(std::fs::read(dir.join("nvinfer_11.dll")).unwrap(), b"nvinfer 11.4");
    assert!(check_of(&store, &o).complete);
}

#[test]
fn a_bad_pack_in_the_per_user_store_is_downloaded_instead() {
    let p = portable("take-shared-bad");
    let other = app(&p.f.t, "other");
    run(&p.f.store, &opts("other-app", &other, vec![p.f.rt1.clone(), p.f.models.clone()], &p.f.mirror)).unwrap();
    // one file of the shared runtime is missing (its .pack.json still lists it), the models have other
    // files than this index
    std::fs::remove_file(p.f.store.pack_dir("trt-runtime", "11.4").join("trtexec.exe")).unwrap();
    let models2 = make_pack(&p.f.mirror, "models-mangajanai", "1.0.0", "models", None, &[("a.onnx", b"model a v2")]);
    let before = snapshot(&p.user);
    conf::write(&p.conf, "runtime").unwrap();
    let store = p.resolved().open().unwrap();
    let o = opts("videojanai", &p.exe, vec![p.f.rt1.clone(), models2.clone()], &p.f.mirror);
    let (states, _, r) = plan_states(&store, &EnsureOptions { check: true, ..clone_opts(&o) });
    r.unwrap();
    assert_eq!(states, vec![PlanState::Adopted, PlanState::Download], "the check can't see the missing file");
    let (states, _, r) = plan_states(&store, &o);
    let r = r.unwrap();
    assert_eq!(states, vec![PlanState::Download, PlanState::Download]);
    assert!(r.adopted.is_empty());
    assert_eq!(std::fs::read(store.pack_dir("trt-runtime", "11.4").join("trtexec.exe")).unwrap(), b"trtexec");
    assert_eq!(std::fs::read(store.pack_dir("models-mangajanai", "1.0.0").join("a.onnx")).unwrap(), b"model a v2");
    assert_eq!(snapshot(&p.user), before);
    assert_eq!(std::fs::read_dir(store.tmp_dir()).unwrap().count(), 0);
}

#[test]
fn the_per_user_store_takes_nothing_from_itself() {
    let f = fixture("take-self");
    let a = app(&f.t, "a");
    let store = Store::open(f.store.root()).unwrap().with_shared(Some(f.store.root().to_path_buf()));
    run(&store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror)).unwrap();
    let (states, _, _) = plan_states(&store, &opts("app-a", &a, vec![f.rt1.clone()], &f.mirror));
    assert_eq!(states, vec![PlanState::Installed]);
    // and the CLI's mode: the per-user store gets no shared store
    let t = Tmp::new("take-self-conf");
    let r = resolve_with(None, None, &t.path().join("none.conf"), f.store.root()).unwrap();
    assert!(r.open().unwrap().shared_root().is_none());
}

// ---- `use`: switching a copy's store (switch.rs) ----

use crate::switch::{Target, UseApp, use_store};

fn use_app(p: &Portable, per_copy: bool) -> UseApp {
    UseApp { app_id: "videojanai".into(), per_copy, app_path: p.exe.clone(), aji_dir: Some(p.aji.clone()) }
}

fn use_to(p: &Portable, target: Target, per_copy: bool) -> Result<switch::UseReport> {
    use_store(&p.resolved(), &target, Some(&use_app(p, per_copy)), &mut || {}, &mut |_, _| {})
}

/// The app's check in the store its settings name now: complete with no mirror (nothing to download).
fn copy_check(p: &Portable, packs: Vec<IndexPack>, per_copy: bool) -> EnsureResult {
    let store = p.resolved().open_read_only().unwrap();
    let mut o = p.opts(packs);
    o.mirror = Some(p.f.t.path().join("no-mirror"));
    o.check = true;
    if per_copy { ensure_per_copy(&store, &o, &mut |_| {}).unwrap() } else { run(&store, &o).unwrap() }
}

#[test]
fn use_moves_a_copy_from_the_per_user_store_into_its_folder() {
    let p = portable("use-in");
    let other = app(&p.f.t, "other");
    run(&p.f.store, &opts("other-app", &other, vec![p.f.rt1.clone()], &p.f.mirror)).unwrap();
    let packs = vec![p.f.rt1.clone(), p.f.models.clone()];
    run(&p.f.store, &p.opts(packs.clone())).unwrap();
    let mut seen = Vec::new();
    let r = use_store(&p.resolved(), &Target::Path("runtime".into()), Some(&use_app(&p, false)), &mut || {}, &mut |d, t| {
        seen.push((d, t))
    })
    .unwrap();
    let inside = p.app_dir.join("runtime");
    assert_eq!((r.root.clone(), r.setting.as_str(), r.relative), (paths::absolute(&inside), "runtime", true));
    assert_eq!(r.copied, vec!["trt-runtime/11.4", "models-mangajanai/1.0.0"]);
    let total: u64 = packs.iter().map(|x| x.installed_bytes).sum();
    assert_eq!((seen.first(), seen.last()), (Some(&(0, total)), Some(&(total, total))));
    assert_eq!(conf::read(&p.conf).unwrap().store.as_deref(), Some("runtime"));
    // the new store: the packs, the ref in relative form, aji-runtime.json relative
    let store = p.resolved().open().unwrap();
    assert!(store.is_relative());
    assert_eq!(installed(&store), vec!["models-mangajanai/1.0.0", "trt-runtime/11.4"]);
    assert_eq!(store.read_ref("videojanai").unwrap().app_path, PathBuf::from("..").join("VideoJaNai.exe"));
    assert_eq!(store.read_ref("videojanai").unwrap().app_version, "1.0");
    assert!(json_of(&p.aji)["trt"].as_str().unwrap().starts_with(".."));
    let c = copy_check(&p, packs.clone(), false);
    assert!(c.complete, "{:?}", c.stale);
    // the old store: the ref is gone, and what only this app used with it; the other app keeps its own
    assert!(p.f.store.read_ref("videojanai").is_none());
    assert_eq!(r.old_gc.as_ref().unwrap().deleted, vec!["models-mangajanai/1.0.0".to_string()]);
    assert_eq!(installed(&p.f.store), vec!["trt-runtime/11.4"]);
    assert!(!r.removed_old, "the per-user store is never removed");
    assert!(p.f.store.root().join("store.json").is_file());
    // and the folder moves with everything in it
    let moved = p.f.t.path().join("Moved");
    std::fs::rename(&p.app_dir, &moved).unwrap();
    let res = resolve_with(None, None, &moved.join("aji-runtime.conf"), &p.user).unwrap();
    let mut o = opts("videojanai", &moved.join("VideoJaNai.exe"), packs, &p.f.t.path().join("no-mirror"));
    o.aji_dir = Some(moved.join("inference"));
    assert!(check_of(&res.open_read_only().unwrap(), &o).complete);
}

#[test]
fn use_moves_a_copy_from_its_folder_to_the_per_user_store() {
    let p = portable("use-out");
    conf::write(&p.conf, "runtime").unwrap();
    let packs = vec![p.f.rt1.clone(), p.f.models.clone()];
    ensure_per_copy(&p.resolved().open().unwrap(), &p.opts(packs.clone()), &mut |_| {}).unwrap();
    let inside = p.app_dir.join("runtime");
    let r = use_to(&p, Target::Shared, true).unwrap();
    assert_eq!((r.root.as_path(), r.setting.as_str(), r.relative), (p.user.as_path(), "", false));
    let text = std::fs::read_to_string(&p.conf).unwrap();
    assert!(text.contains("\nstore = \n"), "the file stays, with an empty store: {text}");
    assert_eq!(p.resolved().source, Source::Default);
    // the per-user store has the copy's per-copy ref, absolute, and the packs
    let id = copy_ref_id(&p.f.store, "videojanai", &p.exe).unwrap();
    assert_eq!(r.ref_id.as_deref(), Some(id.as_str()));
    assert_eq!(p.f.store.read_ref(&id).unwrap().app_path, p.exe);
    assert_eq!(json_of(&p.aji)["trt"], p.f.store.pack_dir("trt-runtime", "11.4").to_string_lossy().as_ref());
    assert!(copy_check(&p, packs, true).complete);
    // the emptied store in the app's folder is removed (on Windows; elsewhere its lock file stays)
    assert!(r.removed_old == cfg!(windows));
    if cfg!(windows) {
        assert!(!inside.exists());
    } else {
        assert_eq!(std::fs::read_dir(&inside).unwrap().map(|e| e.unwrap().file_name()).collect::<Vec<_>>(), ["lock"]);
    }
}

#[test]
fn use_moves_a_copy_from_its_folder_to_a_shared_custom_folder() {
    let p = portable("use-custom");
    conf::write(&p.conf, "runtime").unwrap();
    let packs = vec![p.f.rt1.clone()];
    run(&p.resolved().open().unwrap(), &p.opts(packs.clone())).unwrap();
    // something the user put in the old store's folder: that folder stays, with it
    let inside = p.app_dir.join("runtime");
    std::fs::write(inside.join("notes.txt"), b"mine").unwrap();
    let custom = p.f.t.path().join("D").join("AI runtime");
    let r = use_to(&p, Target::Path(custom.to_string_lossy().into_owned()), false).unwrap();
    assert_eq!((r.root.clone(), r.relative), (custom.clone(), false));
    assert_eq!(p.resolved().root, custom);
    let store = p.resolved().open().unwrap();
    assert_eq!(store.read_ref("videojanai").unwrap().app_path, p.exe);
    assert!(copy_check(&p, packs, false).complete);
    assert!(!r.removed_old);
    assert!(inside.join("notes.txt").is_file());
    assert!(!inside.join("store.json").exists() && !inside.join("packs").exists(), "the store's own files went");
    // a second copy of the app pointed at the same folder shares it
    let second = p.f.t.path().join("Second").join("VideoJaNai.exe");
    std::fs::create_dir_all(second.parent().unwrap()).unwrap();
    std::fs::write(&second, b"exe").unwrap();
    let conf2 = second.parent().unwrap().join("aji-runtime.conf");
    conf::write(&conf2, &custom.to_string_lossy()).unwrap();
    let s2 = resolve_with(None, None, &conf2, &p.user).unwrap().open().unwrap();
    let empty = p.f.t.path().join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let (states, _, r) = plan_states(&s2, &opts("videojanai", &second, vec![p.f.rt1.clone()], &empty));
    r.unwrap();
    assert_eq!(states, vec![PlanState::Installed]);
}

#[test]
fn use_rewrites_the_same_store_in_another_form() {
    let p = portable("use-same");
    let inside = p.app_dir.join("runtime");
    conf::write(&p.conf, &inside.to_string_lossy()).unwrap();
    let packs = vec![p.f.rt1.clone()];
    let abs = p.resolved().open().unwrap();
    assert!(!abs.is_relative());
    ensure_per_copy(&abs, &p.opts(packs.clone()), &mut |_| {}).unwrap();
    let old_id = copy_ref_id(&abs, "videojanai", &p.exe).unwrap();
    let r = use_to(&p, Target::Path("runtime".into()), true).unwrap();
    assert!(r.copied.is_empty() && r.old_gc.is_none());
    let store = p.resolved().open().unwrap();
    assert!(store.is_relative());
    let new_id = copy_ref_id(&store, "videojanai", &p.exe).unwrap();
    assert_ne!(new_id, old_id);
    assert!(store.read_ref(&old_id).is_none());
    assert_eq!(store.read_ref(&new_id).unwrap().app_path, PathBuf::from("..").join("VideoJaNai.exe"));
    assert_eq!(store.refs().len(), 1);
    assert!(json_of(&p.aji)["trt"].as_str().unwrap().starts_with(".."));
    assert_eq!(installed(&store), vec!["trt-runtime/11.4"]);
    assert!(copy_check(&p, packs.clone(), true).complete);
    // and back to absolute
    use_to(&p, Target::Path(inside.to_string_lossy().into_owned()), true).unwrap();
    assert!(!p.resolved().relative);
    assert!(copy_check(&p, packs, true).complete);
}

#[test]
fn use_without_a_ref_only_writes_the_settings_file() {
    let p = portable("use-no-ref");
    let r = use_store(&p.resolved(), &Target::Path("runtime".into()), None, &mut || {}, &mut |_, _| {}).unwrap();
    assert!(r.ref_id.is_none());
    assert_eq!(p.resolved().source, Source::Settings);
    assert!(!p.app_dir.join("runtime").exists(), "created by the first install");
    // with an app id but no ref, or with only another copy's plain ref, the same
    let other = app(&p.f.t, "other-copy");
    run(&p.f.store, &opts("videojanai", &other, vec![p.f.rt1.clone()], &p.f.mirror)).unwrap();
    let p2 = Portable { conf: p.f.t.path().join("c2.conf"), ..p };
    let r = use_to(&p2, Target::Path("rt2".into()), false).unwrap();
    assert!(r.ref_id.is_none());
    assert_eq!(p2.f.store.read_ref("videojanai").unwrap().app_path, other, "the other copy's ref is left alone");
}

#[test]
fn use_refuses_bad_targets_and_changes_nothing() {
    let p = portable("use-refused");
    run(&p.f.store, &p.opts(vec![p.f.rt1.clone()])).unwrap();
    let full = p.app_dir.join("full");
    std::fs::create_dir_all(&full).unwrap();
    std::fs::write(full.join("x.txt"), b"x").unwrap();
    let e = use_to(&p, Target::Path("full".into()), false).unwrap_err().to_string();
    assert!(e.contains("isn't empty"), "{e}");
    let e = use_to(&p, Target::Path(p.user.join("packs").join("new").to_string_lossy().into_owned()), false).unwrap_err().to_string();
    assert!(e.contains("inside"), "{e}");
    let e = use_to(&p, Target::Path(".".into()), false).unwrap_err().to_string();
    assert!(e.contains("isn't empty"), "the app's own folder: {e}");
    #[cfg(windows)]
    {
        let e = use_to(&p, Target::Path("\\\\server\\share\\aji".into()), false).unwrap_err().to_string();
        assert!(e.contains("network"), "{e}");
    }
    assert!(!p.conf.exists());
    assert!(p.f.store.read_ref("videojanai").is_some());
    // a store whose root is a parent of the current one
    conf::write(&p.conf, &Path::new("runtime").join("inner").to_string_lossy()).unwrap();
    let e = use_to(&p, Target::Path("runtime".into()), false).unwrap_err().to_string();
    assert!(e.contains("inside"), "{e}");
    // an existing store is fine to switch to; so is the per-user path given as a folder (= --shared)
    conf::write(&p.conf, "").unwrap();
    let r = use_to(&p, Target::Path(p.user.to_string_lossy().into_owned()), false).unwrap();
    assert_eq!((r.setting.as_str(), r.root.as_path()), ("", p.user.as_path()));
}

#[test]
fn a_failed_use_leaves_the_copy_working() {
    let p = portable("use-failed");
    let packs = vec![p.f.rt1.clone(), p.f.models.clone()];
    run(&p.f.store, &p.opts(packs.clone())).unwrap();
    let json_before = std::fs::read(p.aji.join("aji-runtime.json")).unwrap();
    // aji-runtime.json can't be written (its folder is gone): undone after the packs were copied
    let bad = UseApp { aji_dir: Some(p.app_dir.join("no-such-folder")), ..use_app(&p, false) };
    let e = use_store(&p.resolved(), &Target::Path("runtime".into()), Some(&bad), &mut || {}, &mut |_, _| {});
    assert!(e.is_err());
    assert!(!p.conf.exists(), "the settings file wasn't written");
    let inside = Store::open(&p.app_dir.join("runtime")).unwrap();
    assert!(inside.read_ref("videojanai").is_none(), "the new store's ref was undone");
    assert!(p.f.store.read_ref("videojanai").is_some(), "the old ref is there");
    assert_eq!(std::fs::read(p.aji.join("aji-runtime.json")).unwrap(), json_before);
    assert!(copy_check(&p, packs.clone(), false).complete, "the copy still works on the per-user store");
    // the packs copied in meanwhile are harmless: unreferenced, freed by the next gc
    assert_eq!(installed(&inside).len(), 2);
    let lock = inside.lock(&mut || {}).unwrap();
    assert_eq!(inside.gc(&lock).unwrap().deleted.len(), 2);
    drop(lock);
    // a pack the old store lacks: refused before anything is written
    std::fs::remove_file(p.f.store.pack_dir("models-mangajanai", "1.0.0").join(".pack.json")).unwrap();
    let e = use_to(&p, Target::Path("runtime".into()), false).unwrap_err().to_string();
    assert!(e.contains("isn't complete") && e.contains("Nothing was changed"), "{e}");
    assert!(!p.conf.exists());
    assert!(p.f.store.read_ref("videojanai").is_some());
    assert!(inside.read_ref("videojanai").is_none());
}

#[test]
fn opposite_uses_dont_deadlock() {
    // two copies of the app switching in opposite directions at once: each holds one store's lock at a
    // time, so both finish
    let t = Tmp::new("use-opposite");
    let user = t.path().join("user");
    let mirror = t.path().join("mirror");
    let rt = make_pack(&mirror, "trt-runtime", "11.4", "trt", None, &[("nvinfer_11.dll", b"nvinfer 11.4")]);
    let custom = t.path().join("custom");
    let mk = |name: &str, conf_value: &str| {
        let dir = t.path().join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let exe = dir.join("app.exe");
        std::fs::write(&exe, b"exe").unwrap();
        let conf_file = dir.join("aji-runtime.conf");
        conf::write(&conf_file, conf_value).unwrap();
        let res = resolve_with(None, None, &conf_file, &user).unwrap();
        ensure_per_copy(&res.open().unwrap(), &opts("app", &exe, vec![rt.clone()], &mirror), &mut |_| {}).unwrap();
        (conf_file, exe)
    };
    let a = mk("a", "");
    let b = mk("b", &custom.to_string_lossy());
    let mut handles = Vec::new();
    for ((conf_file, exe), start_shared) in [(a, true), (b, false)] {
        let (user, custom) = (user.clone(), custom.clone());
        handles.push(std::thread::spawn(move || {
            let app = UseApp { app_id: "app".into(), per_copy: true, app_path: exe, aji_dir: None };
            for round in 0..4 {
                let to_custom = start_shared == (round % 2 == 0);
                let target = if to_custom { Target::Path(custom.to_string_lossy().into_owned()) } else { Target::Shared };
                let res = resolve_with(None, None, &conf_file, &user).unwrap();
                let r = use_store(&res, &target, Some(&app), &mut || {}, &mut |_, _| {}).unwrap();
                assert!(r.ref_id.is_some(), "round {round}: the ref went along");
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(Store::open(&user).unwrap().refs().len(), 1);
    assert_eq!(Store::open(&custom).unwrap().refs().len(), 1);
}

#[cfg(windows)]
#[test]
fn an_emptied_store_keeps_a_lock_file_someone_has_open() {
    // a process waiting in Store::lock has the lock file open (std shares delete): deleting it would
    // let that waiter and a third process both hold "the" lock
    let t = Tmp::new("remove-lock-open");
    let root = t.path().join("store");
    let store = Store::open(&root).unwrap();
    let lock = store.lock(&mut || {}).unwrap();
    let waiter =
        std::fs::File::options().read(true).write(true).create(true).truncate(false).open(root.join("lock")).unwrap();
    assert!(!store.remove_if_empty(lock));
    assert!(root.join("lock").is_file(), "the lock file stays");
    assert!(root.is_dir(), "and so does the folder");
    assert!(!root.join("store.json").exists(), "the store's other files went");
    drop(waiter);
    // nobody else has it open: the lock file and the folder go
    let store = Store::open(&root).unwrap();
    let lock = store.lock(&mut || {}).unwrap();
    assert!(store.remove_if_empty(lock));
    assert!(!root.exists());
}

// ---- `unref --app-path` without `--per-copy`: only this copy's plain ref ----
// An older portable copy (from before per-copy ids) may have taken the plain id over: the installed
// copy's uninstaller must not drop that copy's ref and free the packs it's using.

#[test]
fn unref_for_path_drops_only_this_copys_ref() {
    let f = fixture("unref-for-path");
    let (installed_copy, portable_copy) = (app(&f.t, "installed"), app(&f.t, "portable"));
    // the portable copy took the plain ref over
    run(&f.store, &opts("mpv-animejanai", &portable_copy, vec![f.rt1.clone(), f.models.clone()], &f.mirror)).unwrap();
    let (had, other, gc) = unref_for_path(&f.store, "mpv-animejanai", &installed_copy, &mut || {}).unwrap();
    assert!(!had);
    assert_eq!(other.as_deref(), Some(portable_copy.as_path()));
    assert!(gc.deleted.is_empty(), "{:?}", gc.deleted);
    assert_eq!(f.store.read_ref("mpv-animejanai").unwrap().app_path, portable_copy, "left as it is");
    assert_eq!(installed(&f.store), vec!["models-mangajanai/1.0.0", "trt-runtime/11.4"]);
    // the ref naming this copy (its exe already removed by the uninstaller) is dropped, its packs freed
    std::fs::remove_file(&portable_copy).unwrap();
    let (had, other, gc) = unref_for_path(&f.store, "mpv-animejanai", &portable_copy, &mut || {}).unwrap();
    assert!(had && other.is_none());
    assert_eq!(gc.deleted, vec!["models-mangajanai/1.0.0".to_string(), "trt-runtime/11.4".to_string()]);
    assert!(f.store.read_ref("mpv-animejanai").is_none());
    // no ref at all: nothing to drop, nothing left alone
    let (had, other, _) = unref_for_path(&f.store, "mpv-animejanai", &installed_copy, &mut || {}).unwrap();
    assert!(!had && other.is_none());
}

#[test]
fn unref_for_path_resolves_a_relative_ref_against_the_store_root() {
    let p = portable("unref-for-path-rel");
    conf::write(&p.conf, "runtime").unwrap();
    let store = p.resolved().open().unwrap();
    run(&store, &p.opts(vec![p.f.rt1.clone()])).unwrap();
    assert!(store.read_ref("videojanai").unwrap().app_path.is_relative());
    // another exe: left alone, reported by its resolved path
    let other = app(&p.f.t, "other");
    let (had, left, _) = unref_for_path(&store, "videojanai", &other, &mut || {}).unwrap();
    assert!(!had);
    assert!(paths::same_path(&left.unwrap(), &p.exe));
    assert!(store.read_ref("videojanai").is_some());
    // this copy's exe, as an absolute path: the relative ref names it
    let (had, left, gc) = unref_for_path(&store, "videojanai", &p.exe, &mut || {}).unwrap();
    assert!(had && left.is_none());
    assert_eq!(gc.deleted, vec!["trt-runtime/11.4".to_string()]);
}

#[test]
fn plain_unref_still_drops_the_ref_whatever_it_names() {
    // released uninstallers call `unref --app-id ID` alone
    let f = fixture("unref-plain-any");
    let other = app(&f.t, "other");
    run(&f.store, &opts("mpv-animejanai", &other, vec![f.rt1.clone()], &f.mirror)).unwrap();
    let (had, gc) = unref(&f.store, "mpv-animejanai", &mut || {}).unwrap();
    assert!(had);
    assert_eq!(gc.deleted, vec!["trt-runtime/11.4".to_string()]);
}
