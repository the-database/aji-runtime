//! The `aji-runtime` program end to end: the settings file, `location --json`, `move`/`reset`
//! refusals, and `ensure` / `show` / `use` with `--per-copy`. The per-user store is a temporary
//! folder (LOCALAPPDATA / XDG_DATA_HOME for the child), never the real one.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

struct Env {
    dir: PathBuf,
}

impl Env {
    fn new(name: &str) -> Env {
        let dir = std::env::temp_dir().join(format!("aji-runtime-cli-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("App").join("inference")).unwrap();
        std::fs::write(dir.join("App").join("VideoJaNai.exe"), b"exe").unwrap();
        Env { dir }
    }
    fn app(&self) -> PathBuf {
        self.dir.join("App")
    }
    fn exe(&self) -> PathBuf {
        self.app().join("VideoJaNai.exe")
    }
    fn conf(&self) -> PathBuf {
        self.app().join("aji-runtime.conf")
    }
    /// The per-user store as the child sees it.
    fn user_store(&self) -> PathBuf {
        let base = if cfg!(windows) { self.dir.join("local") } else { self.dir.join("xdg") };
        base.join("the-database").join("aji")
    }
    fn run(&self, args: &[&str]) -> (bool, String, String) {
        let out = Command::new(env!("CARGO_BIN_EXE_aji-runtime"))
            .args(args)
            .env_remove("AJI_RUNTIME_STORE")
            .env("AJI_RUNTIME_CONF", self.conf())
            .env("LOCALAPPDATA", self.dir.join("local"))
            .env("XDG_DATA_HOME", self.dir.join("xdg"))
            .output()
            .unwrap();
        (out.status.success(), String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned())
    }
    fn json(&self, args: &[&str]) -> serde_json::Value {
        let (ok, out, e) = self.run(args);
        assert!(ok, "{args:?}: {e}");
        serde_json::from_str(&out).unwrap()
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[test]
fn location_reports_the_settings_store() {
    let e = Env::new("location");
    let j = e.json(&["location", "--json"]);
    assert_eq!((j["source"].as_str(), j["relative"].as_bool()), (Some("default"), Some(false)));
    assert_eq!(j["root"], e.user_store().to_string_lossy().as_ref());
    assert_eq!(j["default_root"], e.user_store().to_string_lossy().as_ref());
    assert_eq!(j["settings_file"], e.conf().to_string_lossy().as_ref());
    assert!(j["setting"].is_null());
    std::fs::write(e.conf(), "\u{feff}# mine\nstore = \"runtime\"\n").unwrap();
    let j = e.json(&["location", "--json"]);
    assert_eq!((j["source"].as_str(), j["setting"].as_str(), j["relative"].as_bool()), (Some("settings"), Some("runtime"), Some(true)));
    assert_eq!(j["root"], e.app().join("runtime").to_string_lossy().as_ref());
    assert_eq!((j["moved"].as_bool(), j["available"].as_bool()), (Some(false), Some(true)));
    assert!(j["target"].is_null());
    let (ok, out, _) = e.run(&["location"]);
    assert!(ok && out.starts_with("settings ") && out.contains(&format!("(from {})", e.conf().display())), "{out}");
    // --store and AJI_RUNTIME_STORE come first
    let flag = e.dir.join("flagged");
    let j = e.json(&["location", "--json", "--store", &flag.to_string_lossy()]);
    assert_eq!((j["source"].as_str(), j["setting"].as_str()), (Some("flag"), Some("runtime")));
}

#[test]
fn a_bad_settings_file_is_an_error_naming_the_line() {
    let e = Env::new("bad-conf");
    std::fs::write(e.conf(), "# mine\nstor = runtime\n").unwrap();
    let (ok, _, err) = e.run(&["location"]);
    assert!(!ok);
    assert!(err.contains("aji-runtime.conf line 2: unknown setting \"stor\" (did you mean \"store\"?)"), "{err}");
}

#[test]
fn move_and_reset_are_refused_for_a_settings_store() {
    let e = Env::new("move-refused");
    std::fs::write(e.conf(), "store = runtime\n").unwrap();
    for args in [vec!["move", "--default"], vec!["reset"], vec!["move", "C:\\elsewhere"]] {
        let (ok, _, err) = e.run(&args);
        assert!(!ok, "{args:?}");
        assert!(err.contains("this copy keeps its runtime in") && err.contains("aji-runtime use"), "{err}");
    }
}

/// A one-pack index and its mirror; returns (index file, its SHA-256, mirror).
fn index(dir: &Path) -> (PathBuf, String, PathBuf) {
    let mirror = dir.join("mirror");
    std::fs::create_dir_all(&mirror).unwrap();
    let files: &[(&str, &[u8])] = &[("nvinfer_11.dll", b"nvinfer 11.4"), ("trtexec.exe", b"trtexec")];
    let archive = mirror.join("trt-runtime-11.4.zip");
    let mut z = zip::ZipWriter::new(std::fs::File::create(&archive).unwrap());
    for (n, d) in files {
        z.start_file(*n, zip::write::SimpleFileOptions::default()).unwrap();
        z.write_all(d).unwrap();
    }
    z.finish().unwrap();
    let idx = serde_json::json!({
        "format": 1, "tag": "runtime-test", "platform": aji_runtime::index::platform(),
        "packs": [{
            "name": "trt-runtime", "version": "11.4", "kind": "trt", "asset": "trt-runtime-11.4.zip",
            "url": "https://invalid.example/trt-runtime-11.4.zip",
            "sha256": aji_runtime::sha256_file(&archive).unwrap(),
            "bytes": std::fs::metadata(&archive).unwrap().len(),
            "files": files.iter().map(|(n, d)| serde_json::json!({"path": n, "sha256": aji_runtime::sha256_bytes(d), "bytes": d.len()})).collect::<Vec<_>>(),
        }]
    });
    let bytes = serde_json::to_vec(&idx).unwrap();
    let file = dir.join("packs.json");
    std::fs::write(&file, &bytes).unwrap();
    (file, aji_runtime::sha256_bytes(&bytes), mirror)
}

#[test]
fn a_portable_copy_installs_inside_and_switches_to_the_shared_store() {
    let e = Env::new("portable");
    let (idx, pin, mirror) = index(&e.dir);
    let exe = e.exe().to_string_lossy().into_owned();
    let aji = e.app().join("inference").to_string_lossy().into_owned();
    let (idx, mirror) = (idx.to_string_lossy().into_owned(), mirror.to_string_lossy().into_owned());
    let ensure = |extra: &[&str]| {
        let mut args = vec![
            "ensure", "--app-id", "videojanai", "--per-copy", "--app-path", &exe, "--index", &idx, "--index-sha256", &pin,
            "--want", "trt-runtime", "--gpus", "none", "--aji-dir", &aji, "--progress", "lines",
        ];
        args.extend_from_slice(extra);
        e.run(&args)
    };
    std::fs::write(e.conf(), "store = runtime\n").unwrap();
    let (ok, out, err) = ensure(&["--mirror", &mirror]);
    assert!(ok, "{err}");
    assert!(out.contains("AJI_PLAN trt-runtime/11.4 ") && out.contains(" download\n") && out.ends_with("AJI_DONE\n"), "{out}");
    assert!(e.app().join("runtime").join("packs").join("trt-runtime").join("11.4").join("nvinfer_11.dll").is_file());
    let runtime_json: serde_json::Value = serde_json::from_slice(&std::fs::read(e.app().join("inference").join("aji-runtime.json")).unwrap()).unwrap();
    assert!(runtime_json["trt"].as_str().unwrap().starts_with(".."), "{runtime_json}");

    // show: the per-copy ref, absolute paths
    let j = e.json(&["show", "--app-id", "videojanai", "--per-copy", "--app-path", &exe]);
    assert_eq!((j["app_id"].as_str(), j["has_ref"].as_bool()), (Some("videojanai"), Some(true)));
    let ref_id = j["ref_id"].as_str().unwrap().to_string();
    assert!(ref_id.starts_with("videojanai-") && ref_id.len() == 19, "{ref_id}");
    assert_eq!(j["app_path"], exe.as_str());
    let dir = PathBuf::from(j["packs"][0]["dir"].as_str().unwrap());
    assert!(dir.is_absolute() && dir.join("trtexec.exe").is_file(), "{dir:?}");
    let (ok, _, err) = e.run(&["show", "--app-id", "videojanai", "--per-copy"]);
    assert!(!ok && err.contains("--app-path"), "{err}");
    // path finds the copy's pack through its ref
    let (ok, out, _) = e.run(&["path", "trt-runtime", "--app-id", "videojanai", "--per-copy", "--app-path", &exe]);
    assert!(ok && Path::new(out.trim()).join("nvinfer_11.dll").is_file(), "{out}");
    let (ok, out, _) = ensure(&["--check"]);
    assert!(ok && out.ends_with("AJI_DONE\n"), "complete: {out}");

    // use --shared: the packs go along without a download, the folder inside goes
    let (ok, out, err) = e.run(&["use", "--shared", "--app-id", "videojanai", "--per-copy", "--app-path", &exe, "--aji-dir", &aji, "--progress", "lines"]);
    assert!(ok, "{err}");
    assert!(out.lines().any(|l| l.starts_with("AJI_MOVE ")) && out.ends_with("AJI_DONE\n"), "{out}");
    assert_eq!(e.json(&["location", "--json"])["source"], "default");
    let j = e.json(&["show", "--app-id", "videojanai", "--per-copy", "--app-path", &exe]);
    assert_eq!((j["has_ref"].as_bool(), j["store"].as_str()), (Some(true), Some(e.user_store().to_string_lossy().as_ref())));
    assert_ne!(j["ref_id"].as_str().unwrap(), ref_id, "the id of the absolute path now");
    let (ok, out, _) = ensure(&["--check"]);
    assert!(ok && out.contains("AJI_PLAN trt-runtime/11.4 ") && out.ends_with("AJI_DONE\n"), "complete in the shared store: {out}");
    if cfg!(windows) {
        assert!(!e.app().join("runtime").exists());
    }

    // and back inside: hard links from the shared store, and unref drops what's left
    let (ok, _, err) = e.run(&["use", "rt", "--app-id", "videojanai", "--per-copy", "--app-path", &exe, "--aji-dir", &aji]);
    assert!(ok, "{err}");
    let (ok, out, _) = ensure(&["--check"]);
    assert!(ok && out.ends_with("AJI_DONE\n"), "{out}");
    let (ok, _, err) = e.run(&["unref", "--app-id", "videojanai", "--per-copy", "--app-path", &exe]);
    assert!(ok && err.contains("Removed trt-runtime/11.4"), "{err}");
    assert!(!e.app().join("rt").join("packs").join("trt-runtime").exists());
}
