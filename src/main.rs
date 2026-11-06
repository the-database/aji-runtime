//! The `aji-runtime` program: the pack store for app installers and uninstallers.
//!
//! ```text
//! aji-runtime ensure --app-id ID --app-path EXE --index URL|FILE --index-sha256 HEX
//!                    [--want trt,trtrtx,PACK...] [--gpus auto|none|smNN,...] [--aji-dir DIR]
//!                    [--app-version V] [--adopt-from DIR]... [--mirror DIR] [--check]
//!                    [--progress lines] [--store DIR]
//! aji-runtime adopt  --from DIR ...ensure's options      (ensure, taking matching old files first)
//! aji-runtime unref  --app-id ID [--store DIR]           (uninstallers: drop the app's ref, then gc)
//! aji-runtime gc     [--store DIR]
//! aji-runtime list   [--store DIR]
//! aji-runtime path   NAME[/VERSION] [--app-id ID] [--store DIR]
//! aji-runtime show   --app-id ID [--store DIR]           (JSON: the app's packs, for its updater)
//! ```
//!
//! `--progress lines` adds one event per stdout line, fields separated by single spaces:
//! `AJI_WAIT` (another installer holds the store), `AJI_PLAN <name>/<version> <bytes>
//! installed|adopted|download`, `AJI_TOTAL <bytes>`, `AJI_DOWNLOAD <name>/<version> <got> <total>`,
//! `AJI_EXTRACT <name>/<version>`, `AJI_DONE`. A failure prints `error: ...` on stderr and exits 1
//! without `AJI_DONE`. Human-readable progress always goes to stderr.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use aji_runtime::nvml::{self, Gpu};
use aji_runtime::{EnsureOptions, Event, Index, PlanState, Store};

const USAGE: &str = "usage:
  aji-runtime ensure --app-id ID --app-path EXE --index URL|FILE --index-sha256 HEX
                     [--want trt,trtrtx,PACK...] [--gpus auto|none|smNN,...] [--aji-dir DIR]
                     [--app-version V] [--adopt-from DIR]... [--mirror DIR] [--check]
                     [--progress lines] [--store DIR]
  aji-runtime adopt  --from DIR (and ensure's options)
  aji-runtime unref  --app-id ID [--store DIR]
  aji-runtime gc     [--store DIR]
  aji-runtime list   [--store DIR]
  aji-runtime path   NAME[/VERSION] [--app-id ID] [--store DIR]
  aji-runtime show   --app-id ID [--store DIR]";

#[derive(Default)]
struct Args {
    cmd: String,
    positional: Vec<String>,
    app_id: Option<String>,
    app_path: Option<PathBuf>,
    app_version: String,
    index: Option<String>,
    index_sha256: Option<String>,
    want: Vec<String>,
    gpus: String,
    aji_dir: Option<PathBuf>,
    adopt_from: Vec<PathBuf>,
    mirror: Option<PathBuf>,
    check: bool,
    lines: bool,
    store: Option<PathBuf>,
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let mut a = Args { gpus: "auto".into(), ..Default::default() };
    a.cmd = it.next().ok_or(USAGE)?;
    while let Some(x) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{x} needs a value"));
        match x.as_str() {
            "--app-id" => a.app_id = Some(val()?),
            "--app-path" => a.app_path = Some(PathBuf::from(val()?)),
            "--app-version" => a.app_version = val()?,
            "--index" => a.index = Some(val()?),
            "--index-sha256" => a.index_sha256 = Some(val()?),
            "--want" => a.want.extend(val()?.split(',').map(str::trim).filter(|s| !s.is_empty()).map(String::from)),
            "--gpus" => a.gpus = val()?,
            "--aji-dir" => a.aji_dir = Some(PathBuf::from(val()?)),
            "--adopt-from" | "--from" => a.adopt_from.push(PathBuf::from(val()?)),
            "--mirror" => a.mirror = Some(PathBuf::from(val()?)),
            "--check" => a.check = true,
            "--progress" => {
                let v = val()?;
                if v != "lines" {
                    return Err(format!("--progress {v}: only `lines` is supported"));
                }
                a.lines = true;
            }
            "--store" => a.store = Some(PathBuf::from(val()?)),
            "-h" | "--help" => return Err(USAGE.into()),
            s if s.starts_with("--") => return Err(format!("unknown option {s}\n{USAGE}")),
            _ => a.positional.push(x),
        }
    }
    Ok(a)
}

fn gpus(spec: &str) -> Result<Vec<Gpu>, String> {
    match spec {
        "auto" => Ok(nvml::gpus()),
        "none" => Ok(Vec::new()),
        list => list
            .split(',')
            .map(|s| {
                let d = s.trim().strip_prefix("sm").filter(|d| d.len() >= 2 && d.chars().all(|c| c.is_ascii_digit()));
                let d = d.ok_or_else(|| format!("--gpus: {s:?} is not smNN"))?;
                let (major, minor) = d.split_at(d.len() - 1);
                Ok(Gpu { name: s.trim().into(), major: major.parse().unwrap(), minor: minor.parse().unwrap() })
            })
            .collect(),
    }
}

fn mb(b: u64) -> String {
    format!("{:.0} MB", b as f64 / 1e6)
}

fn say(lines: bool, s: std::fmt::Arguments) {
    if lines {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{s}");
        let _ = out.flush();
    }
}

fn open_store(a: &Args) -> Result<Store, String> {
    let root = match &a.store {
        Some(p) => p.clone(),
        None => Store::default_root().map_err(|e| e.to_string())?,
    };
    Store::open(&root).map_err(|e| e.to_string())
}

fn ensure(a: &Args) -> Result<(), String> {
    let app_id = a.app_id.clone().ok_or("--app-id is required")?;
    let app_path = a.app_path.clone().ok_or("--app-path is required")?;
    let index_src = a.index.as_deref().ok_or("--index is required")?;
    let pin = a.index_sha256.as_deref().ok_or("--index-sha256 is required")?;
    let index = Index::load(index_src, Some(pin)).map_err(|e| e.to_string())?;
    if index.platform != aji_runtime::index::platform() {
        return Err(format!("the index is for {}, not {}", index.platform, aji_runtime::index::platform()));
    }
    let gpus = gpus(&a.gpus)?;
    for g in &gpus {
        eprintln!("GPU: {} (compute {}.{})", g.name, g.major, g.minor);
    }
    let want = if a.want.is_empty() { vec!["trt".to_string()] } else { a.want.clone() };
    let packs = index.select(&want, &gpus).map_err(|e| e.to_string())?;
    let store = open_store(a)?;
    eprintln!("Store: {}", store.root().display());
    let opts = EnsureOptions {
        app_id,
        app_version: a.app_version.clone(),
        app_path,
        packs,
        mirror: a.mirror.clone(),
        adopt_from: a.adopt_from.clone(),
        aji_dir: a.aji_dir.clone(),
        check: a.check,
    };
    let lines = a.lines;
    let tty = std::io::IsTerminal::is_terminal(&std::io::stderr());
    let mut shown = (String::new(), u64::MAX);
    let r = aji_runtime::ensure(&store, &opts, &mut |ev| match ev {
        Event::Waiting => {
            eprintln!("Waiting for another installer to finish with the store...");
            say(lines, format_args!("AJI_WAIT"));
        }
        Event::Plan { pack, state } => {
            let s = match state {
                PlanState::Installed => "installed",
                PlanState::Adopted => "adopted",
                PlanState::Download => "download",
            };
            eprintln!("  {:<36} {:>8}  {s}", pack.key(), mb(pack.bytes));
            say(lines, format_args!("AJI_PLAN {} {} {s}", pack.key(), pack.bytes));
        }
        Event::Total(t) => {
            eprintln!("To download: {}", mb(t));
            say(lines, format_args!("AJI_TOTAL {t}"));
        }
        Event::Download { pack, got, total } => {
            say(lines, format_args!("AJI_DOWNLOAD {} {got} {total}", pack.key()));
            let pct = (got * 100).checked_div(total).unwrap_or(0);
            let step = if tty { pct } else { pct / 25 * 25 };
            if shown.0 != pack.name || shown.1 != step {
                shown = (pack.name.clone(), step);
                if tty {
                    eprint!("\r  {}: {pct:>3}%  {} / {}   ", pack.name, mb(got), mb(total));
                    if got >= total {
                        eprintln!();
                    }
                } else {
                    eprintln!("  {}: {pct}%", pack.name);
                }
            }
        }
        Event::Extract { pack } => {
            eprintln!("  {}: extracting", pack.name);
            say(lines, format_args!("AJI_EXTRACT {}", pack.key()));
        }
        Event::Done => say(lines, format_args!("AJI_DONE")),
    })
    .map_err(|e| e.to_string())?;
    if a.check {
        eprintln!("{}", if r.complete { "Everything is installed." } else { "Not everything is installed." });
        if r.complete {
            say(lines, format_args!("AJI_DONE"));
        }
        return Ok(());
    }
    report_gc(&r.gc);
    if let Some(d) = &a.aji_dir {
        eprintln!("Wrote {}", d.join("aji-runtime.json").display());
    }
    Ok(())
}

fn report_gc(g: &aji_runtime::GcReport) {
    if g.newer_format {
        eprintln!("The store was written by a newer aji-runtime: nothing was removed.");
    }
    for k in &g.deleted {
        eprintln!("Removed {k} (no installed app uses it)");
    }
    for k in &g.in_use {
        eprintln!("Kept {k} for now: it's in use (removed by a later cleanup)");
    }
    for r in &g.dead_refs {
        eprintln!("Dropped {} (its app is gone)", r.display());
    }
}

fn list(a: &Args) -> Result<(), String> {
    let store = open_store(a)?;
    let refs = store.refs();
    println!("Store: {} (format {})", store.root().display(), store.format());
    let mut total = 0;
    for (m, dir, bytes) in store.packs() {
        total += bytes;
        let key = aji_runtime::store::pack_key(&m.name, &m.version);
        let users: Vec<&str> = refs
            .iter()
            .filter_map(|r| r.app_ref.as_ref().filter(|x| x.packs.contains(&key)).map(|x| x.app_id.as_str()))
            .collect();
        println!("{key:<40} {:>9}  {}  {}", mb(bytes), if users.is_empty() { "-".into() } else { users.join(",") }, dir.display());
    }
    println!("Total: {}", mb(total));
    for r in refs {
        match r.app_ref {
            Some(x) => println!(
                "ref {} {} {} ({})",
                x.app_id,
                x.app_version,
                x.app_path.display(),
                if r.live { "live" } else { "app gone" }
            ),
            None => println!("ref {} (unreadable)", r.file.display()),
        }
    }
    Ok(())
}

fn path(a: &Args) -> Result<(), String> {
    let store = open_store(a)?;
    let want = a.positional.first().ok_or("path needs a pack name")?;
    let key = match want.split_once('/') {
        Some(_) => want.clone(),
        None => {
            let candidates: Vec<String> = match &a.app_id {
                Some(id) => store.read_ref(id).map(|r| r.packs).unwrap_or_default(),
                None => store.packs().into_iter().map(|(m, _, _)| aji_runtime::store::pack_key(&m.name, &m.version)).collect(),
            };
            let hits: Vec<String> = candidates.into_iter().filter(|k| k.split('/').next() == Some(want.as_str())).collect();
            match hits.len() {
                1 => hits[0].clone(),
                0 => return Err(format!("no installed pack {want}")),
                _ => return Err(format!("several versions of {want}: {}; name one or pass --app-id", hits.join(", "))),
            }
        }
    };
    let (name, version) = key.split_once('/').unwrap();
    if store.installed(name, version).is_none() {
        return Err(format!("{key} is not installed"));
    }
    println!("{}", store.pack_dir(name, version).display());
    Ok(())
}

/// `show`: the app's ref as JSON on stdout, for an updater to read what this app has:
/// `{"store", "format", "app_id", "has_ref", "app_version", "packs": [{"name", "version", "kind",
/// "dir", "installed"}]}`. Changes nothing (a missing store reads as empty).
fn show(a: &Args) -> Result<(), String> {
    let id = a.app_id.clone().ok_or("--app-id is required")?;
    aji_runtime::store::check_component("app id", &id).map_err(|e| e.to_string())?;
    let root = match &a.store {
        Some(p) => p.clone(),
        None => Store::default_root().map_err(|e| e.to_string())?,
    };
    let store = Store::open_existing(&root);
    let r = store.as_ref().and_then(|s| s.read_ref(&id));
    let packs: Vec<serde_json::Value> = match (&store, &r) {
        (Some(s), Some(r)) => r
            .packs
            .iter()
            .filter_map(|k| k.split_once('/'))
            .map(|(name, version)| {
                let meta = s.installed(name, version);
                serde_json::json!({
                    "name": name,
                    "version": version,
                    "kind": meta.as_ref().map(|m| m.kind.clone()).unwrap_or_default(),
                    "dir": s.pack_dir(name, version).to_string_lossy(),
                    "installed": meta.is_some(),
                })
            })
            .collect(),
        _ => Vec::new(),
    };
    let out = serde_json::json!({
        "store": root.to_string_lossy(),
        "format": store.as_ref().map(|s| s.format()),
        "app_id": id,
        "has_ref": r.is_some(),
        "app_version": r.as_ref().map(|r| r.app_version.clone()),
        "packs": packs,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
    Ok(())
}

fn run() -> Result<(), String> {
    let a = parse()?;
    match a.cmd.as_str() {
        "ensure" => ensure(&a),
        "adopt" => {
            if a.adopt_from.is_empty() {
                return Err("adopt needs --from".into());
            }
            ensure(&a)
        }
        "unref" => {
            let id = a.app_id.clone().ok_or("--app-id is required")?;
            let store = open_store(&a)?;
            let (had, gc) = aji_runtime::unref(&store, &id, &mut || eprintln!("Waiting for another installer to finish with the store..."))
                .map_err(|e| e.to_string())?;
            if !had {
                eprintln!("{id} had no ref");
            }
            report_gc(&gc);
            Ok(())
        }
        "gc" => {
            let store = open_store(&a)?;
            let lock = store.lock(&mut || eprintln!("Waiting for another installer to finish with the store...")).map_err(|e| e.to_string())?;
            report_gc(&store.gc(&lock).map_err(|e| e.to_string())?);
            Ok(())
        }
        "list" => list(&a),
        "path" => path(&a),
        "show" => show(&a),
        "-h" | "--help" | "help" => {
            println!("{USAGE}");
            Ok(())
        }
        c => Err(format!("unknown command {c}\n{USAGE}")),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
