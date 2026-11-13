//! The `aji-runtime` program: the pack store for app installers and uninstallers.
//!
//! ```text
//! aji-runtime ensure --app-id ID [--per-copy] --app-path EXE --index URL|FILE --index-sha256 HEX
//!                    [--want trt,trtrtx,PACK...] [--gpus auto|none|smNN,...] [--aji-dir DIR]
//!                    [--app-version V] [--adopt-from DIR]... [--mirror DIR] [--check]
//!                    [--progress lines] [--store DIR]
//! aji-runtime adopt  --from DIR ...ensure's options      (ensure, taking matching old files first)
//! aji-runtime unref  --app-id ID [[--per-copy] --app-path EXE]  (uninstallers: drop the ref, then gc)
//! aji-runtime gc
//! aji-runtime list
//! aji-runtime path   NAME[/VERSION] [--app-id ID [--per-copy --app-path EXE]]
//! aji-runtime show   --app-id ID [--per-copy --app-path EXE]   (JSON: the app's packs, for its updater)
//! aji-runtime location [--json]                          (where the store is, and if it's there)
//! aji-runtime move   FOLDER|--default [--progress lines] (the per-user store to another drive)
//! aji-runtime reset                                      (drop a moved store whose folder is gone)
//! aji-runtime use    FOLDER|--shared [--app-id ID [--per-copy] --app-path EXE [--aji-dir DIR]]
//!                    [--progress lines]                  (switch the store this copy uses)
//! ```
//!
//! Every command takes `--store DIR`. The store is `--store` > `AJI_RUNTIME_STORE` > the settings
//! file `aji-runtime.conf` beside this program (`AJI_RUNTIME_CONF` names another) > the per-user
//! store ([`aji_runtime::settings`]).
//!
//! `--progress lines` adds one event per stdout line, fields separated by single spaces:
//! `AJI_WAIT` (another installer holds the store), `AJI_PLAN <name>/<version> <bytes>
//! installed|adopted|download`, `AJI_TOTAL <bytes>`, `AJI_DOWNLOAD <name>/<version> <got> <total>`,
//! `AJI_EXTRACT <name>/<version>`, `AJI_MOVE <copied> <total>` (bytes, while `move` or `use`
//! copies), `AJI_DONE`. A failure prints `error: ...` on stderr and exits 1
//! without `AJI_DONE`. Human-readable progress always goes to stderr.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use aji_runtime::nvml::{self, Gpu};
use aji_runtime::settings::{Resolved, Source};
use aji_runtime::{EnsureOptions, Event, Index, PlanState};

const USAGE: &str = "usage:
  aji-runtime ensure --app-id ID [--per-copy] --app-path EXE --index URL|FILE --index-sha256 HEX
                     [--want trt,trtrtx,PACK...] [--gpus auto|none|smNN,...] [--aji-dir DIR]
                     [--app-version V] [--adopt-from DIR]... [--mirror DIR] [--check]
                     [--progress lines] [--store DIR]
  aji-runtime adopt  --from DIR (and ensure's options)
  aji-runtime unref  --app-id ID [[--per-copy] --app-path EXE] [--store DIR]
  aji-runtime gc     [--store DIR]
  aji-runtime list   [--store DIR]
  aji-runtime path   NAME[/VERSION] [--app-id ID [--per-copy --app-path EXE]] [--store DIR]
  aji-runtime show   --app-id ID [--per-copy --app-path EXE] [--store DIR]
  aji-runtime location [--json] [--store DIR]
  aji-runtime move   FOLDER|--default [--progress lines] [--store DIR]
  aji-runtime reset  [--store DIR]
  aji-runtime use    FOLDER|--shared [--app-id ID [--per-copy] --app-path EXE [--aji-dir DIR]]
                     [--progress lines] [--store DIR]

The store is --store, else AJI_RUNTIME_STORE, else the `store` setting in aji-runtime.conf beside
this program (AJI_RUNTIME_CONF names another file), else the per-user store.";

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
    json: bool,
    to_default: bool,
    per_copy: bool,
    shared: bool,
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
            "--json" => a.json = true,
            "--default" => a.to_default = true,
            "--per-copy" => a.per_copy = true,
            "--shared" => a.shared = true,
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

/// The store this command uses (`--store` > `AJI_RUNTIME_STORE` > the settings file > per-user).
fn resolved(a: &Args) -> Result<Resolved, String> {
    aji_runtime::settings::resolve(a.store.as_deref()).map_err(|e| e.to_string())
}

fn open_store(a: &Args) -> Result<aji_runtime::Store, String> {
    resolved(a)?.open().map_err(|e| e.to_string())
}

/// `--app-path`, required with `--per-copy`.
fn copy_path(a: &Args) -> Result<Option<&std::path::Path>, String> {
    match (a.per_copy, &a.app_path) {
        (true, None) => Err("--per-copy needs --app-path".into()),
        (_, p) => Ok(p.as_deref()),
    }
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
        let old = if g.trt_capable() { "" } else { "; too old for TensorRT, skipped" };
        eprintln!("GPU: {} (compute {}.{}{old})", g.name, g.major, g.minor);
    }
    let want = if a.want.is_empty() { vec!["trt".to_string()] } else { a.want.clone() };
    let packs = index.select(&want, &gpus).map_err(|e| e.to_string())?;
    // a check creates nothing (not even an empty store) and takes no lock
    let res = resolved(a)?;
    let store = if a.check { res.open_read_only() } else { res.open() }.map_err(|e| e.to_string())?;
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
    let run = if a.per_copy { aji_runtime::ensure_per_copy } else { aji_runtime::ensure };
    let r = run(&store, &opts, &mut |ev| match ev {
        Event::Waiting => {
            waiting_note(aji_runtime::store::LOCK_WAIT);
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
        for s in &r.stale {
            eprintln!("  out of date: {s}");
        }
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

/// Said once when another app holds the store lock.
fn waiting_note(limit: std::time::Duration) {
    eprintln!(
        "Waiting for another installer to finish with the store (giving up after {})...",
        aji_runtime::store::minutes(limit)
    );
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
                store.resolve_path(&x.app_path).display(),
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
                Some(id) => aji_runtime::find_ref(&store, id, copy_path(a)?, a.per_copy)
                    .map_err(|e| e.to_string())?
                    .map(|(_, r)| r.packs)
                    .unwrap_or_default(),
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
/// `{"store", "format", "app_id", "ref_id", "has_ref", "app_version", "app_path", "packs": [{"name",
/// "version", "kind", "dir", "installed"}]}`. `app_path` (null without a ref) is the path the ref
/// keeps the packs for, absolute: a ref naming another path than the app's own (a moved AppImage) is
/// dead once that path is gone, so the app ensures again. `ref_id` is the ref read: with
/// `--per-copy`, the copy's own, or the plain `app_id` ref while that still names this copy's path.
/// Changes nothing (a missing store reads as empty).
fn show(a: &Args) -> Result<(), String> {
    let id = a.app_id.clone().ok_or("--app-id is required")?;
    aji_runtime::store::check_component("app id", &id).map_err(|e| e.to_string())?;
    let res = resolved(a)?;
    let app_path = copy_path(a)?;
    let store = res.open_existing();
    // the id this copy's ref has (or would have): the store's mode decides its form
    let looked = res.open_read_only().map_err(|e| e.to_string())?;
    let found = aji_runtime::find_ref(store.as_ref().unwrap_or(&looked), &id, app_path, a.per_copy).map_err(|e| e.to_string())?;
    let ref_id = match (&found, app_path) {
        (Some((rid, _)), _) => rid.clone(),
        (None, Some(p)) if a.per_copy => aji_runtime::copy_ref_id(&looked, &id, p).map_err(|e| e.to_string())?,
        _ => id.clone(),
    };
    let r = found.map(|(_, r)| r);
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
                    "dir": aji_runtime::paths::absolute(&s.pack_dir(name, version)).to_string_lossy(),
                    "installed": meta.is_some(),
                })
            })
            .collect(),
        _ => Vec::new(),
    };
    let app_path = |r: &aji_runtime::AppRef| aji_runtime::paths::absolute(&looked.resolve_path(&r.app_path)).to_string_lossy().into_owned();
    let out = serde_json::json!({
        "store": res.root.to_string_lossy(),
        "format": store.as_ref().map(|s| s.format()),
        "app_id": id,
        "ref_id": ref_id,
        "has_ref": r.is_some(),
        "app_version": r.as_ref().map(|r| r.app_version.clone()),
        "app_path": r.as_ref().map(app_path),
        "packs": packs,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
    Ok(())
}

/// `location`: `default <root>`, `moved <root> -> <target> (available|missing)`, or `settings <root>
/// (from <file>)`; with `--json`, `{"root", "moved", "target", "available", "source", "setting",
/// "relative", "settings_file", "default_root"}`. Changes nothing.
fn location(a: &Args) -> Result<(), String> {
    let res = resolved(a)?;
    if a.json {
        println!("{}", serde_json::to_string_pretty(&res.location_json()).unwrap());
        return Ok(());
    }
    let (moved, target, available) = res.location();
    let root = res.root.display();
    let state = if available { "available" } else { "missing" };
    match (res.source, moved, target) {
        (Source::Settings, false, _) => {
            let missing = if available { "" } else { " (missing)" };
            println!("settings {root} (from {}){missing}", res.settings_file.display())
        }
        (_, true, Some(t)) => println!("moved {root} -> {} ({state})", t.display()),
        _ => println!("default {root}"),
    }
    Ok(())
}

/// `move FOLDER` / `move --default`.
fn move_store(a: &Args) -> Result<(), String> {
    let res = resolved(a)?;
    res.check_movable().map_err(|e| e.to_string())?;
    let root = res.root.clone();
    let lines = a.lines;
    let mut waiting = || {
        say(lines, format_args!("AJI_WAIT"));
        waiting_note(aji_runtime::store::LOCK_WAIT);
    };
    let mut progress = copy_progress(lines, "Copying the store");
    let r = match (a.to_default, a.positional.as_slice()) {
        (true, []) => aji_runtime::location::move_back(&root, &mut waiting, &mut progress),
        (false, [dest]) => aji_runtime::location::move_to(&root, std::path::Path::new(dest), &mut waiting, &mut progress),
        _ => return Err(format!("move needs a folder, or --default
{USAGE}")),
    };
    r.map_err(|e| e.to_string())?;
    match aji_runtime::location::location(&root) {
        aji_runtime::location::Location::Moved { target, .. } => eprintln!("The store is now in {} ({} links there).", target.display(), root.display()),
        aji_runtime::location::Location::Default => eprintln!("The store is in {}.", root.display()),
    }
    say(lines, format_args!("AJI_DONE"));
    Ok(())
}

/// `AJI_MOVE <copied> <total>`, and a note every 100 MB on stderr.
fn copy_progress(lines: bool, what: &'static str) -> impl FnMut(u64, u64) {
    let mut last = 0u64;
    move |done: u64, total: u64| {
        say(lines, format_args!("AJI_MOVE {done} {total}"));
        if done == 0 || done == total || done - last >= 100_000_000 {
            last = done;
            eprintln!("{what}: {} of {}", mb(done), mb(total));
        }
    }
}

/// `use FOLDER` / `use --shared`: switches the store this copy uses (its settings file), taking the
/// app's packs along.
fn use_store(a: &Args) -> Result<(), String> {
    use aji_runtime::switch::{Target, UseApp, use_store};
    let target = match (a.shared, a.positional.as_slice()) {
        (true, []) => Target::Shared,
        (false, [p]) => Target::Path(p.clone()),
        _ => return Err(format!("use needs a folder, or --shared\n{USAGE}")),
    };
    let app = match &a.app_id {
        Some(id) => Some(UseApp {
            app_id: id.clone(),
            per_copy: a.per_copy,
            app_path: a.app_path.clone().ok_or("--app-id needs --app-path")?,
            aji_dir: a.aji_dir.clone(),
        }),
        None if a.per_copy || a.aji_dir.is_some() => return Err("--per-copy and --aji-dir go with --app-id".into()),
        None => None,
    };
    let res = resolved(a)?;
    let lines = a.lines;
    let mut waiting = || {
        say(lines, format_args!("AJI_WAIT"));
        waiting_note(aji_runtime::store::LOCK_WAIT);
    };
    let mut progress = copy_progress(lines, "Copying the runtime");
    let r = use_store(&res, &target, app.as_ref(), &mut waiting, &mut progress).map_err(|e| e.to_string())?;
    for w in &r.warnings {
        eprintln!("Note: {w}");
    }
    if let Some(g) = &r.old_gc {
        report_gc(g);
    }
    if r.removed_old {
        eprintln!("Removed the empty store {}", res.root.display());
    }
    eprintln!("This copy now uses the store {} (set in {}).", r.root.display(), res.settings_file.display());
    if matches!(res.source, Source::Env | Source::Flag) {
        eprintln!("Note: {} still overrides the setting while it's given.", if res.source == Source::Env { "AJI_RUNTIME_STORE" } else { "--store" });
    }
    say(lines, format_args!("AJI_DONE"));
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
            let mut wait = || waiting_note(aji_runtime::UNREF_LOCK_WAIT);
            let (had, other, gc) = match copy_path(&a)? {
                Some(p) if a.per_copy => aji_runtime::unref_per_copy(&store, &id, p, &mut wait).map(|(h, g)| (h, None, g)),
                // only this copy's ref: another copy may have the plain id (an older portable one)
                Some(p) => aji_runtime::unref_for_path(&store, &id, p, &mut wait),
                None => aji_runtime::unref(&store, &id, &mut wait).map(|(h, g)| (h, None, g)),
            }
            .map_err(|e| e.to_string())?;
            match other {
                Some(o) => eprintln!("{id}'s ref is for {}: left as it is", o.display()),
                None if !had => eprintln!("{id} had no ref"),
                None => {}
            }
            report_gc(&gc);
            Ok(())
        }
        "gc" => {
            let store = open_store(&a)?;
            let lock = store.lock(&mut || waiting_note(aji_runtime::store::LOCK_WAIT)).map_err(|e| e.to_string())?;
            report_gc(&store.gc(&lock).map_err(|e| e.to_string())?);
            Ok(())
        }
        "list" => list(&a),
        "path" => path(&a),
        "show" => show(&a),
        "location" => location(&a),
        "move" => move_store(&a),
        "reset" => {
            let res = resolved(&a)?;
            res.check_movable().map_err(|e| e.to_string())?;
            aji_runtime::location::reset(&res.root).map_err(|e| e.to_string())?;
            eprintln!("The store is at the default path {} (empty until an app installs its packs again).", res.root.display());
            Ok(())
        }
        "use" => use_store(&a),
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
