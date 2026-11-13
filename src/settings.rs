//! Which store a copy of an app uses: `aji-runtime.conf` beside the `aji-runtime` executable.
//!
//! The order is `--store DIR` > `AJI_RUNTIME_STORE` > the settings file > the per-user store
//! ([`Store::user_root`]). The settings file is plain text for people to edit by hand:
//!
//! ```text
//! # comment lines start with # or ;
//! store = runtime
//! ```
//!
//! `key = value` lines, blank lines and comments; a UTF-8 BOM is fine; the value is taken as typed,
//! trimmed, with one pair of surrounding double quotes removed (Explorer's "Copy as path"), and no
//! escapes. An empty value, or no file, means the per-user store. The one key is `store`; any other
//! line is an error naming the file and line. A relative `store` is relative to the settings file's
//! folder, and puts the store in relative mode ([`Store::with_relative`]).

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use crate::{Result, Store, err, paths};

/// The settings file's name, beside the executable.
pub const FILE_NAME: &str = "aji-runtime.conf";

/// The one setting.
pub const STORE_KEY: &str = "store";

/// What the settings file says: the raw `store` value, or None (no file, or an empty value).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Settings {
    pub store: Option<String>,
}

/// Parses a settings file's text; `file` is only for the error messages.
pub fn parse(text: &str, file: &Path) -> Result<Settings> {
    let name = file.display();
    let mut out = Settings::default();
    for (i, line) in text.strip_prefix('\u{feff}').unwrap_or(text).lines().enumerate() {
        let n = i + 1;
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') || t.starts_with(';') {
            continue;
        }
        let Some((k, v)) = t.split_once('=') else {
            return Err(err(format!("{name} line {n}: expected `{STORE_KEY} = <folder>`, found {t:?}")));
        };
        let k = k.trim();
        if !k.eq_ignore_ascii_case(STORE_KEY) {
            return Err(err(format!("{name} line {n}: unknown setting {k:?} (did you mean \"{STORE_KEY}\"?)")));
        }
        let v = v.trim();
        let v = v.strip_prefix('"').and_then(|x| x.strip_suffix('"')).unwrap_or(v);
        out.store = if v.is_empty() { None } else { Some(v.to_string()) };
    }
    Ok(out)
}

/// Reads a settings file; a missing one is the default.
pub fn read(file: &Path) -> Result<Settings> {
    match std::fs::read(file) {
        Ok(b) => {
            let text = String::from_utf8(b).map_err(|_| err(format!("{}: not UTF-8 text", file.display())))?;
            parse(&text, file)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Settings::default()),
        Err(e) => Err(err(format!("{}: {e}", file.display()))),
    }
}

/// The settings file's text for a `store` value ("" for the per-user store).
pub fn text(store: &str) -> String {
    let value = if store.trim() != store { format!("\"{store}\"") } else { store.to_string() };
    format!(
        "# Where this copy of the app keeps its NVIDIA TensorRT runtime and models (aji-runtime).\n\
         # store = a folder; a relative one is relative to this file's folder, so the app's folder\n\
         #         can be moved or copied to another PC with the runtime in it.\n\
         # Empty: the per-user store that all apps share.\n\
         # Change it with `aji-runtime use`, which takes what's installed along.\n\
         {STORE_KEY} = {value}\n"
    )
}

/// Writes the settings file whole (a rename, so a reader sees the old file or the new one).
pub fn write(file: &Path, store: &str) -> Result<()> {
    crate::store::write_atomic(file, text(store).as_bytes())
}

/// Where the settings file is: `AJI_RUNTIME_CONF`, else `aji-runtime.conf` beside the executable.
pub fn settings_file() -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("AJI_RUNTIME_CONF").filter(|p| !p.is_empty()) {
        return Ok(paths::absolute(Path::new(&p)));
    }
    let exe = std::env::current_exe().map_err(|e| err(format!("can't find this program's folder: {e}")))?;
    let dir = exe.parent().ok_or_else(|| err("can't find this program's folder"))?;
    Ok(dir.join(FILE_NAME))
}

/// Where the store came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Default,
    Settings,
    Env,
    Flag,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Default => "default",
            Source::Settings => "settings",
            Source::Env => "env",
            Source::Flag => "flag",
        }
    }
}

/// The store a command uses, and why.
#[derive(Debug, Clone)]
pub struct Resolved {
    pub root: PathBuf,
    pub source: Source,
    /// The settings file's raw `store` value (whatever the source), or None.
    pub setting: Option<String>,
    /// Relative mode: the store came from the settings file with a relative path.
    pub relative: bool,
    /// Where the settings file is, or would be.
    pub settings_file: PathBuf,
    /// The per-user store ([`Store::user_root`]).
    pub default_root: PathBuf,
}

/// The folder a `store` value names: relative to the settings file's folder.
pub fn store_dir(value: &str, settings_file: &Path) -> PathBuf {
    let v = Path::new(value);
    if v.is_absolute() {
        paths::normalize(v)
    } else {
        paths::absolute(&settings_file.parent().unwrap_or(Path::new(".")).join(v))
    }
}

/// Whether a `store` value is relative (relative mode).
pub fn is_relative_value(value: &str) -> bool {
    let v = Path::new(value);
    !v.is_absolute() && !v.has_root()
}

/// [`resolve`] with its inputs given: `flag` (`--store`), `env` (`AJI_RUNTIME_STORE`), the settings
/// file and the per-user store. A broken settings file is an error only when it's the one deciding.
pub fn resolve_with(flag: Option<&Path>, env: Option<OsString>, settings_file: &Path, default_root: &Path) -> Result<Resolved> {
    let env = env.filter(|e| !e.is_empty());
    let read = read(settings_file);
    let decides = flag.is_none() && env.is_none();
    let setting = match read {
        Ok(s) => s.store,
        Err(e) if decides => return Err(e),
        Err(_) => None,
    };
    let base = Resolved {
        root: PathBuf::new(),
        source: Source::Default,
        setting: setting.clone(),
        relative: false,
        settings_file: settings_file.to_path_buf(),
        default_root: default_root.to_path_buf(),
    };
    Ok(match (flag, env, &setting) {
        (Some(f), _, _) => Resolved { root: f.to_path_buf(), source: Source::Flag, ..base },
        (None, Some(e), _) => Resolved { root: PathBuf::from(e), source: Source::Env, ..base },
        (None, None, Some(v)) => {
            Resolved { root: store_dir(v, settings_file), source: Source::Settings, relative: is_relative_value(v), ..base }
        }
        (None, None, None) => Resolved { root: default_root.to_path_buf(), ..base },
    })
}

/// The store a command uses (`flag` is `--store`).
pub fn resolve(flag: Option<&Path>) -> Result<Resolved> {
    resolve_with(flag, std::env::var_os("AJI_RUNTIME_STORE"), &settings_file()?, &Store::user_root()?)
}

impl Resolved {
    /// Whether this is the per-user store.
    pub fn is_default_store(&self) -> bool {
        paths::same_dir(&self.root, &self.default_root)
    }

    /// The mode a store handle gets: relative mode, and the per-user store to take packs from.
    fn mode(&self, s: Store) -> Store {
        let shared = (!self.is_default_store()).then(|| self.default_root.clone());
        s.with_relative(self.relative).with_shared(shared)
    }

    /// Opens (creating if needed) the store.
    pub fn open(&self) -> Result<Store> {
        Store::open(&self.root).map(|s| self.mode(s))
    }

    /// The store for reading only ([`Store::open_read_only`]).
    pub fn open_read_only(&self) -> Result<Store> {
        Store::open_read_only(&self.root).map(|s| self.mode(s))
    }

    /// The store if it's there ([`Store::open_existing`]).
    pub fn open_existing(&self) -> Option<Store> {
        Store::open_existing(&self.root).map(|s| self.mode(s))
    }

    /// `move` and `reset` work on the per-user store (or one named by `--store` /
    /// `AJI_RUNTIME_STORE`); a store set in the settings file is changed with `use`.
    pub fn check_movable(&self) -> Result<()> {
        if self.source == Source::Settings {
            return Err(err(format!(
                "this copy keeps its runtime in {} (set in {}); change it with `aji-runtime use`",
                self.root.display(),
                self.settings_file.display()
            )));
        }
        Ok(())
    }

    /// `location --json`: `{"root", "moved", "target", "available", "source", "setting", "relative",
    /// "settings_file", "default_root"}`.
    pub fn location_json(&self) -> serde_json::Value {
        let (moved, target, available) = self.location();
        serde_json::json!({
            "root": self.root.to_string_lossy(),
            "moved": moved,
            "target": target.map(|t| t.to_string_lossy().into_owned()),
            "available": available,
            "source": self.source.as_str(),
            "setting": self.setting,
            "relative": self.relative,
            "settings_file": self.settings_file.to_string_lossy(),
            "default_root": self.default_root.to_string_lossy(),
        })
    }

    /// (moved, target, available). The per-user store (or a link) is as [`crate::location::location`]
    /// says; another store is unavailable only when neither it nor its drive is there (an unplugged
    /// drive): one that isn't there yet on a drive that is gets created by the first install.
    pub fn location(&self) -> (bool, Option<PathBuf>, bool) {
        use crate::location::{Location, location};
        match location(&self.root) {
            Location::Moved { target, available } => (true, Some(target), available),
            Location::Default if self.is_default_store() => (false, None, true),
            Location::Default => (false, None, self.root.exists() || paths::drive_exists(&self.root)),
        }
    }
}
