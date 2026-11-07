//! Where the store lives. By default at [`Store::default_root`](crate::Store::default_root);
//! [`move_to`] puts it on another drive and leaves a link at the default path (a directory junction
//! on Windows, a symlink elsewhere), so every app, old versions included, keeps reaching it through
//! the same path: their `aji-runtime.json` pack paths stay valid and nothing resolves the link.
//!
//! - [`location`] says where it is and whether that's reachable (a removable drive may be gone).
//! - [`move_to`] moves it to a new or empty folder; [`move_back`] moves it back to the default path.
//! - [`reset`] drops a link whose target is missing, leaving an empty store at the default path.
//!
//! A move copies under the store lock, then renames the old folder aside before switching the link:
//! on Windows that rename fails while an app has a file loaded from the store (a DLL), which stops
//! the move with nothing changed. Only then is the old copy deleted.

use std::fs;
use std::path::{Path, PathBuf};

use crate::{Error, Result, Store, err};

/// Where the store at `root` (the default path) is.
#[derive(Debug, Clone, PartialEq)]
pub enum Location {
    /// A real folder at the default path (or nothing there yet).
    Default,
    /// A link at the default path to `target`; `available` is false when the target is gone.
    Moved { target: PathBuf, available: bool },
}

/// The link at `p`'s target, if `p` is a link (a junction or symlink).
fn link_target(p: &Path) -> Option<PathBuf> {
    let meta = fs::symlink_metadata(p).ok()?;
    if !meta.file_type().is_symlink() {
        return None;
    }
    let t = fs::read_link(p).ok()?;
    // Junctions read back with the NT prefix.
    let s = t.to_string_lossy();
    Some(s.strip_prefix(r"\??\").or_else(|| s.strip_prefix(r"\\?\")).map(PathBuf::from).unwrap_or(t))
}

pub fn location(root: &Path) -> Location {
    match link_target(root) {
        Some(target) => {
            let available = target.is_dir();
            Location::Moved { target, available }
        }
        None => Location::Default,
    }
}

fn make_link(link: &Path, target: &Path) -> Result<()> {
    if let Some(parent) = link.parent() {
        fs::create_dir_all(parent).map_err(|e| err(format!("{}: {e}", parent.display())))?;
    }
    #[cfg(windows)]
    let r = junction::create(target, link);
    #[cfg(not(windows))]
    let r = std::os::unix::fs::symlink(target, link);
    r.map_err(|e| err(format!("linking {} to {}: {e}", link.display(), target.display())))
}

/// Removes the link itself, never what it points to.
fn remove_link(link: &Path) -> Result<()> {
    #[cfg(windows)]
    let r = fs::remove_dir(link);
    #[cfg(not(windows))]
    let r = fs::remove_file(link);
    r.map_err(|e| err(format!("removing the link {}: {e}", link.display())))
}

/// `p` with `suffix` added to its file name.
fn sibling(p: &Path, suffix: &str) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

fn same(a: &Path, b: &Path) -> bool {
    let norm = |p: &Path| {
        let s = std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf()).to_string_lossy().trim_end_matches(['\\', '/']).to_string();
        if cfg!(windows) { s.to_lowercase() } else { s }
    };
    norm(a) == norm(b)
}

fn inside(child: &Path, parent: &Path) -> bool {
    let mut c = std::path::absolute(child).unwrap_or_else(|_| child.to_path_buf());
    loop {
        if same(&c, parent) {
            return true;
        }
        if !c.pop() {
            return false;
        }
    }
}

/// Files and bytes under `dir`, skipping the store's own `lock` and `tmp/`.
fn walk(dir: &Path, top: bool, out: &mut Vec<(PathBuf, u64)>) -> Result<()> {
    for e in fs::read_dir(dir).map_err(|e| err(format!("{}: {e}", dir.display())))? {
        let e = e.map_err(|e| err(format!("{}: {e}", dir.display())))?;
        let name = e.file_name();
        if top && (name == "lock" || name == "tmp") {
            continue;
        }
        let p = e.path();
        let ft = e.file_type().map_err(|e| err(format!("{}: {e}", p.display())))?;
        if ft.is_dir() {
            out.push((p.clone(), u64::MAX));
            walk(&p, false, out)?;
        } else if ft.is_file() {
            let len = e.metadata().map(|m| m.len()).unwrap_or(0);
            out.push((p, len));
        }
    }
    Ok(())
}

/// Copies the store at `from` into `to` (created), reporting (copied, total) bytes. A failed copy
/// (a full drive) removes what it wrote.
fn copy_store(from: &Path, to: &Path, progress: &mut dyn FnMut(u64, u64)) -> Result<()> {
    let mut items = Vec::new();
    walk(from, true, &mut items)?;
    let total: u64 = items.iter().filter(|(_, n)| *n != u64::MAX).map(|(_, n)| n).sum();
    let res = (|| -> Result<()> {
        fs::create_dir_all(to).map_err(|e| err(format!("{}: {e}", to.display())))?;
        let mut done = 0;
        progress(0, total);
        for (p, n) in &items {
            let dest = to.join(p.strip_prefix(from).unwrap());
            if *n == u64::MAX {
                fs::create_dir_all(&dest).map_err(|e| err(format!("{}: {e}", dest.display())))?;
            } else {
                fs::copy(p, &dest).map_err(|e| err(format!("copying {} to {}: {e}", p.display(), dest.display())))?;
                done += n;
                progress(done, total);
            }
        }
        // the same files and bytes on the other side
        let mut copied = Vec::new();
        walk(to, true, &mut copied)?;
        let got: u64 = copied.iter().filter(|(_, n)| *n != u64::MAX).map(|(_, n)| n).sum();
        if copied.len() != items.len() || got != total {
            return Err(err(format!("the copy in {} doesn't match the store ({got} of {total} bytes)", to.display())));
        }
        Ok(())
    })();
    if res.is_err() {
        let _ = fs::remove_dir_all(to);
    }
    res
}

fn in_use(dir: &Path, e: std::io::Error) -> Error {
    err(format!(
        "{} is in use ({e}): close the apps that use it (an upscale or a video playing through it), then try again. Nothing was changed.",
        dir.display()
    ))
}

/// Waits for the store lock at `dir` (if a store is there) and copies it to `to`, holding the lock
/// for the copy so no install or gc changes it meanwhile.
fn copy_locked(dir: &Path, to: &Path, waiting: &mut dyn FnMut(), progress: &mut dyn FnMut(u64, u64)) -> Result<()> {
    let lock = match Store::open_existing(dir) {
        Some(s) => Some(s.lock(waiting)?),
        None => None,
    };
    let r = copy_store(dir, to, progress);
    drop(lock); // released before the rename aside, which an open lock file would block on Windows
    r
}

/// Moves the store at `root` (the default path, a real folder or already a link) to `dest`, which
/// must be new or an empty folder on a local drive, and links `root` to it. `dest` being the default
/// path itself is [`move_back`].
pub fn move_to(root: &Path, dest: &Path, waiting: &mut dyn FnMut(), progress: &mut dyn FnMut(u64, u64)) -> Result<()> {
    if same(dest, root) {
        return move_back(root, waiting, progress);
    }
    if !dest.is_absolute() {
        return Err(err(format!("{}: give a full path", dest.display())));
    }
    if cfg!(windows) && dest.to_string_lossy().starts_with(r"\\") {
        return Err(err(format!("{}: a network folder can't hold the store; use a local drive", dest.display())));
    }
    if dest.exists() && (!dest.is_dir() || fs::read_dir(dest).map(|mut d| d.next().is_some()).unwrap_or(true)) {
        return Err(err(format!("{} isn't empty: pick a new or empty folder", dest.display())));
    }
    let current = match location(root) {
        Location::Moved { target, available: true } => target,
        Location::Moved { target, available: false } => {
            return Err(err(format!("the store's folder {} is missing (is its drive connected?); reset it instead", target.display())));
        }
        Location::Default => root.to_path_buf(),
    };
    if inside(dest, &current) || inside(&current, dest) || inside(dest, root) {
        return Err(err(format!("{} is inside the store, or the store is inside it", dest.display())));
    }

    if !current.exists() {
        // nothing installed yet: just point the default path there
        fs::create_dir_all(dest).map_err(|e| err(format!("{}: {e}", dest.display())))?;
        return make_link(root, dest);
    }

    copy_locked(&current, dest, waiting, progress)?;
    let aside = sibling(&current, ".moving");
    let _ = fs::remove_dir_all(&aside);
    if let Err(e) = fs::rename(&current, &aside) {
        let _ = fs::remove_dir_all(dest);
        return Err(in_use(&current, e));
    }
    let relinked = (|| {
        if current != root {
            remove_link(root)?;
        }
        make_link(root, dest)
    })();
    if let Err(e) = relinked {
        // put it back as it was
        let _ = fs::rename(&aside, &current);
        if current != root && link_target(root).is_none() {
            let _ = make_link(root, &current);
        }
        let _ = fs::remove_dir_all(dest);
        return Err(e);
    }
    let _ = fs::remove_dir_all(&aside);
    Ok(())
}

/// Moves a moved store back to the default path `root`, replacing the link with the real folder.
/// Nothing to do when it's already there.
pub fn move_back(root: &Path, waiting: &mut dyn FnMut(), progress: &mut dyn FnMut(u64, u64)) -> Result<()> {
    let target = match location(root) {
        Location::Default => return Ok(()),
        Location::Moved { target, available: false } => {
            return Err(err(format!("the store's folder {} is missing (is its drive connected?); reset it instead", target.display())));
        }
        Location::Moved { target, .. } => target,
    };
    let staged = sibling(root, ".new");
    let _ = fs::remove_dir_all(&staged);
    copy_locked(&target, &staged, waiting, progress)?;
    let aside = sibling(&target, ".moving");
    let _ = fs::remove_dir_all(&aside);
    if let Err(e) = fs::rename(&target, &aside) {
        let _ = fs::remove_dir_all(&staged);
        return Err(in_use(&target, e));
    }
    let swapped = (|| {
        remove_link(root)?;
        fs::rename(&staged, root).map_err(|e| err(format!("{}: {e}", root.display())))
    })();
    if let Err(e) = swapped {
        let _ = fs::rename(&aside, &target);
        if link_target(root).is_none() && !root.exists() {
            let _ = make_link(root, &target);
        }
        let _ = fs::remove_dir_all(&staged);
        return Err(e);
    }
    let _ = fs::remove_dir_all(&aside);
    Ok(())
}

/// Drops the link at `root` when its target is missing (a drive that's gone for good): the next
/// install starts an empty store at the default path. Refused while the target is there, where
/// [`move_back`] keeps everything instead.
pub fn reset(root: &Path) -> Result<()> {
    match location(root) {
        Location::Default => Ok(()),
        Location::Moved { target, available: true } => Err(err(format!(
            "the store's folder {} is there: move it back instead (aji-runtime move --default), which keeps what's installed",
            target.display()
        ))),
        Location::Moved { .. } => remove_link(root),
    }
}
