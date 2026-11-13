//! Path helpers for relative mode and per-copy ids: lexical normalizing, relative paths between two
//! folders, and comparing paths the way Windows does (case-insensitively).

use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

/// `p` with `.` dropped and `..` folded into the folder before it, without touching the disk.
/// A `..` above the root is dropped (as Windows does); a leading `..` of a relative path is kept.
pub fn normalize(p: &Path) -> PathBuf {
    let mut out: Vec<Component> = Vec::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => match out.last() {
                Some(Component::Normal(_)) => {
                    out.pop();
                }
                Some(Component::RootDir) | Some(Component::Prefix(_)) => {}
                _ => out.push(c),
            },
            _ => out.push(c),
        }
    }
    let mut r = PathBuf::new();
    for c in out {
        r.push(c.as_os_str());
    }
    if r.as_os_str().is_empty() {
        r.push(".");
    }
    r
}

/// `p` made absolute (against the current folder) and normalized.
pub fn absolute(p: &Path) -> PathBuf {
    normalize(&std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf()))
}

fn comp_key(c: &Component) -> OsString {
    let s = c.as_os_str();
    if cfg!(windows) { OsString::from(s.to_string_lossy().to_lowercase()) } else { s.to_owned() }
}

/// `target` relative to the folder `base` (`..` components as needed, native separators), or None
/// when there is none (another drive). Both are made absolute first.
pub fn relative_to(base: &Path, target: &Path) -> Option<PathBuf> {
    let (b, t) = (absolute(base), absolute(target));
    let bc: Vec<Component> = b.components().collect();
    let tc: Vec<Component> = t.components().collect();
    // the same drive (prefix) and root
    let anchor = |v: &[Component]| v.iter().take_while(|c| matches!(c, Component::Prefix(_) | Component::RootDir)).count();
    let (ba, ta) = (anchor(&bc), anchor(&tc));
    if ba == 0 || ba != ta || bc[..ba].iter().zip(&tc[..ta]).any(|(x, y)| comp_key(x) != comp_key(y)) {
        return None;
    }
    let mut common = ba;
    while common < bc.len() && common < tc.len() && comp_key(&bc[common]) == comp_key(&tc[common]) {
        common += 1;
    }
    let mut r = PathBuf::new();
    for _ in common..bc.len() {
        r.push("..");
    }
    for c in &tc[common..] {
        r.push(c.as_os_str());
    }
    if r.as_os_str().is_empty() {
        r.push(".");
    }
    Some(r)
}

/// The key a path is compared and hashed by: normalized, `/` separators, no trailing separator,
/// lowercased on Windows. Relative paths stay relative.
pub fn path_key(p: &Path) -> String {
    let s = normalize(p).to_string_lossy().replace('\\', "/");
    let s = if s.len() > 1 { s.trim_end_matches('/').to_string() } else { s };
    if cfg!(windows) { s.to_lowercase() } else { s }
}

/// Whether two paths name the same file (made absolute and normalized; case-insensitively on
/// Windows). Doesn't touch the disk.
pub fn same_path(a: &Path, b: &Path) -> bool {
    path_key(&absolute(a)) == path_key(&absolute(b))
}

/// Whether two folders are the same: the same path, or (when both exist) the same folder once links
/// and junctions are resolved (a moved per-user store is reached through a junction).
pub fn same_dir(a: &Path, b: &Path) -> bool {
    if same_path(a, b) {
        return true;
    }
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => path_key(&x) == path_key(&y),
        _ => false,
    }
}

/// Whether `child` is `parent` or inside it (by path, and when both exist, with links resolved).
pub fn inside(child: &Path, parent: &Path) -> bool {
    let lexical = |c: &Path, p: &Path| {
        let (c, p) = (path_key(&absolute(c)), path_key(&absolute(p)));
        c == p || c.starts_with(&format!("{}/", p.trim_end_matches('/')))
    };
    if lexical(child, parent) {
        return true;
    }
    // resolve links on the longest existing part of `child`
    let canon = |p: &Path| -> Option<PathBuf> {
        let abs = absolute(p);
        let mut existing = abs.clone();
        let mut rest = Vec::new();
        loop {
            if let Ok(c) = std::fs::canonicalize(&existing) {
                let mut c = c;
                for r in rest.iter().rev() {
                    c.push(r);
                }
                return Some(c);
            }
            rest.push(existing.file_name()?.to_owned());
            if !existing.pop() {
                return None;
            }
        }
    };
    match (canon(child), canon(parent)) {
        (Some(c), Some(p)) => lexical(&c, &p),
        _ => false,
    }
}

/// Whether `p` is a network path (`\\server\share\...`) on Windows.
pub fn is_network(p: &Path) -> bool {
    #[cfg(windows)]
    {
        use std::path::Prefix;
        match absolute(p).components().next() {
            Some(Component::Prefix(x)) => matches!(x.kind(), Prefix::UNC(..) | Prefix::VerbatimUNC(..)),
            _ => false,
        }
    }
    #[cfg(not(windows))]
    {
        let _ = p;
        false
    }
}

/// Whether the drive `p` is on is there (Windows: `E:\` of an unplugged USB drive isn't). Always
/// true elsewhere.
pub fn drive_exists(p: &Path) -> bool {
    let abs = absolute(p);
    let anchor: PathBuf =
        abs.components().take_while(|c| matches!(c, Component::Prefix(_) | Component::RootDir)).map(|c| c.as_os_str()).collect();
    anchor.as_os_str().is_empty() || anchor.is_dir()
}
