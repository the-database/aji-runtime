//! The runtime release's index, `packs-<platform>.json` (animejanai-inference
//! `scripts/runtime_packs.py`), which an app pins by SHA-256, and choosing packs from it.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::nvml::Gpu;
use crate::store::{PackFile, check_component};
use crate::{Result, err, sha256_bytes};

/// The index format this version reads.
pub const INDEX_FORMAT: u32 = 1;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Index {
    pub format: u32,
    pub tag: String,
    pub platform: String,
    pub packs: Vec<IndexPack>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct IndexPack {
    pub name: String,
    /// The content version: the store folder is packs/<name>/<version>.
    pub version: String,
    /// trt | trt-builder | trtrtx | models | private (not for the store: apps keep it themselves)
    pub kind: String,
    /// trt-builder packs: the GPU generation ("sm120") or "ptx", the fallback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sm: Option<String>,
    pub asset: String,
    pub url: String,
    pub sha256: String,
    pub bytes: u64,
    #[serde(default)]
    pub installed_bytes: u64,
    pub files: Vec<PackFile>,
}

impl IndexPack {
    pub fn key(&self) -> String {
        crate::store::pack_key(&self.name, &self.version)
    }
}

/// The current platform's name in the index ("windows-x64" / "linux-x64").
pub fn platform() -> &'static str {
    if cfg!(windows) { "windows-x64" } else { "linux-x64" }
}

impl Index {
    /// Parses an index and checks it against the SHA-256 the app pins (`None` only for tests).
    pub fn parse(bytes: &[u8], pin: Option<&str>) -> Result<Index> {
        if let Some(pin) = pin {
            let got = sha256_bytes(bytes);
            if !got.eq_ignore_ascii_case(pin) {
                return Err(err(format!("the runtime index doesn't match its pin (SHA-256 {got}, pinned {pin})")));
            }
        }
        let index: Index = serde_json::from_slice(bytes).map_err(|e| err(format!("runtime index: {e}")))?;
        if index.format != INDEX_FORMAT {
            return Err(err(format!("runtime index format {} (this program reads {INDEX_FORMAT})", index.format)));
        }
        for p in &index.packs {
            check_component("pack name", &p.name)?;
            check_component("pack version", &p.version)?;
            for f in &p.files {
                check_rel_path(&f.path)?;
            }
        }
        Ok(index)
    }

    /// Reads an index from a URL (http/https) or a local file, checked against its pin.
    pub fn load(source: &str, pin: Option<&str>) -> Result<Index> {
        let bytes = if source.starts_with("http://") || source.starts_with("https://") {
            crate::fetch::fetch_small(source)?
        } else {
            let p = source.strip_prefix("file://").unwrap_or(source);
            std::fs::read(Path::new(p)).map_err(|e| err(format!("{p}: {e}")))?
        };
        Index::parse(&bytes, pin)
    }

    pub fn pack(&self, name: &str) -> Option<&IndexPack> {
        self.packs.iter().find(|p| p.name == name)
    }

    /// The store packs for `wants`, in install order, without duplicates:
    /// - `trt`: the TensorRT runtime and the builder resources for each GPU generation in `gpus`
    ///   (its own pack, else the PTX fallback). GPUs below [`crate::nvml::TRT_MIN_COMPUTE`] are
    ///   left out, and an error if no other GPU is there;
    /// - `trtrtx`: the TensorRT-RTX runtime;
    /// - anything else: the pack of that name (models-..., or a builder pack by name).
    pub fn select(&self, wants: &[String], gpus: &[Gpu]) -> Result<Vec<IndexPack>> {
        let mut out: Vec<IndexPack> = Vec::new();
        let add = |p: &IndexPack, out: &mut Vec<IndexPack>| -> Result<()> {
            if p.kind == "private" {
                return Err(err(format!("{} is not a store pack", p.name)));
            }
            if !out.iter().any(|q| q.name == p.name) {
                out.push(p.clone());
            }
            Ok(())
        };
        let need = |kind: &str| -> Result<&IndexPack> {
            self.packs.iter().find(|p| p.kind == kind).ok_or_else(|| err(format!("the runtime index has no {kind} pack")))
        };
        for w in wants {
            match w.as_str() {
                "trt" => {
                    add(need("trt")?, &mut out)?;
                    if gpus.is_empty() {
                        return Err(err("no NVIDIA GPU found for the TensorRT builder packs"));
                    }
                    // GPUs older than TensorRT supports get no builder pack (not the PTX one,
                    // which only compiles forward); with no other GPU there's nothing to install.
                    if !gpus.iter().any(Gpu::trt_capable) {
                        return Err(err(crate::nvml::too_old_message(gpus)));
                    }
                    for g in gpus.iter().filter(|g| g.trt_capable()) {
                        let sm = g.sm();
                        let builders = || self.packs.iter().filter(|p| p.kind == "trt-builder");
                        let b = builders()
                            .find(|p| p.sm.as_deref() == Some(sm.as_str()))
                            .or_else(|| builders().find(|p| p.sm.as_deref() == Some("ptx")))
                            .ok_or_else(|| err(format!("the runtime index has no builder pack for {sm} and no PTX fallback")))?;
                        add(b, &mut out)?;
                    }
                }
                "trtrtx" => add(need("trtrtx")?, &mut out)?,
                name => add(self.pack(name).ok_or_else(|| err(format!("the runtime index has no pack {name}")))?, &mut out)?,
            }
        }
        Ok(out)
    }
}

/// A pack file's path: relative, no `..`, no drive or root.
pub fn check_rel_path(p: &str) -> Result<()> {
    let bad = p.is_empty()
        || p.starts_with('/')
        || p.starts_with('\\')
        || p.contains(':')
        || p.split(['/', '\\']).any(|c| c.is_empty() || c == "." || c == "..");
    if bad { Err(err(format!("invalid file path in a pack: {p:?}"))) } else { Ok(()) }
}
