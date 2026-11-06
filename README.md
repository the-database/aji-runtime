# aji-runtime

The per-user store of runtime and model packs that libaji apps share (MangaJaNaiCLI, yosh's
upscaling add-on; later MangaJaNai GUI, VideoJaNai and mpv-AnimeJaNai). Each NVIDIA TensorRT
runtime, per-GPU builder pack, TensorRT-RTX runtime and model set is downloaded and stored
**once per user**, and every installed app records the packs it uses. A pack is deleted when no
installed app uses it any more.

Design: `plans/shared-runtime-packs.md` in the-database/animejan.ai.

## Store

```
%LOCALAPPDATA%\the-database\aji\          (Linux: $XDG_DATA_HOME/the-database/aji/)
  store.json                               {"format": 1}
  lock                                     held while anything changes the store
  tmp\                                     downloads and deletions in progress
  packs\<name>\<version>\                  one folder per exact pack version, never changed in place
      .pack.json                           written last: a pack is complete only with it
  refs\<app-id>.json                       one per installed app: the packs it uses
```

`AJI_RUNTIME_STORE` (or `--store`) points elsewhere, for tests.

## Program

```
aji-runtime ensure --app-id ID --app-path EXE --index URL|FILE --index-sha256 HEX
                   [--want trt,trtrtx,PACK...] [--gpus auto|none|smNN,...] [--aji-dir DIR]
                   [--app-version V] [--adopt-from DIR]... [--mirror DIR] [--check]
                   [--progress lines] [--store DIR]
aji-runtime adopt  --from DIR (and ensure's options)
aji-runtime unref  --app-id ID
aji-runtime gc
aji-runtime list
aji-runtime path   NAME[/VERSION] [--app-id ID]
```

- **`ensure`** installs what the app pins and writes its ref with exactly those packs. With
  `--aji-dir`, it also writes `aji-runtime.json` beside the app's libaji, which then loads
  TensorRT from the store. Its `models` key lists the app's model pack folders, for the app
  itself; libaji ignores it. Last, it deletes packs no app uses any more.
  - The index is the shared runtime release's `packs-<platform>.json` (animejanai-inference
    `scripts/runtime_packs.py`, tag `runtime-<n>`), pinned by SHA-256.
  - `--want trt` takes the TensorRT runtime plus a builder pack for each GPU generation found
    (falling back to PTX). `trtrtx` takes the TensorRT-RTX runtime; any other value is a pack
    name.
- **`adopt`** first takes matching files from an app's old private install. They're found by
  SHA-256 and hard-linked, or copied across drives. The old install keeps working until the app
  deletes its copies.
- **`unref`**: uninstallers call it to drop the app's ref; it then frees what no app uses.

Portable apps never write a ref and keep private packs.

## Library

`aji_runtime::{ensure, unref, adopt, app_packs, Index, Store}`. MangaJaNaiCLI's `setup` uses it.

## Build

```
cargo build --release
cargo test
cargo clippy --all-targets
```

Releases: push a `v*` tag equal to `Cargo.toml`'s version. `.github/workflows/release.yml` builds
the Windows and Linux binaries and publishes them.
