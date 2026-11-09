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

`move` puts the store on another drive: the folder above becomes a link to the new one (a
directory junction on Windows, which needs no admin rights or Developer Mode; a symlink on Linux).
Apps keep using the default path and the pack paths in their `aji-runtime.json`, so every app,
old versions included, keeps working without knowing about the move.

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
aji-runtime show   --app-id ID
aji-runtime location [--json]
aji-runtime move   FOLDER|--default [--progress lines]
aji-runtime reset
```

- **`ensure`** installs what the app pins and writes its ref with exactly those packs. With
  `--aji-dir`, it also writes `aji-runtime.json` beside the app's libaji, which then loads
  TensorRT from the store. Its `models` key lists the app's model pack folders, for the app
  itself; libaji ignores it. Last, it deletes packs no app uses any more.
  - The index is the shared runtime release's `packs-<platform>.json` (animejanai-inference
    `scripts/runtime_packs.py`, tag `runtime-<n>`), pinned by SHA-256.
  - `--want trt` takes the TensorRT runtime plus a builder pack for each GPU generation found
    (falling back to PTX). GPUs below compute capability 7.5 (older than the RTX 20 / GTX 16
    series) are skipped, since TensorRT doesn't run on them; with only such GPUs it fails. `trtrtx` takes the TensorRT-RTX runtime; any other value is a pack
    name.
- **`adopt`** first takes matching files from an app's old private install. They're found by
  SHA-256 and hard-linked, or copied across drives. The old install keeps working until the app
  deletes its copies.
- **`ensure --check`** prints the plan and changes nothing: each pack is installed, adopted (its
  old files are there with the pinned SHA-256s) or download, and the total is what would be
  downloaded.
- **`show`** prints the app's packs as JSON (name, version, kind, folder, installed), for its
  updater. It changes nothing.
- **`unref`**: uninstallers call it to drop the app's ref; it then frees what no app uses.
- **`location`** says where the store is: `default`, or `moved` with its folder and whether it's
  reachable (a removable drive may be unplugged). `--json`: `{"root", "moved", "target",
  "available"}`. It changes nothing.
- **`move FOLDER`** moves the store to a new or empty folder on a local drive and links the default
  path to it; **`move --default`** moves it back. A move copies under the store lock (`--progress
  lines` adds `AJI_MOVE <copied> <total>`), then renames the old folder aside before switching the
  link. On Windows that rename fails while an app has a file loaded from the store, which stops the
  move with nothing changed. Only then is the old copy deleted.
- **`reset`** drops the link when the store's folder is gone for good. The next install starts an
  empty store at the default path; the old folder is never touched. It's refused while the folder
  is there, since `move --default` keeps what's installed. Until a move back or reset, a store whose
  folder is missing fails to open, so no app quietly re-downloads everything to the default drive.

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
