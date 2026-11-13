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

### Which store a copy uses

Each app ships `aji-runtime(.exe)` beside itself, and every command uses the first of:

1. `--store DIR`;
2. `AJI_RUNTIME_STORE` (for tests);
3. the settings file `aji-runtime.conf` beside the `aji-runtime` executable (`AJI_RUNTIME_CONF`
   names another file);
4. the per-user store above.

The settings file is plain text that a user can edit:

```
# comment lines start with # or ;
store = runtime
```

One setting, `store`: a folder, absolute or relative to the settings file's folder. The value is
taken as typed (trimmed, one pair of surrounding quotes removed, no escapes); empty or no file means
the per-user store. Anything else is an error that names the file and the line. `aji-runtime use`
writes it.

A **relative** `store` (e.g. `runtime`, or `..\shared-runtime`) puts the store in relative mode:
refs keep the app's path relative to the store, and `aji-runtime.json` names the pack folders
relative to its own folder. The app's folder, with its store inside or beside it, can then be moved
or copied to another PC and keeps working with no fix-up. A portable zip ships with
`store = runtime` and keeps nothing in AppData; several apps can share a store by naming the same
folder. Whatever store an app uses, `ensure` copies a pack the per-user store already has (hard
links where possible) instead of downloading it.

aji-runtime 0.2.1 and older read a relative `app_path` in a ref against the current folder, so they
see such refs as dead. They can't be pointed at a store by a settings file, so this only matters
when one is run with `--store` on a store written in relative mode.

### Moving the per-user store

`move` puts the per-user store on another drive: the folder above becomes a link to the new one (a
directory junction on Windows, which needs no admin rights or Developer Mode; a symlink on Linux).
Apps keep using the default path and the pack paths in their `aji-runtime.json`, so every app,
old versions included, keeps working without knowing about the move. A copy whose store is set in
its settings file changes it with `use` instead.

## Program

```
aji-runtime ensure --app-id ID [--per-copy] --app-path EXE --index URL|FILE --index-sha256 HEX
                   [--want trt,trtrtx,PACK...] [--gpus auto|none|smNN,...] [--aji-dir DIR]
                   [--app-version V] [--adopt-from DIR]... [--mirror DIR] [--check]
                   [--progress lines] [--store DIR]
aji-runtime adopt  --from DIR (and ensure's options)
aji-runtime unref  --app-id ID [--per-copy --app-path EXE]
aji-runtime gc
aji-runtime list
aji-runtime path   NAME[/VERSION] [--app-id ID [--per-copy --app-path EXE]]
aji-runtime show   --app-id ID [--per-copy --app-path EXE]
aji-runtime location [--json]
aji-runtime move   FOLDER|--default [--progress lines]
aji-runtime reset
aji-runtime use    FOLDER|--shared [--app-id ID [--per-copy] --app-path EXE [--aji-dir DIR]]
                   [--progress lines]
```

Every command takes `--store DIR`.

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
  - A pack is installed when its folder's `.pack.json` lists the index's files (paths, sizes,
    SHA-256s), whatever archive it came in, so a newer runtime release with the same files
    downloads nothing. A pack folder is never replaced: an index with other files under an
    installed name and version fails.
  - Downloads give up connecting after 30 s, and fail when no data arrives for 30 s (a slow
    download that keeps coming is fine). Up to 4 retries in a row (after 2, 5, 15 and 30 s), each
    resuming where the last stopped (HTTP Range), then the whole file's SHA-256 is checked as
    always. A retry that got further starts the count over, up to 20 attempts in all.
  - Only one app changes the store at a time. One that finds it busy prints `AJI_WAIT` once and
    waits up to an hour (`unref`: 5 minutes, since its uninstaller waits too; the ref dies with the
    app anyway), then fails with nothing changed. The lock is never broken.
  - In a store other than the per-user one, a pack the per-user store has complete with the
    index's files is copied in instead of downloaded (each file hard-linked, or copied across
    drives; staged in `tmp\` and renamed in, checked against its `.pack.json`) and reported as
    `adopted`. The per-user store is only read: never locked or changed. Any mismatch or error
    falls back to the download.
  - **`--per-copy`**, for apps that can have several copies on one PC (a portable zip beside an
    installed copy): the ref is `ID-<8 hex>`, from the SHA-256 of the copy's app path as the ref
    stores it (relative to the store in relative mode, so it survives moving the folder). A plain
    `ID` ref naming the same app path, from before the app used `--per-copy`, is removed once the
    per-copy ref is written; one naming another path is left alone. `adopt`, `unref`, `show`,
    `path` and `use` take `--per-copy` too (with `--app-path`).
- **`adopt`** first takes matching files from an app's old private install. They're found by
  SHA-256 and hard-linked, or copied across drives. The old install keeps working until the app
  deletes its copies.
- **`ensure --check`** prints the plan and changes nothing: each pack is installed, adopted (its
  old files are there with the pinned SHA-256s, or the per-user store has it) or download, and the
  total is what would be
  downloaded. It ends with `AJI_DONE` (complete) only when nothing would be installed and the
  app's ref lists exactly these packs for this `--app-path` (and, with `--aji-dir`,
  `aji-runtime.json` is what an ensure writes, paths in the form this store's mode writes them; with
  `--per-copy`, no plain ref for this path is left). A ref for another path (a moved or renamed
  AppImage) is dead once that path is gone, and the next gc frees its packs, so an updater that
  ensures on an incomplete check writes it again. It takes no lock, so it never waits for another
  app's download: every change to the store is a rename, so it sees each pack and ref whole.
- **`show`** prints the app's packs as JSON (name, version, kind, folder, installed), and the
  ref's `app_version` and `app_path` (null without a ref), for its updater or launcher; paths are
  absolute. `ref_id` is the ref it read (with `--per-copy`, the plain `ID` ref while that still
  names this copy's path). It changes nothing.
- **`unref`**: uninstallers call it to drop the app's ref; it then frees what no app uses. With
  `--per-copy`, it also drops a plain `ID` ref naming this copy's path.
- **`location`** says where the store is: `default`, `moved` with its folder and whether it's
  reachable (a removable drive may be unplugged), or `settings <root> (from <file>)`. `--json`:
  `{"root", "moved", "target", "available", "source", "setting", "relative", "settings_file",
  "default_root"}`: `source` is `default`, `settings`, `env` or `flag`; `setting` the settings
  file's raw `store` value (or null); `relative` whether it's in relative mode; `default_root` the
  per-user store. A store other than the per-user one is unavailable only when neither it nor its
  drive is there; one not created yet on a drive that is will be by the first install. It changes
  nothing.
- **`move FOLDER`** moves the store to a new or empty folder on a local drive and links the default
  path to it; **`move --default`** moves it back. A move copies under the store lock (`--progress
  lines` adds `AJI_MOVE <copied> <total>`), then renames the old folder aside before switching the
  link. On Windows that rename fails while an app has a file loaded from the store, which stops the
  move with nothing changed. Only then is the old copy deleted.
- **`reset`** drops the link when the store's folder is gone for good. The next install starts an
  empty store at the default path; the old folder is never touched. It's refused while the folder
  is there, since `move --default` keeps what's installed. Until a move back or reset, a store whose
  folder is missing fails to open, so no app quietly re-downloads everything to the default drive.
  `move` and `reset` are refused for a store set in the settings file: that copy changes it with
  `use`.
- **`use FOLDER`** / **`use --shared`** switches the store this copy uses and takes the app's packs
  along without downloading. `FOLDER` is relative to the settings file's folder (relative mode) or
  absolute; `--shared` (or the per-user store's own path) is the per-user store. It's refused for
  a network folder, a folder that has other files and isn't a store, and a folder inside the
  current store or holding it. With `--app-id` and a ref in the current store, it copies the packs
  the new store lacks (hard links where possible; `AJI_MOVE` progress), writes the ref there, with
  `--aji-dir` rewrites `aji-runtime.json`, writes the settings file, and then drops the ref in the
  old store and frees what no app uses. An old store other than the per-user one that's left empty
  is removed: only its own files, and its `lock` file (with the folder) only when no other process
  has it open, which is checked on Windows; on Linux the `lock` file and the folder stay. Without a ref, only the settings file is written; for the same
  store in another form (absolute to relative), the ref, `aji-runtime.json` and the settings file
  are rewritten. A failure leaves the copy working on one store or the other.

A **portable copy** (a zip) ships `aji-runtime.conf` with `store = runtime`: its packs, ref and
`aji-runtime.json` live inside its own folder, nothing goes to AppData, and the folder can be moved
or copied to another PC. It writes its ref with `--per-copy`, so it never touches an installed copy's
ref. `use --shared` moves it to the per-user store, and `use FOLDER` to a folder of the user's
choice that other apps can point at too.

## Library

`aji_runtime::{ensure, unref, adopt, app_packs, Index, Store}`. MangaJaNaiCLI's `setup` uses it.
For several copies of one app: `ensure_per_copy`, `unref_per_copy`, `copy_ref_id`, `find_ref`.
Relative mode and taking packs from the per-user store are set on the store
(`Store::with_relative`, `Store::with_shared`); `settings::resolve` opens the store the program
would, and `switch::use_store` is `use`.

## Build

```
cargo build --release
cargo test
cargo clippy --all-targets
```

Releases: push a `v*` tag equal to `Cargo.toml`'s version. `.github/workflows/release.yml` builds
the Windows and Linux binaries and publishes them.
