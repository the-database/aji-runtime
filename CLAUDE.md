# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The shared runtime pack store for libaji apps: a library (`src/lib.rs`, used by MangaJaNaiCLI's
`setup`) and the `aji-runtime` program (`src/main.rs`) for the other apps' installers. The
README has the layout and the commands; the design is `plans/shared-runtime-packs.md` in
the-database/animejan.ai.

- `store.rs`: the store layout, locking, refs, gc, `.pack.json`.
- `index.rs`: the runtime release's index (`packs-<platform>.json`, built by
  animejanai-inference `scripts/runtime_packs.py`) and pack selection.
- `fetch.rs`: downloads and extraction.
- `nvml.rs`: NVIDIA GPU detection (copied from MangaJaNaiCLI).
- `lib.rs`: `ensure` / `adopt` / `install` / `unref` (and their `--per-copy` forms), copying packs
  between stores, and `aji-runtime.json` (read by libaji's `core/runtime_packs`).
- `settings.rs`: which store a command uses (`--store` > `AJI_RUNTIME_STORE` > `aji-runtime.conf`
  beside the executable > the per-user store), the settings file's parser and writer.
- `switch.rs`: `use`, switching the store a copy uses. `location.rs`: `move` / `reset` of the
  per-user store. `paths.rs`: relative paths and path comparison.

## Rules that must not break

**A gc bug deletes runtimes another app needs and breaks an app the user never touched.** Every
store rule has a test in `src/tests.rs`. Add one for any change.

- A pack is deleted only when no live ref lists it. A ref is live if it parses and its
  `app_path` exists. A relative `app_path` is resolved against the store's root, in every store
  however it was opened (never against the current folder): a store in relative mode moved with
  its app's folder keeps its packs, and one whose exe is gone frees them.
- Relative mode (the store came from the settings file as a relative path) writes the ref's
  `app_path` relative to the store root and `aji-runtime.json`'s pack folders relative to its own
  folder (absolute when there's no relative path, another drive). Outside it, everything is
  absolute as before. `ensure --check` compares against the form this mode writes.
- Per-copy refs (`ID-<8 hex>` of the app path in the ref's form): a plain `ID` ref is taken over
  or dropped only when it names the same app path. A plain ref naming another path belongs to
  another copy and is never touched.
- A pack copied from another store (the per-user one in `ensure`, the old one in `use`) is staged in
  `tmp/` and renamed in like any install, checked against its `.pack.json`. The source store is
  never locked and never changed; in `ensure` any error falls back to downloading.
- `use` holds one store's lock at a time (the new store's, released before the old one's), so
  opposite runs can't deadlock, and orders its steps so a failure leaves the copy working on one
  store or the other. Removing an emptied store deletes only the store's own files (never an
  app's `engines/` or anything else), and its lock file only on Windows.
- `move` and `reset` are for the per-user store; a store set in the settings file is changed with
  `use`.
- Everything that changes the store holds the lock. The functions take the `StoreLock` as a
  parameter, so they can't be called without it.
- Pack folders are immutable, and complete only with `.pack.json`, which is written last. New
  packs are staged in `tmp/` and renamed in. Deletions are renamed out to `tmp/` first, and a
  Windows pack with a file in use is skipped whole.
- A complete pack is installed once and only gc deletes it. It counts as installed when its
  `.pack.json` lists the index's files (paths, sizes, SHA-256s), whatever archive SHA-256 the index
  pins: a later runtime release may repack the same files. An index with other files under an
  installed name/version is an error; `commit_pack` never replaces a complete pack.
- A store with a newer `store.json` format is added to but never deleted from.
- Every change to the store is a rename (packs in and out whole, refs and `aji-runtime.json` by
  `write_atomic`): that's what lets `ensure --check` and `show` read without the lock. Keep it so.
- `ensure --check` is complete only if an ensure would change nothing, the app's ref included
  (exactly its packs, its `app_path`): updaters and launchers ensure again only when it isn't.
- Downloads hold the lock, so they can't hang: connect and stall timeouts and retries in
  `fetch.rs`. Waiting for the lock gives up after `LOCK_WAIT` (`unref`: `UNREF_LOCK_WAIT`), never
  by breaking it.
- The `--progress lines` protocol and the `aji-runtime.json` format are contracts with apps and
  with libaji. Extend them; don't change existing lines or keys.

## Commands

```sh
cargo build --release
cargo test               # some tests are Windows-only (loaded DLLs)
cargo clippy --all-targets
```

## Conventions

- Commit author is `the-database <25811902+the-database@users.noreply.github.com>`, with a short
  imperative subject and no Co-Authored-By trailer.
- Be sparing with CI runs; the release workflow runs on `v*` tags only.
- A release ends with `notify-downstream.yml`, which asks VideoJaNai, MangaJaNai GUI,
  mpv-AnimeJaNai and MangaJaNaiCLI to open bump PRs (needs the `JANAI_BUMP_TOKEN` secret).
