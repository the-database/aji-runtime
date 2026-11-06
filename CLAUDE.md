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
- `lib.rs`: `ensure` / `adopt` / `install` / `unref`, and `aji-runtime.json` (read by libaji's
  `core/runtime_packs`).

## Rules that must not break

**A gc bug deletes runtimes another app needs and breaks an app the user never touched.** Every
store rule has a test in `src/tests.rs`. Add one for any change.

- A pack is deleted only when no live ref lists it. A ref is live if it parses and its
  `app_path` exists.
- Everything that changes the store holds the lock. The functions take the `StoreLock` as a
  parameter, so they can't be called without it.
- Pack folders are immutable, and complete only with `.pack.json`, which is written last. New
  packs are staged in `tmp/` and renamed in. Deletions are renamed out to `tmp/` first, and a
  Windows pack with a file in use is skipped whole.
- A store with a newer `store.json` format is added to but never deleted from.
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
