# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) and other AI coding agents (Codex, Cursor, etc.) when working with code in this repository.

`AGENTS.md` in the repo root is a symlink to this file (`AGENTS.md -> CLAUDE.md`). All AI coding agents read the same instructions — edit `CLAUDE.md` only; do not replace the symlink with a separate file.

## Repository Language Rule

**Under any circumstances, every piece of content tracked in this repository MUST be written in English. Other languages are strictly prohibited inside the repo, with no exceptions.**

This applies to (non-exhaustive list):

- source code, identifiers, and string literals
- inline comments and docstrings
- Markdown files, READMEs, and generated documentation
- configuration files, YAML, TOML, JSON, and any human-readable values inside them
- example snippets, fixtures, log messages, error messages, CLI help text
- commit subjects, commit bodies, and any text that ends up in `git log`
- file names and directory names
- PR titles, PR descriptions, issue text, and review comments authored on behalf of this repository

The rule is bidirectional and has no exception window:

- **Inside the repo (any tracked artifact, any commit) → English only**, regardless of which language the user is speaking, regardless of which language the user asks you to use, regardless of which language earlier history happens to contain. If the user asks in Chinese / Japanese / etc. for "中文注释" or "日本語のREADME", refuse to write non-English into tracked files; offer the equivalent English version instead.
- **Outside the repo (terminal/chat replies to the user) → match the user's language.** If the user writes in Chinese, reply in Chinese. If the user writes in Japanese, reply in Japanese. Do not force English on the conversation just because the repo is English-only.

If you find yourself about to write non-English text into a tool call that edits, writes, or commits a tracked file, stop and translate to English first. The conversation reply that wraps that tool call can stay in the user's language.

## No Local Absolute Paths

Never commit local absolute filesystem paths into tracked files, fixtures, examples, logs, documentation, commit messages, PR descriptions, or issue text. Anything that would leak a developer's home directory, OS layout, machine name, or local checkout location must be rewritten before it lands in the repo.

When you need a path in tracked content, use one of:

- repository-relative paths (e.g. `sites/subsplease/match.py`, not `/home/<user>/.../sites/subsplease/match.py`)
- environment variables or configuration placeholders (e.g. `$HF_HOME`, `<output_dir>`, `<temp_dir>`)
- generic placeholders such as `/path/to/...` or `~/...` only when an example genuinely needs an abstract path

This rule applies to commit messages and PR bodies as well — sanitize any pasted shell output before committing it.

## Commit Identity Policy

Use whatever git identity is already configured for this repository. Before committing, read it:

```bash
git config user.name
git config user.email
```

Do **not** edit `git config` to change identity, do **not** override the author on a per-commit basis (`--author=...`), and do **not** invent a name or email — just commit as the identity the user has already set up locally. If `user.name` or `user.email` is unset, stop and ask the user to configure them rather than guessing.

Commit message format — first line must be `dev(<author>): <summary>` in English, where `<author>` is a short handle derived from the configured `user.name` (e.g. for `user.name = alice`, use `dev(alice): add subsplease retry guard`). For multi-line messages:

- one blank line after the summary
- one concise paragraph on intent or user-visible outcome
- a flat bullet list of concrete change points
- a `Tests:` section when validation was run, one bullet per command

When committing from the shell, use one `-m` per paragraph (or write the message to a file and use `-F`). Do not embed `\n` in a single `-m` argument.

Keep commit scope narrow — one site adapter, one workflow change, one util fix per commit.

## Working with `gh` (GitHub CLI)

Before running ANY `gh` command (creating PRs/issues, commenting, merging, releasing, etc.), you MUST:

1. Run `git config user.name` and `git config user.email` to identify the current repo's git author.
2. Run `gh auth status` to list every GitHub account `gh` knows about (active and inactive). Identify the account whose login (or associated email) matches the git user from step 1 — call this `<MATCHED_USER>`.
3. Run every `gh` invocation with that account's token injected via env var, NOT by switching the active account. The verified pattern is:

   ```bash
   GH_TOKEN=$(gh auth token --user <MATCHED_USER>) gh <subcommand> ...
   ```

   `GH_TOKEN` takes precedence over `gh`'s stored active account for that single process, so the command runs as `<MATCHED_USER>` regardless of which account is "active". Verify with `GH_TOKEN=$(gh auth token --user <MATCHED_USER>) gh api user --jq .login` — it must print `<MATCHED_USER>`.
4. If no matching account can be found in `gh auth status` (no auth, no account matches the git user, ambiguous mapping), **refuse to run the `gh` command** and report the mismatch to the user. Do NOT guess, do NOT proceed with a non-matching account, and do NOT silently use whatever account `gh` defaults to.

### Forbidden: `gh auth switch`

Do **NOT** use `gh auth switch` to change accounts before running `gh` commands. `gh auth switch` mutates global state in `~/.config/gh/hosts.yml` (the "active account" pointer); when multiple processes/agents run concurrently on this machine, one process's switch silently changes the active account under another process's feet, causing PRs/comments to be created under the wrong identity. Always use the per-process `GH_TOKEN=$(gh auth token --user ...)` pattern instead — it scopes the account choice to one command and cannot race with other processes.

Rationale: developer machines routinely run concurrent agents/automation across several GitHub identities. Running `gh` under the wrong account creates PRs/comments attributed to the wrong person and is hard to undo.

---

## Project Overview

`anime_sites` syncs anime metadata from a set of public anime sites into HuggingFace datasets, matching each entry against MyAnimeList (via the Jikan API) with help from an LLM.

The runtime entrypoints are GitHub Actions workflows scheduled in `.github/workflows/`; each workflow runs a Python module under `sites/` that crawls a source site, enriches it with MAL data, and uploads a parquet table plus rendered `README.md` to the configured HuggingFace dataset repository.

## Repository Layout

- `sites/` — one subpackage per source site, plus shared utilities
  - `sites/subsplease/` — [subsplease.org](https://subsplease.org) shows
    - `lst.py` — list all shows from the index page
    - `info.py` — scrape one show page (title, synopsis, batch / episode lists)
    - `llm.py` — LLM-assisted matching against MAL search results
    - `match.py` — orchestrates the sync and uploads the dataset (entrypoint: `python -m sites.subsplease.match`)
    - `rp.py` — auxiliary report generation
  - `sites/fancaps/` — [fancaps.net](https://fancaps.net) anime entries
    - `data.py` — pulls the prebuilt `bangumi.json` mapping from `deepghs/fancaps_index`
    - `llm.py` — LLM-assisted MAL matching for fancaps titles
    - `match.py` — sync orchestrator and dataset uploader
  - `sites/erairaws/` — [erai-raws.info](https://www.erai-raws.info) anime list (login-cookie protected)
    - `info.py` — page scraper for one anime, including release tables and external links
    - `lst.py` — full sync entrypoint that uploads `animes.parquet`, `items.parquet`, and an images directory
  - `sites/utils/` — shared helpers
    - `session.py` — `get_requests_session()` with retry/timeout adapter, `srequest()` wrapper
    - `download.py` — streaming file download with progress
    - `parallel.py` — `parallel_call(...)` thread pool helper
    - `mal.py` — `get_items_from_myanimelist(title)` via `pyanimeinfo.JikanV4Client`, with 429 backoff
    - `llm.py` — `get_openai_client()` (OpenAI-compatible client, base URL from env)
- `.github/workflows/`
  - `subsplease_mal.yml` — daily cron that runs `python -m sites.subsplease.match`
  - `date.yml` — weekly job that publishes a date stamp to the `date-keep` branch
- `requirements.txt` — runtime dependencies
- `README.md` — short project description

The top-level `test_*.py` files and `test_*.json` / `test_*.html` fixtures are exploratory / scratch scripts, not a unit-test suite.

## Environment Variables

Set these via the workflow `env` block (backed by GitHub Actions secrets) or, locally, via `.env`:

- `HF_TOKEN` — HuggingFace API token, required for dataset uploads
- `LLM_API_KEY`, `LLM_SITE` — credentials and base URL for the OpenAI-compatible LLM endpoint used by the matching code
- `ERAI_RAW_COOKIE` — full `Cookie` header value for `erai-raws.info`; required by `sites.erairaws.info.get_session()` unless `no_login=True`
- `PP_SITE` — optional HTTP/HTTPS proxy URL used as a proxy pool for outbound scraping

Do not commit secrets, tokens, cookies, dataset repository IDs, or local credentials. The only place these belong is GitHub Actions secrets and a local untracked `.env`.

## Common Operations

```bash
# install runtime deps
pip install -r requirements.txt

# run the subsplease ↔ MAL sync end to end
python -m sites.subsplease.match

# run the fancaps sync
python -m sites.fancaps.match

# run the erai-raws sync (requires ERAI_RAW_COOKIE)
python -m sites.erairaws.lst
```

Each `match` / `lst` entrypoint creates the target HuggingFace dataset repo (private) on first run, downloads the existing parquet table to merge new rows incrementally, downloads / re-uses cover images under `assets/`, regenerates the dataset `README.md`, and uploads everything in rate-limited batches.

## Architecture Notes

- **Incremental sync** — every site keeps a `table.parquet` (or `animes.parquet` + `items.parquet`) on the HuggingFace dataset. Each run loads it, skips entries already matched, and only re-deploys when there are updates.
- **Rate limiting** — `pyrate_limiter` controls how often `upload_directory_as_directory` is called (`upload_time_span`); `deploy_span` further debounces uploads inside one run.
- **LLM matching** — for each source entry, the code calls Jikan to fetch MAL search candidates and then asks the LLM to pick the best `mal_id` / `title` / `year` triple based on title similarity, synopsis, status, and year. The LLM client is OpenAI-compatible; `LLM_SITE` lets it point at any OpenAI-API-compatible endpoint.
- **Image handling** — covers from the source site and from MAL are downloaded into `assets/<source>/...` and `assets/mal/...`, then linked into the rendered dataset `README.md` via `hf_hub_url`.
- **Resilience** — `srequest()` retries with exponential-style backoff on `RequestException`; MAL queries retry on HTTP 429; per-record exceptions are logged and skipped rather than aborting a whole sync.

When you change a workflow-backed module, validate it with the same `python -m sites.<site>.<module>` entrypoint that the workflow uses, not just by importing it.
