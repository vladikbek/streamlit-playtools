# AGENTS.md

## Defaults

- All code, code comments, and repo docs (AGENTS.md, README.md) must be in English.
- Default implementation language is Python unless the repo clearly uses something else.
- Always work inside a virtual environment (prefer `.venv/`).

## Quick commands (Python)

- Create venv: `python -m venv .venv`
- Activate (macOS/Linux): `source .venv/bin/activate`
- Activate (Windows): `.venv\Scripts\activate`
- Install deps: `python -m pip install -U pip` then `python -m pip install -r requirements.txt`
- Run: follow `README.md` (fallback: `python -m <module>` or `python main.py`)
- Validate:
  - Prefer the repo's own scripts (Makefile, hatch, uv, npm, etc.) if present.
  - Otherwise run: `python -m compileall .`

## Coding rules

- Minimalism first: prefer stdlib; add a dependency only if it reduces total complexity.
- Keep structure simple: few files, shallow nesting, small functions.
- Add short comments only where intent is not obvious.
- Keep one consistent style across the codebase.

## Project hygiene

- Keep `README.md` updated with setup, run, and config steps (markdownlint clean).
- Maintain `requirements.txt` (pin exact versions; use latest stable when adding).
- Maintain `.gitignore` (ignore `.venv/`, `.env`, caches, build artifacts).
- Put runtime configuration in `.env`. Never commit `.env`. Update `.env.example`.

## Boundaries

- Do: follow existing project structure and scripts.
- Ask: before big refactors, deleting files, or adding major new dependencies.
- Never: commit secrets, introduce "enterprise" abstractions, or add security theater.

## Big tasks (optional)

- If the task touches many files or has unknowns, write a short `PLAN.md` (bullets) and keep it updated while implementing.

## Definition of done

- [ ] Code runs inside venv
- [ ] `README.md` matches current behavior
- [ ] `requirements.txt` and `.gitignore` updated
- [ ] `.env.example` updated when config changes
