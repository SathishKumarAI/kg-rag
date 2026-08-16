# kg-rag — STATUS

Update this when you STOP working, not when you start.

- **Last touched:** 2026-08-16 (read-only survey; no code changed)
- **Where I stopped:** Not being worked on. This is a **fork of someone else's book
  code**, essentially unmodified.
- **Next action:** none. Decide whether to keep it at all — see below.
- **Blocked on:** nothing. It is dormant by choice, not by obstacle.

## What this actually is

A fork of [`tomasonjo/kg-rag`](https://github.com/tomasonjo/kg-rag) — the companion code
for Manning's *Essential GraphRAG*. Both remotes are configured:

```
origin    https://github.com/SathishKumarAI/kg-rag.git
upstream  https://github.com/tomasonjo/kg-rag.git
```

Contents are the book's: chapter notebooks `ch02`–`ch08` plus helper modules
(`text2cypher.py`, `schema_utils.py`, `utils.py`, `ch0{5,7,8}_tools.py`),
`data/benchmark_data.csv`, and an `.env.example`. The README is the book's marketing
blurb. There are **no tests, no CI, no project docs of your own.**

Your only commit on top of upstream is `e5e9433 "local_test in codebase"`, touching
`.env.example`, `notebooks/ch02.ipynb` and `notebooks/utils.py`. The tree is clean.

## Why it is not part of the RAG program

The three sibling apps (`ai-due-diligence-copilot`, `healthcare-knowledge-navigator`,
`engineering-intelligence-hub`) share one engine, run **local-first with no API keys**,
and are covered by an offline test suite. This repo is the opposite on every axis:

- **Different architecture** — knowledge graph / Cypher, not vector retrieval
- **Requires an OpenAI key** (`openai==1.10.0`) and a running **Neo4j** instance
- Notebook-shaped, so nothing here is importable, testable, or deployable as a service

The program design doc lists it as explicitly out of scope, and nothing in this session
changed that.

## If you come back to it

Pick one deliberately, because "a fork sitting in `shelf/`" is the state that costs the
most to re-read later:

1. **Learning only** — move it to `~/coding/learn/`, where reference material lives. It
   is book code you are reading, not a project you are building.
2. **Build on it** — then it needs your own README explaining what *you* are adding,
   plus the Neo4j and OpenAI setup written down. Expect real cost: it cannot run
   keyless, so it will not inherit the local-first posture the other three have.
3. **Archive it** — `~/coding/archive/`. Upstream is public; re-cloning is free, and a
   fork with one cosmetic commit carries no work worth preserving.

Do not leave it in `shelf/` unlabelled. `shelf/` means paused-but-alive, and nothing
here is paused mid-thought.
