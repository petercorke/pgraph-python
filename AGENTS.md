# pgraph-python — Agent Instructions

Part of the RVC ecosystem. **Read [rvc-ecosystem/AGENTS.md](https://github.com/petercorke/rvc-ecosystem/blob/main/AGENTS.md) first** — it defines shared conventions: repo ownership, math invariants, dependency boundaries, git/PR workflow, code standards, tech-debt tracking. This file only adds what's specific to this repo.

| | |
|---|---|
| PyPI package | `pgraph-python` |
| Nickname | pgraph-python |
| Owner | Peter Corke (`petercorke`) |
| Default branch | `main` |
| Contribution model | Branch → PR; direct push to `main` at Peter's discretion |

## Notes specific to this repo

- Minimal graph/path-planning utility package — no internal ecosystem dependencies.
- Used by `robotics-toolbox-python` (path/roadmap planning) and `machinevision-toolbox-python`.
- Pure Python — builds with Hatch (`hatchling`) directly.
