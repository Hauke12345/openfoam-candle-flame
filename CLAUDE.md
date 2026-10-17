# Instructions for Claude

This repo holds a 2D axisymmetric OpenFOAM candle flame (see README.md). The active project is the
**quasi-3D (2.5D) reacting solver** described in `docs/2p5d_plan.md`. Read that file first.

## How to work
- Work **autonomously** through the stages in `docs/2p5d_plan.md`, in order. The owner doesn't want to
  micromanage: don't ask for approval between stages, and make reasonable engineering choices yourself.
- Ask the owner only for an irreversible or account-level action (changing `main` history, deleting
  data, new paid services), or if a stage is truly blocked.
- Branch: `2p5d-prototype`. Commit and push after every completed stage, and at least every couple of hours.
  Never force-push `main`.
- Keep `quasi3d/STATUS.md` current: stage, what passed (with numbers), what's next, open problems.
  A fresh session must be able to resume from it alone.

## Delegation (automatic; keep the main thread lean)
- `solver-dev` (Sonnet): writing and refactoring code, setting up runs, plots, renders, docs.
- `numerics-expert` (Opus): divergence, NaNs, oscillations, a validation check that has failed twice,
  or design of discretisation, axis treatment and boundary conditions.
- `validator` (Opus, read-only): independent check of each stage against its acceptance criteria
  before it is marked done.
- Give subagents precise, self-contained tasks (file paths, the exact check, the expected output).
  Don't paste large logs. Point to files instead.

## Budget discipline
- Develop on coarse grids and short simulated times. Scale up only after a stage passes on coarse settings.
- Start long runs in the background and check them rarely (≥ 10 min apart). Never print large logs.
- If a stage has failed 3 times on the same issue after `numerics-expert` has looked at it, write up the
  findings in STATUS.md, commit, and move to the next stage that doesn't depend on it, or stop.

## Environment notes (learned the hard way)
- `conda env create -f environment.yml` (conda-forge). Keep Cantera pinned to 3.2: installing ffmpeg
  separately once silently downgraded it to a broken 3.0.1.
- Source `etc/env.sh` before running OpenFOAM. It links `lib/sys-mpich`; without it parallel runs fail with
  "dummy Pstream".
- Blender: `bpy==5.2.2` needs Python 3.13 (separate interpreter). Set `colorspace_settings` *before*
  `pixels.foreach_set`, otherwise the pixels are wiped.
- Never stop processes with `pkill -f <pattern>` when the pattern also matches your own shell command.
  Use PIDs (`pgrep -x`).
- Network: conda-forge (conda.anaconda.org) and PyPI must be allowed in the session's network settings.
