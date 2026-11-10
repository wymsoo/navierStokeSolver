# Navier-Stokes Solver and STRidge Models

This repository contains a 2-D finite-difference incompressible Navier-Stokes solver and STRidge experiments that identify terms in the momentum equations from saved simulation snapshots. The current workflow is organized around `numerical_solver/` and `stridge_model/models/`.

## Project layout

```text
numerical_solver/
  run_all.py                 Batch simulation entry point
  run.py                     Single-run experiment script
  config.py                  Solver parameters and output configuration
  solver_main.py             Projection solver and flow boundary conditions
  models/operators/          Advection, viscosity, pressure, and plotting operators
  metadata.json              Registry of simulation parameters and output paths

stridge_model/models/
  advective_model/           2-D U/V momentum equation identification
  onedir_model/              One-directional flow equation identification

history_trials/              Archived experimental data and reports
reports/                     Project report sources and build artefacts
```

Generated simulation data is written under `stridge_model/` and is ignored by Git. It is not expected to be present in a fresh checkout.

## Requirements

Use Python 3.10 or later with NumPy, SciPy, and Matplotlib. The repository does not currently provide a `requirements.txt` or `pyproject.toml`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib
```

On Windows, activate `.venv` from its `Scripts` directory instead.

## Numerical solver

The solver uses a staggered grid and a projection step. Each time step applies the selected velocity boundary conditions, evaluates advection and viscosity, advances intermediate velocities, solves a sparse pressure Poisson equation, corrects the velocities, and saves an NPZ snapshot. The implemented flow classes are `Poiseuille`, `TwoMovingWalls`, `LidCavitySolver`, and `FourMovingWalls`.

### Run the configured batch

Run from the repository root:

```bash
python numerical_solver/run_all.py
```

The active `FLOW_CONFIGS` in `numerical_solver/run_all.py` currently runs lid-cavity cases with the moving wall on each side and a four-moving-walls case. Other flow entries are present but commented out. Edit `FLOW_CONFIGS` and the parameters in `run_experiment()` to select other cases or change grid size, density, viscosity, time step, body force, and stopping conditions. The defaults use an 81 by 81 grid and can produce long runs.

`numerical_solver/run.py` is an alternative editable experiment script; its flow selection and parameter setup are hard-coded. The batch script is the documented entry point for the current set of cases.

### Solver output

The batch runner writes a numbered run directory under `stridge_model/output_stokes_npz/`, for example:

```text
stridge_model/output_stokes_npz/
  lidcavity_Nx81_sideN/run_001/data_t=1.npz
  lidcavity_Nx81_sideN/run_001/data_t=2.npz
  ...
```

Each compressed snapshot contains `u`, `v`, `p`, `dt`, `dx`, and `dy`. Velocity and pressure are stored on staggered grids. Run parameters and the output directory are appended to `numerical_solver/metadata.json`.

The solver saves field plots periodically and at the end of a run. Plotting functions may open Matplotlib windows depending on the active backend.

## STRidge models

Both learners recursively find run directories containing `data_t=*.npz` files. They calculate time and spatial derivatives with `numpy.gradient`, build candidate-term matrices, fit STRidge models, and write Markdown reports. Supply parent directories that contain one or more run folders; do not point to an individual snapshot file.

### 2-D momentum equations

`stridge_model/models/advective_model/` builds separate candidate libraries for the horizontal and vertical momentum equations using velocity advection, second derivatives, and pressure gradients. Its configuration exposes `--input_directories`, `--data_start`, cropping options, STRidge parameters, and `--report`.

The intended invocation from the repository root is:

```bash
python stridge_model/models/advective_model/run_stridge.py \
  --input_directories stridge_model/output_stokes_npz \
  --data_start 1 \
  --report stridge_model/models/advective_model/stridge_report.md
```

The current 2-D loader randomly samples 1,000 time layers without replacement, so each input run needs at least 1,000 saved snapshots. Its `--crop_t` option is accepted but does not currently affect that sampling. The random sample itself is not controlled by `--seed`; that seed is used for the train/validation split.

### One-directional equation

`stridge_model/models/onedir_model/` extracts a transverse velocity profile from the middle x-index of each snapshot and fits a one-dimensional candidate library, including `u*u_y`, `u*u_yy`, `u_y*u_yy`, `u`, `u_y`, `u_yy`, and a constant. It defaults to searching `one_dir_flow_output_npz/` beside the script and writes a Markdown report beside it.

Example using a directory containing run folders:

```bash
python stridge_model/models/onedir_model/run_stridge_one_dir.py \
  --input_directories stridge_model/output_stokes_npz \
  --data_start 1 \
  --report stridge_model/models/onedir_model/stridge_model_1d_report.md
```

At least three usable snapshots are required per run. `--data_start` defaults to 3000 to ensure steady flow. Use `--crop_t` and `--crop_y` to discard temporal and near-wall samples; other options include `--lam`, `--tol_min`, `--tol_max`, `--num_tol`, `--l0_penalty`, and `--seed`.

## Reproducibility and data notes

- Record the flow type, wall direction where applicable, grid dimensions, forcing, density, viscosity, time step, stopping conditions, STRidge tolerance range, regularisation, cropping, and random seed with each experiment.
- Check `numerical_solver/metadata.json` for the parameters recorded by the simulation runner. The STRidge report records its input paths and fitting settings.
- Snapshot data and generated reports are experiment outputs, not a stable package API. Input paths and run lengths should be checked before starting a training run.
