# Navier-Stokes Solver and STRidge Model

This repository contains finite-difference incompressible Navier-Stokes solvers and experiments that use Sequential Threshold Ridge Regression (STRidge) to identify PDE terms from generated data.

## Contents

```text
2dsolver/                 Body-force-driven Poiseuille flow
2dsolver_2movingwalls/    Two moving horizontal walls
2dsolver_4movingwalls/    Four moving cavity walls
2dsolver_lidcavity/       Standard 2-D lid-driven cavity
3dsolver/                 3-D lid-driven cavity
stridge_model/            Data generation, STRidge training, and visualisation
reports/                  Project report source and build artefacts
```

The solver variants are intentionally separate copies of the numerical modules. Each variant has its own boundary conditions and parameter setup.

## Requirements

Use Python 3 with NumPy, SciPy, and Matplotlib. The repository does not currently include a `requirements.txt` or `pyproject.toml`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib
```

On macOS, use `source .venv/bin/activate`; on Windows, activate the corresponding `Scripts` directory instead.

## Finite-difference solvers

The 2-D solvers use a staggered grid. Their time step follows this pattern:

1. Apply velocity boundary conditions.
2. Compute advection and viscous terms.
3. Advance the intermediate velocity explicitly.
4. Solve a pressure Poisson equation.
5. Correct the velocity so that it is approximately divergence-free.
6. Save snapshots and produce plots after convergence or the configured iteration limit.

The main implementation is `lid_cav_solver.py`. Supporting modules implement advection, viscosity, divergence, boundary conditions, pressure solution, staggering, and plotting.

### Run a 2-D case

Run each solver from its own directory because the modules use local imports such as `from advective import advective`.

```bash
cd 2dsolver
python lid_cav_solver.py
```

Other 2-D cases:

```bash
cd 2dsolver_2movingwalls && python lid_cav_solver.py
cd 2dsolver_4movingwalls && python lid_cav_solver.py
cd 2dsolver_lidcavity && python lid_cav_solver.py
```

The Poiseuille-style variants centralise their physical and numerical parameters in `global_var.py`. Typical settings are a `31 x 31` grid, `max_ts = 10000`, density `rho = 920`, dynamic viscosity `viscosity = 0.081`, and body force `G = 9.81`. The actual `dt` is calculated from viscous and CFL limits. The lid-cavity variant keeps its parameters in `lid_cav_solver.py`.

### Run the 3-D case

```bash
cd 3dsolver
python lid_cav_solver.py
```

The 3-D solver uses a `31 x 31 x 31` grid, explicit stepping with `dt = 0.001`, and 100 time steps by default. It generates visualisations every ten steps rather than writing the same text snapshot layout as the 2-D solvers.

### Solver output

The 2-D solvers create these directories automatically and write one text file per saved iteration:

```text
<solver>/output/
  u_velocity_field/u_velocity_t=<iteration>.txt
  v_velocity_field/v_velocity_t=<iteration>.txt
  pressure_field/pressure_field_t=<iteration>.txt
```

The 2-D post-processing scripts can be run from their solver directory after a simulation:

```bash
python velocityfieldplot.py
python pressurefieldplot.py
python compare_with_theory_plot.py
python plot_loss.py
```

Some plotting functions call `plt.show()` and therefore expect an interactive Matplotlib backend. The 3-D plotting scripts save images in `3dsolver/velocity/` and `3dsolver/pressure/`.

## STRidge workflow

STRidge builds a candidate library of fields and derivatives, fits the target time derivative, and thresholds small coefficients while selecting the tolerance with a validation score. For startup Poiseuille flow, the target equation is

$$u_t = g + \nu u_{yy},$$

where `g = -(1/rho) * dpdx`.

### 1. Generate reproducible training data

The self-contained generator uses a backward-Euler solve for startup plane Poiseuille flow and stores both 1-D and repeated 2-D fields in an `.npz` file.

```bash
cd stridge_model
python generate_startup_poiseuille.py \
  --output startup_poiseuille_data.npz \
  --Ny 129 --Nx 16 --dt 5e-4 --t_end 1.0 \
  --nu 0.05 --rho 1.0 --dpdx -1.0
```

Useful options include `--noise` and `--seed` for reproducible noisy measurements. The generated file contains `u`, `u2d`, `t`, `x`, `y`, `dt`, `dy`, `dx`, `nu`, `rho`, `dpdx`, `g`, and the analytical steady profile.

### 2. Learn the PDE

```bash
cd stridge_model
python learn_startup_poiseuille.py \
  --input startup_poiseuille_data.npz \
  --lam 1e-8 --tol_min 1e-8 --tol_max 1e-1 \
  --num_tol 40 --l0_penalty 1e-6 --seed 0 \
  --crop_t 1 --crop_y 2
```

The learner computes `u_t`, `u_y`, and `u_yy`, constructs a library containing constant, linear, polynomial, derivative, and product terms, and reports the selected tolerance and errors. Boundary and temporal cropping reduces finite-difference edge effects.

### Other STRidge scripts

These scripts are exploratory variants and expect particular existing data layouts:

```bash
cd stridge_model
python model.py
python model_2d.py
python model_burger.py
python visualize_data.py
```

`model.py` reads lid-cavity snapshots from `stridge_model/output_lidcav/`; `model_2d.py` reads 2-D velocity and pressure snapshots from `output_4movingwalls/`; and `model_burger.py` reads `burgers_shock.mat`. Check the input directories and snapshot numbering before running them. These scripts are not unified command-line applications and contain hard-coded data paths and experiment-specific candidate libraries.

## Data and path caveats

- Run solver modules from their variant directory so local imports resolve consistently.
- STRidge snapshot learners require contiguous, correctly named files such as `u_velocity_t=10.txt`.
- `learn_startup_poiseuille.py` currently imports `endtime` and `time_res` from `global_var.py`, although those names are not defined there. The import must be corrected before that script can run in the current checkout.
- `visualize_data.py` refers to a `2d_stridge_model/` input path, while the checked-in Burgers data is under `stridge_model/`.
- The root `test.py` is a tolerance-grid plotting experiment; it is not an automated solver test.
- Text snapshots and generated images are experiment outputs, not a stable package API.

## Reproducibility checklist

Record the solver variant, grid dimensions, physical parameters, time step, number of iterations, boundary condition case, STRidge library, cropping values, regularisation parameter, tolerance range, and random seed with each result. This is especially important because most scripts currently store configuration in module-level constants rather than a shared configuration file.
