import numpy as np
import matplotlib.pyplot as plt
import config
import os
from pathlib import Path
from metadata_schema import ExperimentMetadata
from solver_main import NavierStokesSolver2d, LidCavitySolver, FourMovingWalls, TwoMovingWalls, Poiseuille

# 1. Select a flow
# flow = "poiseuille"
# flow = "twomovingwalls"
# flow = "lidcavity"
flow = "fourmovingwalls"

# 2. Select Grid Size
Nx = 31
Ny = 31

# 3. Define fluid parameters
fx_ext = 0.0
fy_ext = 0.0
rho = 10
u_max = 1
viscosity = 10

# define stopping conditions if desired
# stopping_bc_conditions = True
# stopping_bc_time = 0.5  # Physical time at which the boundary condition changes
# relaxation_time = 0.1  # Physical time to simulate after the transition

stopping_bc_conditions = False
stopping_bc_time = 0.0  # Physical time at which the boundary condition changes
relaxation_time = 0.0  # Physical time to simulate after the transition


# 4. System Setup
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
experiment_root = Path(os.path.dirname(base_dir)) / "ns_solver" / "stridge_model" / "output_npz"
experiment_name = f"{flow}_Nx{Nx}"
experiment_dir = experiment_root / experiment_name
experiment_dir.mkdir(parents=True, exist_ok=True)

existing_runs = [
    path for path in experiment_dir.glob("run_*")
    if path.is_dir() and path.name[4:].isdigit()
]
run_number = max(
    (int(path.name[4:]) for path in existing_runs),
    default=0,
) + 1
run_name = f"run_{run_number:03d}"
trial = f"{experiment_name}_{run_name}"
output_dir = experiment_dir / run_name
output_dir.mkdir()
# initial dt for convergence, will be set to adaptive timesteps in runtime; values are by experiment
dt = 0.00001 if (flow=="poiseuille" or flow == "twomovingwalls") else 0.0000001


# 5. Initialize solver with parameters
config.init(
    Nx, Ny, trial, base_dir, str(output_dir), dt, rho, u_max, viscosity, fx_ext, fy_ext,
    stopping_bc_conditions_var=stopping_bc_conditions,
    stopping_bc_time_var=stopping_bc_time,
    relaxation_time_var=relaxation_time,
)


# 6. Store metadata to keep track of parameters used
ExperimentMetadata(
    run_number=run_number,
    run_name=run_name,
    output_dir=str(output_dir),
    flow=flow,
    Nx=Nx,
    Ny=Ny,
    Fx_ext=fx_ext,
    Fy_ext=fy_ext,
    rho=rho,
    u_max=u_max,
    viscosity=viscosity,
    dt_initial=dt,
    stopping_bc_conditions=stopping_bc_conditions,
    stopping_bc_time=stopping_bc_time,
    relaxation_time=relaxation_time,
).append_to(Path(__file__).with_name("metadata.json"))


match flow:
    case "poiseuille":
        solver = Poiseuille()
    case "lidcavity":
        solver = LidCavitySolver()
    case "fourmovingwalls":
        solver = FourMovingWalls()
    case "twomovingwalls":
        solver = TwoMovingWalls()

solver.run()