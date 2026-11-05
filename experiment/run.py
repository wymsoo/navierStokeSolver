import numpy as np
import matplotlib.pyplot as plt
import config
import os
from solver_main import NavierStokesSolver2d, LidCavitySolver, FourMovingWalls, TwoMovingWalls, Poiseuille

 # 1. Select a flow
flow = "poiseuille"
# flow = "twomovingwalls"
# flow = "lidcavity"
# flow = "fourmovingwalls"

# 2. Select Grid Size
Nx = 31
Ny = 31

# 3. Define fluid parameters
fx_ext = 10.0
fy_ext = 0.0
rho = 10
u_max = 1
viscosity = 10


# 4. System Setup
trial = f"{flow}_{Nx}"
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(os.path.dirname(base_dir), f"ns_solver/stridge_model/output_npz/{trial}_Nx={Nx}_Fx={fx_ext}_Fy={fy_ext}")
os.makedirs(output_dir, exist_ok=True)
# initial dt for convergence, will be set to adaptive timesteps in runtime; values are by experiment
dt = 0.00001 if (flow=="poiseuille" or flow == "twomovingwalls") else 0.0000001


# 5. Initialize solver with parameters
config.init(Nx, Ny, trial, base_dir, output_dir, dt, rho, u_max, viscosity, fx_ext, fy_ext)
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