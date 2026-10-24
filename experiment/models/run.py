import numpy as np
import matplotlib.pyplot as plt
import os
from solver_main import NavierStokesSolver2d, LidCavitySolver, FourMovingWalls, TwoMovingWalls, Poiseuille, Poiseuille_with_thing

Nx = 31
Ny = 31
trial = f"lid_cavity_{Nx}"
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(os.path.dirname(base_dir), f"stridge_model/output_npz/{trial}")
os.makedirs(output_dir, exist_ok=True)
# Parameters
dt = 0.00001 # for poiseuille
#dt = 0.0000001 
rho = 10
u_max = 1
viscosity = 10

solver = Poiseuille_with_thing(Nx,Ny,dt,viscosity,rho,output_dir,base_dir,Fx_ext=10.0)
solver.run()