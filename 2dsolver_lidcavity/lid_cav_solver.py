import numpy as np
import matplotlib.pyplot as plt
from advective import advective
from set_dirichlet_bc_1 import set_Dirichlet_BC
from viscous import viscous
from solve_poisson import Solve_Poisson
from velocityfieldplot import velocityField
from pressurefieldplot import PressureField


def main():
    # Parameters
    dt = 0.01
    
    # Grid size
    Nx = 21
    Ny = 21

    dx = 1.0 / Nx
    dy = 1.0 / Ny

    G = 9.81
    rho = 1000
    
    Re = 500.0 # Reynolds number

    # Velocity fields (staggered)
    U = np.zeros((Nx - 1, Ny))      # u-velocity at x-faces
    V = np.zeros((Nx, Ny - 1))      # v-velocity at y-faces
    
    # Number of iterations
    timesteps = 10000
    H = 5  # Subsampling for visualization
    
    # Time iteration loop
    time = dt
    
    for i in range(1, timesteps + 1):
        # Set Boundary Conditions
        Ubc, Vbc = set_Dirichlet_BC(U, V)
        
        # Non-linear terms
        advectU, advectV = advective(Ubc, Vbc, dx, dy)
        
        # Viscous terms
        viscousU, viscousV = viscous(Ubc, Vbc, Re, dx, dy)
        
        # Compute intermediate velocities
        Ustar = U + advectU * dt + viscousU * dt + G*dt
        Vstar = V + advectV * dt + viscousV * dt 
        
        # Solve Poisson's equation for pressure
        P = Solve_Poisson(Ustar, Vstar, dx, dy, Nx, Ny, dt)
        
        # Compute pressure gradients
        Px = np.diff(P, axis=0) / dx  # x-direction
        Py = np.diff(P, axis=1) / dy  # y-direction
        
        # Apply pressure correction
        U = Ustar - dt * Px
        V = Vstar - dt * Py

        
        # Visualization
        if i % 1000 == 0:
            velocityField(U, V, P, Nx, Ny, time, H)
            PressureField(P, Nx, Ny, time)
        
        print(f"Iteration: {i}")
        time += dt
    
    # Final time adjustment
    print("Simulation completed.")




if __name__ == "__main__":
    main()