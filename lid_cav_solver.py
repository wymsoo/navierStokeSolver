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
    dt = 0.001
    
    # Grid size
    Nx = 31
    Ny = 31
    Nz = 31
    dx = 1.0 / Nx
    dy = 1.0 / Ny
    dz = 1.0 / Nz
    
    Re = 500.0 # Reynolds number

    # Velocity fields (staggered)
    U = np.zeros((Nx - 1, Ny, Nz))      # u-velocity at x-faces
    V = np.zeros((Nx, Ny - 1, Nz))      # v-velocity at y-faces
    W = np.zeros((Nx, Ny, Nz - 1))      # w-velocity at z-faces
    
    # Number of iterations
    timesteps = 10
    H = 5  # Subsampling for visualization
    
    # Time iteration loop
    time = dt
    
    for i in range(1, timesteps + 1):
        # Set Boundary Conditions
        Ubc, Vbc, Wbc = set_Dirichlet_BC(U, V, W)
        # new u (32, 33, 33)
        # new v (33, 32, 33)
        # new w (33, 33, 32)
        
        # Non-linear terms
        advectU, advectV, advectW = advective(Ubc, Vbc, Wbc, dx, dy, dz)
        
        # Viscous terms
        viscousU, viscousV, viscousW = viscous(Ubc, Vbc, Wbc, Re, dx, dy, dz)
        
        # Compute intermediate velocities
        Ustar = U + advectU * dt + viscousU * dt
        Vstar = V + advectV * dt + viscousV * dt
        Wstar = W + advectW * dt + viscousW * dt  # No gravity term
        
        # Solve Poisson's equation for pressure
        P = Solve_Poisson(Ustar, Vstar, Wstar, dx, dy, dz, Nx, Ny, Nz, dt)
        
        # Compute pressure gradients
        Px = np.diff(P, axis=0) / dx  # x-direction
        Py = np.diff(P, axis=1) / dy  # y-direction
        Pz = np.diff(P, axis=2) / dz  # z-direction
        
        # Apply pressure correction
        U = Ustar - dt * Px
        V = Vstar - dt * Py
        W = Wstar - dt * Pz

        print("U:", U)
        
        # Visualization
        if i % 1 == 0:
            velocityField(U, V, W, P, Nx, Ny, Nz, time, H)
            PressureField(P, Nx, Ny, Nz, time)
            # plt.tight_layout()
            # plt.draw()
            # plt.pause(0.01) 
        
        print(f"Iteration: {i}")
        time += dt
    
    # Final time adjustment
    time -= dt
    
    # plt.show()
    print("Simulation completed.")




if __name__ == "__main__":
    main()