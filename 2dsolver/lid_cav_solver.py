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
    Nx = 11
    Ny = 11
    dx = 1.0 / Nx
    dy = 1.0 / Ny


    Re = 500 # Reynolds number
    rho = 900
    G = 9.81

    # Velocity fields (staggered)
    U = np.zeros((Nx - 1, Ny))      # u-velocity at x-faces
    V = np.zeros((Nx, Ny - 1))      # v-velocity at y-faces
    
    # Number of iterations
    timesteps = 10
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
        # print(viscousU*dt)
        
        # Compute intermediate velocities
        Ustar = U + advectU * dt + viscousU * dt + G*dt
        Vstar = V + advectV * dt + viscousV * dt
        # Wstar = W + advectW * dt + viscousW * dt  # No gravity term
        print("Stars",Ustar,Vstar)
        
        # Solve Poisson's equation for pressure
        P = Solve_Poisson(Ustar, Vstar, dx, dy, Nx, Ny, dt)
        
        # Compute pressure gradients
        Px = np.diff(P, axis=0) / dx  # x-direction
        Py = np.diff(P, axis=1) / dy  # y-direction
        # Pz = np.diff(P, axis=2) / dz  # z-direction
        
        # Apply pressure correction
        U = Ustar - dt * Px
        V = Vstar - dt * Py
        # W = Wstar - dt * Pz
        print("U:",U)
        print("V",V)
        V[0,0] = 0
        
        # Visualization
        if i % 1 == 0:
            velocityField(U, V, P, Nx, Ny, time, H)
            PressureField(P, Nx, Ny, time)
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