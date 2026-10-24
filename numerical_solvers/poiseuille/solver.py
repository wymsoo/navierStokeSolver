import numpy as np
import matplotlib.pyplot as plt
from advective import advective
from set_dirichlet_bc_1 import set_Dirichlet_BC
from viscous import viscous
from solve_poisson import Solve_Poisson
from velocityfieldplot import velocityField
from pressurefieldplot import PressureField
from stagger import stagger_back
from plot_loss import plot_loss
from compare_with_theory_plot import compare_with_theory
# from global_var import Nx, Ny, Re, D, G ,dx, dy, dt, max_ts, H, L, rho, viscosity, epsilon
import os

def main():
    Nx = 31
    Ny = 31
    trial = f"poiseuille_vertical_{31}"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(os.path.dirname(base_dir), f"stridge_model/output_npz/{trial}")
    os.makedirs(output_dir, exist_ok=True)

    # Parameters
    max_ts = 10000
    dt = 0.001
    epsilon = 5e-10
    # Grid size

    dx = 1.0 / Nx
    dy = 1.0 / Ny
    G = 9.81
    rho = 100
    # D = 0.005
    D = 1.0
    viscosity = 10

    P = np.zeros((Nx, Ny))
    y = np.linspace(-D/2,D/2,Ny)
    x = np.linspace(0,D,Nx)
    divFreePressure = P+rho*G*x #added divergence Free pressure
    print(divFreePressure)
    # Velocity fields (staggered)
    U = np.zeros((Nx - 1, Ny))      # u-velocity at x-faces
    V = np.zeros((Nx, Ny - 1))      # v-velocity at y-faces
    iteration = []
    loss = []
    
    # Time iteration loop
    time = dt
    
    for i in range(1, max_ts + 1):

        Ubc, Vbc = set_Dirichlet_BC(U, V)

        #Euler's method
        advectU, advectV = advective(Ubc, Vbc, dx, dy)
        viscousU, viscousV = viscous(Ubc, Vbc, Re, dx, dy, viscosity)

        Ustar = U + advectU * dt + viscosity/rho * viscousU * dt + G*dt
        Vstar = V + advectV * dt + viscosity/rho * viscousV * dt

        # Solve Poisson's equation for pressure
        P = Solve_Poisson(Ustar, Vstar, dx, dy, Nx, Ny, dt, pressure=divFreePressure) 
        # print("Pressure", P)
        # Compute pressure gradients
        Px = np.diff(P, axis=0) / dx
        Py = np.diff(P, axis=1) / dy 
        
        # Apply pressure correction
        U_new = Ustar - dt * Px/rho
        V_new = Vstar - dt * Py/rho
    
        U_loss = np.sum((U_new-U)**2/(Nx*Ny)) #MSE
        V_loss = np.sum((V_new-V)**2/(Nx*Ny))
        total_loss = U_loss+V_loss
        U = U_new
        V = V_new

        if ((U_loss+V_loss)<epsilon):
            break
            
        print(f"Iteration: {i} || Loss: {total_loss}")
        time += dt
        iteration.append(i)
        loss.append(total_loss)
    
        if (i % 1 == 0):
            np.savez_compressed(
                os.path.join(output_dir,f"data_t={i}.npz"),
                u=U,
                v=V,
                p=P,
                dt=dt,
                dx=dx,
                dy=dy
            )
        

    # Final time adjustment
    time -= dt
    velocityField(U,V,P, Nx, Ny, time)
    compare_with_theory(U,V)
    plot_loss(iteration,loss)
    print(f"Simulation completed. File written to {output_dir}")
    mag_v = np.sqrt(np.max(abs(U))**2+np.max(abs(V))**2)
    print("mag v:",mag_v)
    Re = (rho * mag_v * D) / viscosity
    print("Reynolds number=", Re)



if __name__ == "__main__":
    main()