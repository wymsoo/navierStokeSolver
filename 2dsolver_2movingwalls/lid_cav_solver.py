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
from global_var import Nx, Ny, Re, D, G ,dx, dy, dt, max_ts, H, L, rho, viscosity, epsilon
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    u_dir = os.path.join(output_dir, "u_velocity_field")
    v_dir = os.path.join(output_dir, "v_velocity_field")
    p_dir = os.path.join(output_dir, "pressure_field")
    os.makedirs(u_dir, exist_ok=True)
    os.makedirs(v_dir, exist_ok=True)
    os.makedirs(p_dir, exist_ok=True)

    P = np.zeros((Nx, Ny))
    y = np.linspace(-D/2,D/2,Ny)
    x = np.linspace(0,D,Nx)
    divFreePressure = P+rho*G*x #added divergence Free pressure
    # print(divFreePressure)
    # Velocity fields (staggered)
    U = np.zeros((Nx - 1, Ny))      # u-velocity at x-faces
    V = np.zeros((Nx, Ny - 1))      # v-velocity at y-faces
    iteration = []
    loss = []
    stag_Ux = []
    stag_Uy = []
    stag_Vx = []
    stag_Vy = []
    
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
        # print("Pressure Gradient", Px)
        # print("Pressure Gradient Y", Py)
        
        # Apply pressure correction
        U_new = Ustar - dt * Px/rho
        V_new = Vstar - dt * Py/rho
        # U_new = Ustar
        # V_new = Vstar
    
        U_loss = np.sum((U_new-U)**2/(Nx*Ny)) #MSE
        V_loss = np.sum((V_new-V)**2/(Nx*Ny))
        total_loss = U_loss+V_loss
        U = U_new
        V = V_new

        if ((U_loss+V_loss)<epsilon):
            break
        if (i%100==0):
            print(f"Iteration: {i}")
        time += dt
        iteration.append(i)
        loss.append(total_loss)
    
        if (i % 1 == 0):
            # velocityField(U,V,P, Nx, Ny, time, H)
            np.savetxt(os.path.join(u_dir, f"u_velocity_t={i}.txt"), U, delimiter="\t")
            np.savetxt(os.path.join(v_dir, f"v_velocity_t={i}.txt"), V, delimiter="\t")
            np.savetxt(os.path.join(p_dir, f"pressure_field_t={i}.txt"), P, delimiter="\t")
        

    # Final time adjustment
    time -= dt
    velocityField(U,V,P, Nx, Ny, time, H)
    plot_loss(iteration,loss)
    print("Simulation completed.")
    print("REYNOLD", Re)



if __name__ == "__main__":
    main()