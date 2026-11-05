import numpy as np
import matplotlib.pyplot as plt
from models.operators.advective import advective
from models.operators.viscous import viscous
from models.operators.solve_poisson import Solve_Poisson
from models.operators.velocityfieldplot import velocityField
from models.operators.pressurefieldplot import PressureField
from models.operators.plot_loss import plot_loss
import sys
import os
from pathlib import Path
import config

class NavierStokesSolver2d:
    def __init__(self):
        self.Nx = config.Nx
        self.Ny = config.Ny
        self.dt = config.dt_fixed
        self.l = config.L
        self.w = config.W
        self.dx = config.dx
        self.dy = config.dy
        self.d_viscosity = config.viscosity
        self.density = config.rho
        self.k_viscosity = config.viscosity / config.rho
        self.epsilon = config.epsilon
        self.Fx_ext = config.Fx_ext
        self.Fy_ext = config.Fy_ext
        self.max_ts = config.max_ts
        self.U = np.zeros((self.Nx - 1, self.Ny))      # u-velocity at x-faces
        self.V = np.zeros((self.Nx, self.Ny - 1))
        self.P = np.zeros((self.Nx-1,self.Ny-1))
        self.iter = []
        self.loss = []
        self.fluid_data_output_dir = config.output_dir
        self.base_dir = config.base_dir
        self.stopping_bc_conditions = config.stopping_bc_conditions
        self.stopping_bc_time = config.stopping_bc_time
        self.relaxation_time = config.relaxation_time
        self.active_boundary_condition = self.set_boundary_condition 
        self.periodic_pressure_x = False
        self.periodic_pressure_y = False
        self.change_boundary_on_stop = True
        self.active_Fx_ext = self.Fx_ext
        self.active_Fy_ext = self.Fy_ext
        self.flow_stopped = False

    def compute_reynold(self,u_max,L_flowdir):
        if L_flowdir==0:
            d = self.l
        elif L_flowdir==1:
            d = self.w

        return (self.density * u_max * d) / self.d_viscosity

    def viscous(self,Ubc,Vbc):
        return viscous(Ubc,Vbc,self.k_viscosity,self.dx,self.dy)
        
    def advective(self,Ubc,Vbc):
        return advective(Ubc,Vbc, self.dx, self.dy)

    def solve_poisson(self, Ustar, Vstar, dt):
        return Solve_Poisson(self, Ustar,Vstar,self.dx, self.dy, self.Nx, self.Ny, dt)

    def plot_fields(self,time,path_dir):
        velocityField(self.U,self.V,self.P,self.Nx,self.Ny,time,path_dir)
        PressureField(self.P,self.Nx,self.Ny,time)
        return

    def plot_loss(self,path_dir):
        if (path_dir):
            path_dir = path_dir
        else:
            path_dir = Path(__file__).resolve().parent / 'plots'
        plot_loss(self.iteration, self.loss, path_dir)

        return

    # Adaptive Timesteps
    def compute_timestep(self, dt_prev, u_wall=1.0, cfl=0.3):
        # 1. Compute local velocity magnitudes
        u_mag = np.sqrt(self.U[:,:-1]**2 + self.V[:-1,:]**2)
        u_max = np.max(u_mag)
        
        # Fallback to boundary wall velocity at step 0 if interior flow is at rest
        if u_max < 1e-6:
            u_max = u_wall

        # 2. Compute minimum grid spacing
        ds_min = min(self.dx, self.dy)
        # 3. Compute Advective CFL Limit
        dt_cfl = cfl * (ds_min / u_max)
        # 4. Compute Viscous / Diffusion Limit
        dt_visc = 0.25 * (self.density * (ds_min**2) / self.d_viscosity)
        # 5. Take strict minimum between CFL and Viscous criteria
        dt_target = min(dt_cfl, dt_visc)

        # 6. Apply 10% maximum growth cap from previous timestep (if available)
        if dt_prev is not None:
            dt_new = min(dt_target, 1.1 * dt_prev)
        else:
            dt_new = dt_target

        return dt_new

    def stopping_bc(self, U, V):
        # Apply stationary no-slip walls while preserving the unpadded state.
        lenU = U.shape
        Unorth = -U[:, -1].reshape((lenU[0], 1))
        Usouth = -U[:, 0].reshape((lenU[0], 1))
        U_padded = np.concatenate((Usouth, U, Unorth), axis=1)
        lenU = U_padded.shape
        Uwest = np.zeros((1, lenU[1]))
        Ueast = np.zeros((1, lenU[1]))
        U_padded = np.concatenate((Uwest, U_padded, Ueast), axis=0)

        lenV = V.shape
        Vsouth = np.zeros((lenV[0], 1))
        Vnorth = np.zeros((lenV[0], 1))
        V_padded = np.concatenate((Vsouth, V, Vnorth), axis=1)
        lenV = V_padded.shape
        Vwest = -V_padded[0, :].reshape((1, lenV[1]))
        Veast = -V_padded[-1, :].reshape((1, lenV[1]))
        V_padded = np.concatenate((Vwest, V_padded, Veast), axis=0)

        return U_padded, V_padded

    def update_boundary_condition(self, time):
        if (
            not self.flow_stopped
            and
            self.stopping_bc_conditions
            and self.stopping_bc_time is not None
            and time >= self.stopping_bc_time
        ):
            self.active_Fx_ext = 0.0
            self.active_Fy_ext = 0.0
            if self.change_boundary_on_stop:
                self.active_boundary_condition = self.stopping_bc
                self.periodic_pressure_x = False
                self.periodic_pressure_y = False
            self.flow_stopped = True

    def run(self):
        dt = self.dt
        time = 0
        for i in range(1, self.max_ts + 1):
            self.update_boundary_condition(time)

            dt = self.compute_timestep(dt_prev=dt, u_wall=1.0, cfl=0.3) # cfl 40% safety buffer below the theoretical 2D limit (0.5) 
            if (
                not self.flow_stopped
                and self.stopping_bc_conditions
                and self.stopping_bc_time is not None
                and time < self.stopping_bc_time < time + dt
            ):
                dt = self.stopping_bc_time - time
            # Set Boundary Conditions
            U_copy = self.U.copy()
            V_copy = self.V.copy()
            Ubc, Vbc = self.active_boundary_condition(U_copy, V_copy)
            
            # Non-linear terms
            advectU, advectV = self.advective(Ubc, Vbc)
            viscousU, viscousV = self.viscous(Ubc, Vbc)
            
            # Compute intermediate velocities
            Ustar = U_copy - advectU * dt + viscousU * dt + self.active_Fx_ext * dt
            Vstar = V_copy - advectV * dt + viscousV * dt + self.active_Fy_ext * dt
            
            # Solve Poisson's equation for pressure
            self.P = self.solve_poisson(Ustar, Vstar, dt)
            P_copy = self.P.copy()
            # Compute pressure gradients
            Px = np.diff(P_copy, axis=0) / self.dx  # x-direction
            Py = np.diff(P_copy, axis=1) / self.dy  # y-direction
            
            # Apply pressure correction
            U_new = Ustar - dt * Px / self.density
            V_new = Vstar - dt * Py / self.density

            U_loss = np.sum((U_new-U_copy)**2/(self.Nx*self.Ny)) #MSE
            V_loss = np.sum((V_new-V_copy)**2/(self.Nx*self.Ny))
            total_loss = U_loss+V_loss
            print(f"iteration{i} total loss:", total_loss)
            self.U = U_new
            self.V = V_new

            self.iter.append(i)
            self.loss.append(total_loss)

            time += dt

            np.savez_compressed(
                os.path.join(self.fluid_data_output_dir,f"data_t={i}.npz"),
                u=self.U,
                v=self.V,
                p=self.P,
                dt=dt,
                dx=self.dx,
                dy=self.dy
            )

            relaxation_finished = (
                self.stopping_bc_conditions
                and self.flow_stopped
                and self.stopping_bc_time is not None
                and time >= self.stopping_bc_time + self.relaxation_time
            )
            if relaxation_finished:
                break

            if not self.stopping_bc_conditions and total_loss <= self.epsilon:
                break

        # uncomment this line if you want to plot the dynamic field changes
            if i % 50 == 0:
                self.plot_fields(time, self.base_dir)

        self.plot_fields(time, self.base_dir)
        print("Simulation completed.")
        u_max = np.max(self.U)
        v_max = np.max(self.V)
        max_v = max(u_max, v_max)
        print("max v:",max_v)
        L_flowdirection = 0 if (u_max>v_max) else 1
        Re = self.compute_reynold(u_max,L_flowdirection)
        print("Reynolds number=", Re)
        print("Output saved to ", self.fluid_data_output_dir)

        
class LidCavitySolver(NavierStokesSolver2d):
    def __init__(self):
        super().__init__()
    def set_boundary_condition(self,U,V):
        lenU = U.shape
        # Apply north and south boundaries for U
        lenU = U.shape
        Unorth = 2*np.ones((lenU[0], 1)) - U[:,-1].reshape((lenU[0],1))  # set the last row of velocity field to be north boundary
        Usouth = -U[:, 0].reshape((lenU[0],1)) # set the first row of velocity field to be south boundary
        U = np.concatenate((Usouth, U), axis=1)
        U = np.concatenate((U, Unorth), axis=1) #u_1/2 = c1
        # Apply front and back boundaries for U
        lenU = U.shape
        Ueast = np.zeros((1, lenU[1]))
        Uwest = np.zeros((1, lenU[1]))
        U = np.concatenate((Ueast, U), axis=0)
        U = np.concatenate((U, Uwest), axis=0)
        # Same thing for V
        lenV = V.shape
        Vnorth = np.zeros((lenV[0],1))
        Vsouth = np.zeros((lenV[0],1))
        V = np.concatenate((Vsouth, V), axis=1)
        V = np.concatenate((V, Vnorth), axis=1)
        lenV = V.shape
        # Apply front and back boundaries for V
        Veast = -V[-1, :].reshape((1, lenV[1]))
        Vwest = -V[0, :].reshape((1, lenV[1]))
        V = np.concatenate((Vwest, V), axis=0)
        V = np.concatenate((V, Veast), axis=0)

        return U, V

class FourMovingWalls(NavierStokesSolver2d):
    def __init__(self):
        super().__init__()
    def set_boundary_condition(self,U,V,u_wall=1.0):
        # =========================================================
        # 1. HORIZONTAL VELOCITY (U)
        # =========================================================
        # --- Top/Bottom Boundaries (Tangential Shear) ---
        # Linear interpolation across wall: (U_ghost + U_interior)/2 = U_wall
        lenU = U.shape
        Unorth = -2.0 * u_wall - U[:, -1].reshape((lenU[0], 1))  # Moving Left (-u_wall)
        Usouth =  2.0 * u_wall - U[:,  0].reshape((lenU[0], 1))  # Moving Right (+u_wall)
        
        U_padded = np.concatenate((Usouth, U, Unorth), axis=1)

        # --- Left/Right Boundaries (Normal Impermeable Walls) ---
        # U faces sit directly ON the East/West walls -> U_wall = 0
        lenU_p = U_padded.shape
        Uwest = np.zeros((1, lenU_p[1]))
        Ueast = np.zeros((1, lenU_p[1]))
        
        U_padded = np.concatenate((Uwest, U_padded, Ueast), axis=0)


        # =========================================================
        # 2. VERTICAL VELOCITY (V)
        # =========================================================
        # --- Top/Bottom Boundaries (Normal Impermeable Walls) ---
        # V faces sit directly ON the North/South walls -> V_wall = 0
        lenV = V.shape
        Vsouth = np.zeros((lenV[0], 1))
        Vnorth = np.zeros((lenV[0], 1))
        
        V_padded = np.concatenate((Vsouth, V, Vnorth), axis=1)

        # --- Left/Right Boundaries (Tangential Shear) ---
        # Linear interpolation across wall: (V_ghost + V_interior)/2 = V_wall
        lenV_p = V_padded.shape
        Vwest = -2.0 * u_wall - V_padded[ 0, :].reshape((1, lenV_p[1])) # Moving Down (-u_wall)
        Veast =  2.0 * u_wall - V_padded[-1, :].reshape((1, lenV_p[1])) # Moving Up (+u_wall)
        
        V_padded = np.concatenate((Vwest, V_padded, Veast), axis=0)

        return U_padded, V_padded

class TwoMovingWalls(NavierStokesSolver2d):
    def __init__(self):
        super().__init__()
        self.change_boundary_on_stop = False
        self.periodic_pressure_x = self.Fx_ext != 0 and self.Fy_ext == 0
        self.periodic_pressure_y = self.Fy_ext != 0 and self.Fx_ext == 0
    def set_boundary_condition(self,U,V,u_wall=1.0):
        lenU = U.shape
        Uwest = U[-1,:].reshape((1,lenU[1]))
        Ueast = U[0,:].reshape((1,lenU[1]))                            #periodic boundary conditions 
        U = np.concatenate((Uwest, U), axis=0)
        U = np.concatenate((U, Ueast), axis=0)

        u_wall = 1.0
        lenU = U.shape
        Unorth = -2.0 * u_wall - U[:, -1].reshape((lenU[0], 1))  # Moving Left (-u_wall)
        Usouth =  2.0 * u_wall - U[:,  0].reshape((lenU[0], 1))  # Moving Right (+u_wall)
        U = np.concatenate((Usouth, U, Unorth), axis=1)

        lenV = V.shape
        Veast = -V[-1,:].reshape((1, lenV[1])) #no vertical velocity
        Vwest = -V[0,:].reshape((1, lenV[1]))
        V = np.concatenate((Vwest, V), axis=0)
        V = np.concatenate((V, Veast), axis=0)

        lenV = V.shape
        Vnorth= np.zeros((lenV[0], 1)) #no slip condition
        Vsouth= np.zeros((lenV[0], 1))
        V = np.concatenate((Vsouth, V), axis=1)
        V = np.concatenate((V, Vnorth), axis=1)
        

        return U,V

class Poiseuille(NavierStokesSolver2d):
    def __init__(self):
        super().__init__()
        self.change_boundary_on_stop = False
        self.periodic_pressure_x = self.Fx_ext != 0 and self.Fy_ext == 0
        self.periodic_pressure_y = self.Fy_ext != 0 and self.Fx_ext == 0
    def set_boundary_condition(self,U,V):
        if (self.Fx_ext>0 and self.Fy_ext==0):
            lenU = U.shape
            Uwest = U[-1,:].reshape((1,lenU[1]))
            Ueast = U[0,:].reshape((1,lenU[1]))                            #periodic boundary conditions 
            U = np.concatenate((Uwest, U), axis=0)
            U = np.concatenate((U, Ueast), axis=0)

            lenU = U.shape
            Unorth = - U[:,-1].reshape((lenU[0],1))  # for flow, there is no slip in the top and bottom boundary
            Usouth = - U[:,0].reshape((lenU[0],1))
            U = np.concatenate((Usouth, U), axis=1)
            U = np.concatenate((U, Unorth), axis=1)

            lenV = V.shape
            Veast = -V[-1,:].reshape((1, lenV[1])) #no vertical velocity
            Vwest = -V[0,:].reshape((1, lenV[1]))
            V = np.concatenate((Vwest, V), axis=0)
            V = np.concatenate((V, Veast), axis=0)

            lenV = V.shape
            
            Vnorth= np.zeros((lenV[0], 1)) #no slip condition
            Vsouth= np.zeros((lenV[0], 1))
            V = np.concatenate((Vsouth, V), axis=1)
            V = np.concatenate((V, Vnorth), axis=1)

        elif (self.Fy_ext>0 and self.Fx_ext==0):
            # --- V is the streamwise velocity (y-direction) ---
            lenV = V.shape
            # Periodic boundary conditions in the y-direction (axis 1)
            Vsouth = V[:, -1].reshape((lenV[0], 1))
            Vnorth = V[:, 0].reshape((lenV[0], 1))
            V = np.concatenate((Vsouth, V), axis=1)
            V = np.concatenate((V, Vnorth), axis=1)

            # No-slip condition at west and east walls (axis 0)
            lenV = V.shape
            Vwest = -V[0, :].reshape((1, lenV[1]))
            Veast = -V[-1, :].reshape((1, lenV[1]))
            V = np.concatenate((Vwest, V), axis=0)
            V = np.concatenate((V, Veast), axis=0)

            # --- U is the wall-normal velocity (x-direction) ---
            lenU = U.shape
            # Periodic boundary conditions in the y-direction (axis 1) 
            # with staggered grid anti-symmetric sign convention
            Usouth = -U[:, -1].reshape((lenU[0], 1))
            Unorth = -U[:, 0].reshape((lenU[0], 1))
            U = np.concatenate((Usouth, U), axis=1)
            U = np.concatenate((U, Unorth), axis=1)

            # No-penetration (zero velocity) at west and east walls (axis 0)
            lenU = U.shape
            Uwest = np.zeros((1, lenU[1]))
            Ueast = np.zeros((1, lenU[1]))
            U = np.concatenate((Uwest, U), axis=0)
            U = np.concatenate((U, Ueast), axis=0)

        else:
            print("Poiseuille flow only allows external force in one direction.")
            sys.exit()
        

        return U,V











    
