import numpy as np
from advective import advective
from viscous import viscous
from global_var import dt,dx,dy,Re, viscosity, rho

def padding(U,V):
    # For U - pad columns first, then rows (like you started)
    U = np.insert(U, 0, U[:, 0], axis=1)   # Add column at beginning
    U = np.append(U, U[:, -1:], axis=1)    # Add column at end  
    U = np.insert(U, 0, U[0, :], axis=0)   # Add row at top
    U = np.append(U, U[-1:, :], axis=0)    # Add row at bottom

    # For V - use the SAME order of operations
    V = np.insert(V, 0, V[:, 0], axis=1)   # Add column at beginning
    V = np.append(V, V[:, -1:], axis=1)    # Add column at end
    V = np.insert(V, 0, V[0, :], axis=0)   # Add row at top
    V = np.append(V, V[-1:, :], axis=0)    # Add row at bottom

    return U,V

def remove_padding(U,V):
    return U[1:-1, 1:-1],V[1:-1, 1:-1]


def rk4_velocityfield(Ubc, Vbc, U, V):
        ## RK4 method
        advU1, advV1 = advective(Ubc, Vbc, dx, dy) 
        advU1, advV1 =  padding(advU1, advV1)
        k1U_adv = dt * np.advU1
        k1V_adv = dt * advV1
        
        U_temp = Ubc + 0.5 * k1U_adv
        V_temp = Vbc + 0.5 * k1V_adv
        
        advU2, advV2 = advective(U_temp, V_temp, dx, dy)
        advU2, advV2 =  padding(advU2, advV2)
        k2U_adv = dt * advU2
        k2V_adv = dt * advV2
        
        U_temp = Ubc + 0.5 * k2U_adv
        V_temp = Vbc + 0.5 * k2V_adv
        
        advU3, advV3 = advective(U_temp, V_temp, dx, dy)
        advU3, advV3 =  padding(advU3, advV3)
        k3U_adv = dt * advU3
        k3V_adv = dt * advV3
        
        U_temp = Ubc + k3U_adv
        V_temp = Vbc + k3V_adv
        
        advU4, advV4 = advective(U_temp, V_temp, dx, dy)
        advU4, advV4 =  padding(advU4, advV4)
        k4U_adv = dt * advU4
        k4V_adv = dt * advV4
        
        advectU = (k1U_adv + 2*k2U_adv + 2*k3U_adv + k4U_adv) / 6.0
        advectV = (k1V_adv + 2*k2V_adv + 2*k3V_adv + k4V_adv) / 6.0
        advectU,advectV = remove_padding(advectU,advectV)
        

        visU1, visV1 = viscous(Ubc, Vbc, Re, dx, dy, viscosity)
        visU1, visV1 =  padding(visU1, visV1)
        k1U_visc = dt * visU1
        k1V_visc = dt * visV1
        
        U_temp = Ubc + 0.5 * k1U_visc
        V_temp = Vbc + 0.5 * k1V_visc
        
        visU2, visV2 = viscous(U_temp, V_temp, Re, dx, dy, viscosity)
        visU2, visV2 =  padding(visU2, visV2)
        k2U_visc = dt * visU2
        k2V_visc = dt * visV2
        
        U_temp = Ubc + 0.5 * k2U_visc
        V_temp = Vbc + 0.5 * k2V_visc
        
        visU3, visV3 = viscous(U_temp, V_temp, Re, dx, dy, viscosity)
        visU3, visV3 =  padding(visU3, visV3)
        k3U_visc = dt * visU3
        k3V_visc = dt * visV3
        
        U_temp = Ubc + k3U_visc
        V_temp = Vbc + k3V_visc
        
        visU4, visV4 = viscous(U_temp, V_temp, Re, dx, dy, viscosity)
        visU4, visV4 =  padding(visU4, visV4)
        k4U_visc = dt * visU4
        k4V_visc = dt * visV4
        
        viscousU = (k1U_visc + 2*k2U_visc + 2*k3U_visc + k4U_visc) / 6.0
        viscousV = (k1V_visc + 2*k2V_visc + 2*k3V_visc + k4V_visc) / 6.0
        viscousU,viscousV = remove_padding(viscousU,viscousV)
        
        Ustar = U + advectU + viscosity/rho * viscousU 
        Vstar = V + advectV + viscosity/rho * viscousV 