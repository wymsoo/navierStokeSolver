import numpy as np 
from scipy.sparse.linalg import spsolve
from laplacian import Laplacian
from divergence import div
from set_dirichlet_bc_1 import set_Dirichlet_BC
from global_var import rho


def Solve_Poisson(Ustar: np.ndarray, Vstar: np.ndarray,dx: float, dy: float, Nx: int, Ny: int, dt: float, pressure=None) -> np.ndarray:
    

    # Apply Dirichlet boundary conditions
    Ustarbc, Vstarbc = set_Dirichlet_BC(Ustar, Vstar)
    P = np.zeros((Nx,Ny))
    # Compute divergence 
    RHS = (1/dt) * div(Ustarbc, Vstarbc, dx, dy)
    # print(RHS)
    # if np.allclose(RHS, 0.0, atol=1e-8): 
    #     print("RHS is div free.")
    #     # 2. Check if pressure is provided and not None
    #     if pressure is not None:
    #         P = pressure
    #         print("HI", P)
            
    # return P


    # Create Laplacian operator
    L = Laplacian(Nx, Ny, dx, dy)
    
    # Reshape RHS to column vector
    rhs = RHS.reshape(Nx * Ny, 1, order='F')
    
    L = L.tocsr()  # Ensure CSR format for efficient slicing
    L[0, :] = 0    # Set first row to zero
    L[0, 0] = 1  # Set diagonal to 1
    rhs[0] = 0     # Set first RHS element to 0

    Pcorr = spsolve(L, rhs) 
    P = Pcorr.reshape((Nx, Ny), order='F')
    
    return P