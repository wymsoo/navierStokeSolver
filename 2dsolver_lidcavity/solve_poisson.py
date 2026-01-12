import numpy as np 
from scipy.sparse.linalg import spsolve
from laplacian import Laplacian
from divergence import div
from set_dirichlet_bc_1 import set_Dirichlet_BC


def Solve_Poisson(Ustar: np.ndarray, Vstar: np.ndarray,
                  dx: float, dy: float,
                  Nx: int, Ny: int, dt: float) -> np.ndarray:

    # Apply Dirichlet boundary conditions
    Ustarbc, Vstarbc = set_Dirichlet_BC(Ustar, Vstar)
    
    # Compute divergence 
    RHS = (1/dt) * div(Ustarbc, Vstarbc, dx, dy)
    
    # Create Laplacian operator
    L = Laplacian(Nx, Ny, dx, dy)
    
    # Reshape RHS to column vector
    rhs = RHS.reshape(Nx * Ny, 1, order='F')  # Use 'F' for Fortran-like column-major ordering
    
    L = L.tocsr()  # Ensure CSR format for efficient slicing
    L[0, :] = 0    # Set first row to zero
    L[0, 0] = 1    # Set diagonal to 1
    rhs[0] = 0     # Set first RHS element to 0
    
    Pcorr = spsolve(L, rhs) 
    P = Pcorr.reshape((Nx, Ny), order='F')
    
    return P