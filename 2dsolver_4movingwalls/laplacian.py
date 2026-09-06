import numpy as np 
from scipy.sparse import diags, eye, kron, csr_matrix

def Laplacian(nx: int, ny: int,
              dx: float, dy: float) -> csr_matrix:

# Create 1D Laplacian operator in x-direction
    Ix = np.ones(nx)
    Dxx = diags([Ix, -2*Ix, Ix], [-1, 0, 1], shape=(nx, nx), format='csr')
    Dxx[0, 0] = -1
    Dxx[0, 1] = 1
    Dxx[-1, -2] = 1
    Dxx[-1, -1] = -1
    Dxx = Dxx / (dx**2)
    
    Iy = np.ones(ny)
    Dyy = diags([Iy, -2*Iy, Iy], [-1, 0, 1], shape=(ny, ny), format='csr')
    Dyy[0, 0] = -1
    Dyy[0, 1] = 1
    Dyy[-1, -2] = 1
    Dyy[-1, -1] = -1
    Dyy = Dyy / (dy**2)
    
    # Create identity matrices
    Ix_sparse = eye(nx, format='csr')
    Iy_sparse = eye(ny, format='csr')

    # L = Dxx ⊗ Iy ⊗ Iz + Ix ⊗ Dyy ⊗ Iz + Ix ⊗ Iy ⊗ Dzz
    term1 = kron(Dxx, Iy_sparse)
    term2 = kron(Ix_sparse, Dyy)
    
    L = term1 + term2 
    
    return L
