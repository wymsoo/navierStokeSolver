import numpy as np 
from scipy.sparse import diags, eye, kron, csr_matrix

def _second_derivative(n, spacing, periodic):
    values = np.ones(n)
    derivative = diags(
        [values, -2 * values, values],
        [-1, 0, 1],
        shape=(n, n),
        format='lil',
    )

    if periodic:
        derivative[0, -1] = 1
        derivative[-1, 0] = 1
    else:
        derivative[0, 0] = -1
        derivative[0, 1] = 1
        derivative[-1, -2] = 1
        derivative[-1, -1] = -1

    return derivative.tocsr() / spacing**2


def Laplacian(nx: int, ny: int,
              dx: float, dy: float,
              periodic_x: bool = False,
              periodic_y: bool = False) -> csr_matrix:

    Dxx = _second_derivative(nx, dx, periodic_x)
    Dyy = _second_derivative(ny, dy, periodic_y)
    
    # Create identity matrices
    Ix_sparse = eye(nx, format='csr')
    Iy_sparse = eye(ny, format='csr')

    # L = Dxx ⊗ Iy ⊗ Iz + Ix ⊗ Dyy ⊗ Iz + Ix ⊗ Iy ⊗ Dzz
    term1 = kron(Iy_sparse, Dxx)
    term2 = kron(Dyy, Ix_sparse)
    
    L = term1 + term2 
    
    return L
