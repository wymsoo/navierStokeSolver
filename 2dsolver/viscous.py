import numpy as np 
from global_var import rho

def viscous(U: np.ndarray, V: np.ndarray,
            Re: float, dx: float, dy: float, viscosity: float):

    # Uxx: second derivative in x-direction (axis 0)
    Uxx = np.diff(U[:, 1:-1], n=2, axis=0) / dx**2
    Uyy = np.diff(U[1:-1, :], n=2, axis=1) / dy**2
    
    viscousx =  (Uxx + Uyy)
    
    # For V component
    Vxx = np.diff(V[:, 1:-1], n=2, axis=0) / dx**2
    Vyy = np.diff(V[1:-1, :], n=2, axis=1) / dy**2

    
#     viscousy = (1/Re) * (Vxx + Vyy)
    viscousy =  (Vxx + Vyy)

    return viscousx, viscousy
