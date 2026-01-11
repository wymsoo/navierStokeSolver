import numpy as np 

# def viscous(U: np.ndarray, V: np.ndarray, W: np.ndarray, 
#             Re: float, dx: float, dy: float, dz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

def viscous(U: np.ndarray, V: np.ndarray, W: np.ndarray, 
            Re: float, dx: float, dy: float, dz: float):
    # For U component
    # Uxx: second derivative in x-direction (axis 0)
    Uxx = np.diff(U[:, 1:-1, 1:-1], n=2, axis=0) / dx**2
    
    # Uyy: second derivative in y-direction (axis 1)
    Uyy = np.diff(U[1:-1, :, 1:-1], n=2, axis=1) / dy**2
    
    # Uzz: second derivative in z-direction (axis 2)
    Uzz = np.diff(U[1:-1, 1:-1, :], n=2, axis=2) / dz**2

    viscousx = (1/Re) * (Uxx + Uyy + Uzz)
    
    # For V component
    Vxx = np.diff(V[:, 1:-1, 1:-1], n=2, axis=0) / dx**2
    Vyy = np.diff(V[1:-1, :, 1:-1], n=2, axis=1) / dy**2
    Vzz = np.diff(V[1:-1, 1:-1, :], n=2, axis=2) / dz**2
    
    viscousy = (1/Re) * (Vxx + Vyy + Vzz)
    
    # For W component
    Wxx = np.diff(W[:, 1:-1, 1:-1], n=2, axis=0) / dx**2
    Wyy = np.diff(W[1:-1, :, 1:-1], n=2, axis=1) / dy**2
    Wzz = np.diff(W[1:-1, 1:-1, :], n=2, axis=2) / dz**2

    
    viscousz = (1/Re) * (Wxx + Wyy + Wzz)
    
    return viscousx, viscousy, viscousz
