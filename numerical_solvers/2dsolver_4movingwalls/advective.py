import numpy as np 
from average import average

# def advective(U: np.ndarray, V: np.ndarray, W: np.ndarray,
#               dx: float, dy: float, dz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
def advective(U: np.ndarray, V: np.ndarray,
              dx: float, dy: float):

    # Calculate squared terms
    Uavgx = average(U, axis=0)
    Usquared = Uavgx**2

    Vavgy = average(V, axis=1)
    Vsquared = Vavgy**2
    
    # Mixed terms
    Uavgy = average(U, axis=1)
    Vavgx = average(V, axis=0)
    UV = Uavgy * Vavgx
    
    
    # Derivatives for U component
    USQRDx = np.diff(Usquared, axis=0) / dx
    UVy = np.diff(UV, axis=1) / dy
    
    advecU = (USQRDx[:, 1:-1] + 
               UVy[1:-1, :])
    
    # Derivatives for V component
    VSQRDy = np.diff(Vsquared, axis=1) / dy
    UVx = np.diff(UV, axis=0) / dx
    
    advecV = (VSQRDy[1:-1, :] + 
               UVx[:, 1:-1])
    
    
    return advecU, advecV
