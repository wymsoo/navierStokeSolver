import numpy as np 
import global_var
from average import average

# def advective(U: np.ndarray, V: np.ndarray, W: np.ndarray,
#               dx: float, dy: float, dz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
def advective (U: np.ndarray, V: np.ndarray, dx: float, dy: float ): 

    # Calculate squared terms
    Uavgx = average(U, axis=0)
    Usquared = Uavgx**2

    Vavgy = average(V, axis=1)
    Vsquared = Vavgy**2
    
    # Wavgz = average(W, axis=2)
    # Wsquared = Wavgz**2
    
    # Mixed terms
    Uavgy = average(U, axis=1)
    Vavgx = average(V, axis=0)
    UV = Uavgy * Vavgx
    
    # Uavgz = average(U, axis=2)
    # Wavgx = average(W, axis=0)
    # UW = Uavgz * Wavgx
    
    # Vavgz = average(V, axis=2)
    # Wavgy = average(W, axis=1)
    # VW = Vavgz * Wavgy
    
    # Derivatives for U component
    USQRDx = np.diff(Usquared, axis=0) / dx #udu/dx
    UVy = np.diff(UV, axis=1) / dy 
    # UWz = np.diff(UW, axis=2) / dz
    
    advecU = (USQRDx[:, 1:-1] + UVy[1:-1, :])
    
    # Derivatives for V component
    VSQRDy = np.diff(Vsquared, axis=1) / dy
    UVx = np.diff(UV, axis=0) / dx
    # VWz = np.diff(VW, axis=2) / dz
    
    advecV = (VSQRDy[1:-1, :] + UVx[:, 1:-1])
    
    # Derivatives for W component
    # WSQRDz = np.diff(Wsquared, axis=2) / dz
    # UWx = np.diff(UW, axis=0) / dx
    # VWy = np.diff(VW, axis=1) / dy
    
    # advecW = (WSQRDz[1:-1, 1:-1, :] + 
    #            UWx[:, 1:-1, 1:-1] + 
    #            VWy[1:-1, :, 1:-1])
    
    return advecU, advecV
