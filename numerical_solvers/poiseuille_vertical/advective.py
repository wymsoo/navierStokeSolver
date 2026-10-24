import numpy as np 
from global_var import rho
from average import average


def advective (U: np.ndarray, V: np.ndarray, dx: float, dy: float ): 

    # Calculate squared terms
    lenU = U.shape
    #Uavgx = np.zeros((lenU[0]-1,lenU[1]))
    Uavgx = average(U, axis=0)
    Usquared = Uavgx**2

    lenV = V.shape
    Vavgy = average(V, axis=1)
    #Vavgy = np.zeros((lenV[0],lenV[1]-1))
    Vsquared = Vavgy**2

    # Mixed terms
    #Uavgy = np.zeros((lenU[0],lenU[1]-1))
    Uavgy = average(U, axis=1)
    #Vavgx = np.zeros((lenU[0]-1,lenU[1]))
    Vavgx = average(V, axis=0)
    UV = Uavgy * Vavgx

    # Derivatives for U component
    USQRDx = np.diff(Usquared, axis=0) / dx #udu/dx
    UVy = np.diff(UV, axis=1) / dy 
    
    advecU = (USQRDx[:, 1:-1] + UVy[1:-1, :])
    
    # Derivatives for V component
    VSQRDy = np.diff(Vsquared, axis=1) / dy
    UVx = np.diff(UV, axis=0) / dx
    
    advecV = (VSQRDy[1:-1, :] + UVx[:, 1:-1])

    return advecU, advecV
