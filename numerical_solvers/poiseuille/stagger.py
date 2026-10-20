import numpy as np 
from average import average

def stagger_back(U: np.ndarray, V: np.ndarray):

    # For U component
    lenU = U.shape
    Ueast = np.zeros((1, lenU[1]))
    Uwest = np.zeros((1, lenU[1]))
    U = np.concatenate([Uwest, U, Ueast], axis=0)
    U = average(U, axis=0)
    
    # For V component
    lenV = V.shape
    Vnorth = np.zeros((lenV[0], 1))
    Vsouth = np.zeros((lenV[0], 1))
    V = np.concatenate([Vsouth, V, Vnorth], axis=1)
    V = average(V, axis=1)
    
    return U, V
