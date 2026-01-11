import numpy as np 
from average import average

# def stagger_back(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
def stagger_back(U: np.ndarray, V: np.ndarray, W: np.ndarray):

    # For U component
    lenU = U.shape
    Ueast = np.zeros((1, lenU[1], lenU[2]))
    Uwest = np.zeros((1, lenU[1], lenU[2]))
    U = np.concatenate([Uwest, U, Ueast], axis=0)
    U = average(U, axis=0)
    
    # For V component
    lenV = V.shape
    Vnorth = np.zeros((lenV[0], 1, lenV[2]))
    Vsouth = np.zeros((lenV[0], 1, lenV[2]))
    V = np.concatenate([Vsouth, V, Vnorth], axis=1)
    V = average(V, axis=1)
    
    # For W component
    lenW = W.shape
    Wfront = np.zeros((lenW[0], lenW[1], 1))
    Wback = np.zeros((lenW[0], lenW[1], 1))
    W = np.concatenate([Wback, W, Wfront], axis=2)
    W = average(W, axis=2)
    
    return U, V, W
