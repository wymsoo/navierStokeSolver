import numpy as np

def set_Dirichlet_BC(U, V, W):

    lenU = U.shape

    # Apply east and west boundaries for U
    Ufront = np.zeros((1, lenU[1], lenU[2])) # Set the x-boundary to zero (no-slip condition)
    Uback = np.zeros((1, lenU[1], lenU[2])) 
    U = np.concatenate((Uback, U), axis=0) 
    U = np.concatenate((U, Ufront), axis=0)


    
    # Apply north and south boundaries for U
    lenU = U.shape
    Ueast = -U[:, -1, :].reshape((lenU[0],1,lenU[2])) # set the last row of velocity field to be north boundary
    Uwest = -U[:, 0, :].reshape((lenU[0],1,lenU[2])) # set the first row of velocity field to be south boundary
    U = np.concatenate((Uwest, U), axis=1)
    U = np.concatenate((U, Ueast), axis=1) #u_1/2 = c1

    
    # Apply front and back boundaries for U
    lenU = U.shape
    Unorth = 2*np.ones((lenU[0], lenU[1], 1)) - U[:, :, -1].reshape((lenU[0],lenU[1],1)) 
    # Unorth = -U[:, :, -1].reshape((lenU[0],lenU[1],1))  
    Usouth = -U[:, :, 0].reshape((lenU[0],lenU[1],1)) 
    U = np.concatenate((Usouth, U), axis=2)
    U = np.concatenate((U, Unorth), axis=2)

    
    # Same thing for V
    lenV = V.shape
    
    # Apply north and south boundaries for V
    # Vnorth = 2*np.ones((lenV[0], lenV[1], 1)) - V[:, :, -1].reshape((lenV[0],lenV[1],1)) 
    Vnorth= -V[:, :, -1].reshape((lenV[0],lenV[1],1))
    Vsouth= -V[:, :, 0].reshape((lenV[0],lenV[1],1))
    V = np.concatenate((Vsouth, V), axis=2)
    V = np.concatenate((V, Vnorth), axis=2)

    lenV = V.shape

    Vfront = -V[0, :, :].reshape((1,lenV[1],lenV[2]))
    Vback = -V[-1, :, :].reshape((1,lenV[1],lenV[2]))
    V = np.concatenate((Vback, V), axis=0)
    V = np.concatenate((V, Vfront), axis=0)
    
    lenV = V.shape
    # Apply front and back boundaries for V
    Veast = np.zeros((lenV[0], 1, lenV[2]))
    Vwest = np.zeros((lenV[0], 1, lenV[2]))
    V = np.concatenate((Vwest, V), axis=1)
    V = np.concatenate((V, Veast), axis=1)

    # And W
    lenW = W.shape
    
    # Apply front and back boundaries for W
    Weast = -W[:, -1, :].reshape((lenW[0],1,lenW[2]))
    Wwest = -W[:, 0, :].reshape((lenW[0],1,lenW[2]))
    W = np.concatenate((Wwest, W), axis=1)
    W = np.concatenate((W, Weast), axis=1)

    lenW = W.shape
    
    # Apply north and south boundaries for W
    Wnorth = np.zeros((lenW[0], lenW[1], 1))
    Wsouth = np.zeros((lenW[0], lenW[1], 1))
    W = np.concatenate((Wsouth, W), axis=2)
    W = np.concatenate((W, Wnorth), axis=2)

    lenW = W.shape
    
    # Apply east and west boundaries for W
    Wfront = -W[-1, :, :].reshape((1,lenW[1],lenW[2]))
    Wback = -W[0, :, :].reshape((1,lenW[1],lenW[2]))
    W = np.concatenate((Wback, W), axis=0)
    W = np.concatenate((W, Wfront), axis=0)
    
    return U, V, W
