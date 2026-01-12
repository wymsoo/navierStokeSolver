import numpy as np

def set_Dirichlet_BC(U, V):

    lenU = U.shape
 
    # Apply north and south boundaries for U
    lenU = U.shape
    Unorth = 2*np.ones((lenU[0], 1)) - U[:,-1].reshape((lenU[0],1))  # set the last row of velocity field to be north boundary
    Usouth = -U[:, 0].reshape((lenU[0],1)) # set the first row of velocity field to be south boundary
    U = np.concatenate((Usouth, U), axis=1)
    U = np.concatenate((U, Unorth), axis=1) #u_1/2 = c1

    
    # Apply front and back boundaries for U
    lenU = U.shape
    Ueast = np.zeros((1, lenU[1]))
    Uwest = np.zeros((1, lenU[1]))
    U = np.concatenate((Ueast, U), axis=0)
    U = np.concatenate((U, Uwest), axis=0)

    
    # Same thing for V
    lenV = V.shape
    
    # Apply north and south boundaries for V
    # Vnorth = 2*np.ones((lenV[0], lenV[1], 1)) - V[:, :, -1].reshape((lenV[0],lenV[1],1)) 
    Vnorth= -V[:, -1].reshape((lenV[0],1))
    Vsouth= -V[:, 0].reshape((lenV[0],1))
    V = np.concatenate((Vsouth, V), axis=1)
    V = np.concatenate((V, Vnorth), axis=1)
    
    lenV = V.shape
    # Apply front and back boundaries for V
    Veast = -V[-1, :].reshape((1, lenV[1]))
    Vwest = -V[0, :].reshape((1, lenV[1]))
    V = np.concatenate((Vwest, V), axis=0)
    V = np.concatenate((V, Veast), axis=0)

    
    return U, V
