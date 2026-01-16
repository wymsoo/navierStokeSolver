import numpy as np
from global_var import Nx, Ny, Re, R, G ,dx, rho

def set_Dirichlet_BC(U, V):

    lenU = U.shape
    lenV = V.shape


    lenU = U.shape
    Uwest = np.ones((1,lenU[1]))*(Re*R**2/4*G*rho) #c2
    Ueast = Uwest #periodic boundary conditions
    U = np.concatenate((Uwest, U), axis=0)
    U = np.concatenate((U, Ueast), axis=0)

    # Apply front and back boundaries for U
    lenU = U.shape
    Unorth = - U[:,-1].reshape((lenU[0],1))  # for poiseuille flow, there is no slip in the top and bottom boundary
    Usouth = - U[:,0].reshape((lenU[0],1))
    U = np.concatenate((Usouth, U), axis=1)
    U = np.concatenate((U, Unorth), axis=1)

    lenV = V.shape
    # Apply front and back boundaries for V
    Veast = -V[-1,:].reshape((1, lenV[1])) #boundary at the end of V-field
    Vwest = -V[0,:].reshape((1, lenV[1]))
    V = np.concatenate((Vwest, V), axis=0)
    V = np.concatenate((V, Veast), axis=0)

    lenV = V.shape
    
    # Apply north and south boundaries for V
    Vnorth= np.zeros((lenV[0], 1))
    Vsouth= np.zeros((lenV[0], 1))
    V = np.concatenate((Vsouth, V), axis=1)
    V = np.concatenate((V, Vnorth), axis=1)
    

    return U,V
