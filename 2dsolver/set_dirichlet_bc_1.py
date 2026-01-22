import numpy as np
from global_var import Nx, Ny, Re, D, G ,dx, dy, rho, dt, viscosity

def set_Dirichlet_BC(U, V):

    lenU = U.shape
    lenV = V.shape

    lenU = U.shape
    Uwest = U[0,:].reshape((1,lenU[1]))+0.1 #velocity due to gravity
    Ueast = Uwest                              #periodic boundary conditions 
    U = np.concatenate((Uwest, U), axis=0)
    U = np.concatenate((U, Ueast), axis=0)

    lenU = U.shape
    Unorth = - U[:,-1].reshape((lenU[0],1))  # for poiseuille flow, there is no slip in the top and bottom boundary
    Usouth = - U[:,0].reshape((lenU[0],1))
    U = np.concatenate((Usouth, U), axis=1)
    U = np.concatenate((U, Unorth), axis=1)

    lenV = V.shape
    Veast = -V[-1,:].reshape((1, lenV[1])) #no vertical velocity
    Vwest = -V[0,:].reshape((1, lenV[1]))
    V = np.concatenate((Vwest, V), axis=0)
    V = np.concatenate((V, Veast), axis=0)

    lenV = V.shape
    
    Vnorth= np.zeros((lenV[0], 1)) #no slip condition
    Vsouth= np.zeros((lenV[0], 1))
    V = np.concatenate((Vsouth, V), axis=1)
    V = np.concatenate((V, Vnorth), axis=1)
    

    return U,V
