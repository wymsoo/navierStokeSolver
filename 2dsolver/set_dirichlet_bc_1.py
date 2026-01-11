import numpy as np

def set_Dirichlet_BC(U, V):

    Ly = 0.001
    Ny= 11
    dt = 0.01
    rho=900
    g = 9.81


    # Apply front and back boundaries for U
    lenU = U.shape
    # Unorth = 2*np.ones((lenU[0], 1)) - U[:, -1].reshape((lenU[0],1)) 
    # Unorth = -U[:, :, -1].reshape((lenU[0],lenU[1],1))  
    Unorth = -U[:,-1].reshape((lenU[0],1))  # for poiseuille flow, there is no slip in the top and bottom boundary
    Usouth = -U[:,0].reshape((lenU[0],1))
    U = np.concatenate((Usouth, U), axis=1)
    U = np.concatenate((U, Unorth), axis=1)

    midU = Ny//2
    lenU = U.shape
    # Ueast = np.zeros((1, lenU[1])) # set the last row of velocity field to be north boundary
    # Uwest = np.zeros((1, lenU[1])) # set the first row of velocity field to be south boundary
    Uwest = np.zeros((1, lenU[1])) #periodic boundary conditions
    Ueast = U[-1,:].reshape((1, lenU[1])) #neumann boundary condition: boundary = previous last cell
    U = np.concatenate((Uwest, U), axis=0)
    U = np.concatenate((U, Ueast), axis=0) #u_1/2 = c1


    U[:,midU+1] = U[:,midU]
    U[:,midU-1] = U[:,midU] # dUdx at middle = 0
    
    # Same thing for V
    lenV = V.shape
    
    # Apply north and south boundaries for V
    # Vnorth = 2*np.ones((lenV[0], lenV[1], 1)) - V[:, :, -1].reshape((lenV[0],lenV[1],1)) 
    Vnorth= np.zeros((lenV[0], 1))
    Vsouth= np.zeros((lenV[0], 1))
    V = np.concatenate((Vsouth, V), axis=1)
    V = np.concatenate((V, Vnorth), axis=1)
    
    lenV = V.shape
    # Apply front and back boundaries for V
    Veast = -V[-1,:].reshape((1, lenV[1])) #boundary at the end of V-field
    Vwest = -V[0,:].reshape((1, lenV[1]))
    V = np.concatenate((Vwest, V), axis=0)
    V = np.concatenate((V, Veast), axis=0)

    return U,V

    # print(U[-1,:])
    # print("U1:",U[:,-1])
    # print("U2:",U[:,0])
    # print("V1",V[:,-1])
    # print("V2",V[:,0])
    # print("U11:",U[-1,:])
    # print("U21:",U[0,:])
    # print("V12",V[-1,:])
    # print("V22",V[0,:])
