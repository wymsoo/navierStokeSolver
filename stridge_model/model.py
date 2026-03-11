import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys; sys.path.append('../')
import scipy.io as sio
import itertools
from global_var import Nx, Ny, Re, D, G ,dx, dy, dt, max_ts, H, L, rho, viscosity, epsilon, endtime, time_res
import os


def count_files_scandir(directory):
    count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                count += 1
    return count

time_len = count_files_scandir('output/u_velocity_field')

# coordinate system

x = np.arange(0,L,dx)
y = np.arange(-H/2,H/2,dy)
t = np.arange(0,endtime,dt*time_res)


starttime = 50
x_size = 11
y_size = 11
U = np.zeros((time_len,Nx-1,Ny)) #(time,x-1,y)
V = np.zeros((time_len,Nx,Ny-1)) #(time,x,y-1)
P = np.zeros((time_len,Nx,Ny)) #(time,x,y)

for i in range(time_len):
    U[i] = np.loadtxt(f"output/u_velocity_field/u_velocity_t={starttime}.txt")
    V[i] = np.loadtxt(f"output/v_velocity_field/v_velocity_t={starttime}.txt")
    P[i] = np.loadtxt(f"output/pressure_field/pressure_field_t={starttime}.txt")
    starttime+=time_res

def gradient(u, axis):
    diff_u = np.gradient(u,axis=axis)
    return diff_u

# Construct Operator Dictionary
import itertools

deg = 2
dict_size = np.zeros((Nx, 36)) # 36 is counted from the number of combinations, + 8 of the single objects

# ground truth dU/dt or dV/dt
u_t = gradient(U,0)[:,:,:-1] # ensure size consistency
print(u_t.shape)
v_t = gradient(V, 0)[:,:-1,:] # (51, 30, 30) # ensure size consistency
print("Shape of du/dt:", u_t.shape) #(51,30,30)


# u, v, p
u_x = gradient(U,1)
u_xx = gradient(u_x, 1)
p_x = gradient(P,1)
p_xx = gradient(p_x, 1)
u_y = gradient(U,2)
u_yy = gradient(u_y, 2)

v_x = gradient(V,1)
v_xx = gradient(v_x, 1)
v_y = gradient(V,2)
v_yy = gradient(v_y, 1)
p_y = gradient(P,2)
p_yy = gradient(p_y, 2)

# print(U.shape, V.shape, u_x.shape,u_xx.shape,v_x.shape, v_xx.shape, p_x.shape, p_xx.shape, u_y.shape, u_yy.shape, v_y.shape, v_yy.shape, p_y.shape, p_yy.shape)
all_differentials = [U[:,:,:-1], V[:,:-1,:], u_x[:,:,:-1],u_xx[:,:,:-1], p_x[:,:-1,:-1], p_xx[:,:-1,:-1], u_y[:,:,:-1], u_yy[:,:,:-1]]
all_labels = ['u','v','u_x','u_xx','p_x','p_xx','u_y','u_yy']

# for diff, label in zip(all_differentials, all_labels):
#   print(label,":",diff.shape)

# Dictionary: for display
dictionary = {}
combinations = itertools.combinations(all_differentials, 2)
label_combinations = itertools.combinations(all_labels, 2)

# compute all product of combinations
oper_dict = [ a*b for a, b in combinations]
# print(len(oper_dict))


for label, oper in zip(label_combinations, oper_dict):
    dictionary[label] = oper

# add back the single terms
for diff, label in zip(all_differentials, all_labels):
    oper_dict.append(diff)
    dictionary[label] = diff

for key in dictionary:
    print(key)

oper_dict = np.transpose(np.array(oper_dict),(1,0,2,3))
print(oper_dict.shape)


# dimensions (length of dictionary)
d = 36
# Print shape information
print("Original operdict shape:", oper_dict.shape)

# Min-max normalization for the whole array
min_val = np.min(oper_dict)
max_val = np.max(oper_dict)
oper_dict = (oper_dict - min_val) / (max_val - min_val)

# Reshaping to ensure compatibility
y = u_t.reshape(51 * (Nx-1) * (Nx-1))

# flatten arrays
#y = np.zeros((51*(Nx-1)*(Ny-1))) # du/dt = 0 due to steady flow, add small value to avoid trivial solution
x = oper_dict.reshape(51 * (Nx-1) * (Ny-1), d)

print(x.shape)
print(y.shape)
print(np.linalg.inv(x.T.dot(x) + 10**-8 * np.eye(d)))



def stridge1(X, Y, penalty=10**-5, tol=5.0):
    tolerance = 0
    smallinds = []
    count = 0  # initial max iterations count
    biginds = np.zeros((d,))  # initialise array for big indices

    # Calculate initial weights
    weights = np.linalg.lstsq(X.T.dot(X) + penalty * np.eye(d),X.T.dot(y),rcond=None)[0]
    # Get all the indices that are larger than tolerance
    smallinds = np.where(abs(weights) <= tol)[0]

    biginds = np.where(abs(weights) > tol)[0]
    print("Biginds", biginds)
    num_biginds = len(biginds)

    # Initialise old weights for comparison of new weights later
    prev_weights = weights.copy()

    # Loop and reduce coefficients to zero if below tolerance,
    # until it reaches a certain number of terms
    while (count <= 5):
        smallinds = np.where(abs(weights) <= tol)[0]

        # Check for big indices again
        biginds = np.where(abs(weights) > tol)[0]


        # If the weights do not change, break the loop
        if (num_biginds==len(biginds)):
            print("Tolerance too low. Weights: ", weights)
            tolerance = -1
            break

        if (len(biginds) == 0):
            if count==0 :
                print("Tolerance too high. Weights: ", weights)
                weights = prev_weights
                tolerance = 1
                break

        weights[smallinds] = 0

        # Recalculate weights after reducing small indices to zero
        weights[biginds] = np.linalg.lstsq(
            X[:, biginds].T.dot(X[:, biginds]) + penalty * np.eye(len(biginds)),
            X[:, biginds].T.dot(y),
            rcond=None
        )[0]

        prev_weights = weights.copy()
        num_biginds = len(biginds)
        count += 1

    remaininginds = np.where(weights != 0)[0]
    if (len(biginds)!=0):
        weights[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=None)[0]
        for i in remaininginds:
            operator = list(dictionary.keys())[i]
            print(operator)
    # print("Weights", weights)
    return weights, tolerance


# Call the function
stridge1(x, y)


def trainStridge(X, y, tol=3.0, penalty=1e-5, print_best_tol=False):
    d_tol = 3.0
    max_iter = 50

    # initial full ridge
    w_best = np.linalg.lstsq(X, y, rcond=None)[0]
    err_best = np.linalg.norm(y - X @ w_best, 2) + penalty * np.count_nonzero(w_best)
    tol_best = 0

    for iter in range(max_iter):
        print(f"iter {iter}, tol={tol}, d_tol={d_tol}")
        # calculate weights with new tolerance
        print("TOL:", tol)
        weights, tolerance = stridge1(X, y, penalty=penalty, tol=tol)
        nnz = np.count_nonzero(weights)
        err = np.linalg.norm(y - X @ weights, 2) + penalty * nnz
        print(f"  flag={tolerance}, nnz={nnz}, err={err}")

        if err <= err_best:
            # Improved: keep direction
            print("error improved")
            err_best = err
            w_best = weights
            tol_best = tol
            tol = tol + d_tol
        else:
            # No improvement: use tolerance flag to adapt tol and step size
            if tolerance == -1:  # tolerance too low (threshold too small) → increase tol
                d_tol *= 0.9 # steps get smaller after more iterations
                print(f"tol = {tol} too low, increase by", d_tol)
                tol = tol + d_tol
            elif tolerance == 1:  # tolerance too high (threshold too large) → decrease tol
                d_tol *= 0.9
                print(f"tol={tol} too high, decrease by", d_tol)
                tol = max(0.0, tol - d_tol)
            else:
                # tol = max([0,tol - 2*d_tol])
                # d_tol  = 2*d_tol / (max_iter - iter)
                print(f"normal, test if larger tolerance is possible, tol={tol}increase by ", d_tol)
                tol = tol + d_tol
        if err_best <= 10**-6:
            break




    if print_best_tol:
        print("Optimal tolerance:", tol_best)

    return w_best

trainStridge(x,y)