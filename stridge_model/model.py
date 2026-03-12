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
# U = np.zeros((time_len,Nx-1,Ny)) #(time,x-1,y)
U = np.zeros((1,Ny)) #(time,x-1,y)
# V = np.zeros((time_len,Nx,Ny-1)) #(time,x,y-1)
V = np.zeros((1,Ny-1)) #(time,x,y-1)
P = np.zeros((1,Ny)) #(time,x,y)

# for i in range(time_len):
#     U[i] = np.loadtxt(f"output/u_velocity_field/u_velocity_t={starttime}.txt")
#     V[i] = np.loadtxt(f"output/v_velocity_field/v_velocity_t={starttime}.txt")
#     P[i] = np.loadtxt(f"output/pressure_field/pressure_field_t={starttime}.txt")
#     starttime+=time_res

U_arr = np.loadtxt(f"output/u_velocity_field/u_velocity_t={time_len*starttime}.txt")
V_arr = np.loadtxt(f"output/v_velocity_field/v_velocity_t={time_len*starttime}.txt")
P_arr = np.loadtxt(f"output/pressure_field/pressure_field_t={time_len*starttime}.txt")
U = U_arr[Nx//2, :]
V = V_arr[Nx//2, :]
P = P_arr[Nx//2, :]


def gradient(u, axis):
    diff_u = np.gradient(u,axis=axis)
    return diff_u

# Construct Operator Dictionary
import itertools

deg = 2
dict_size = np.zeros((Nx, 36)) # 36 is counted from the number of combinations, + 8 of the single objects

# ground truth dU/dt or dV/dt
# u_t = gradient(U,0)[:,:,:-1] # ensure size consistency
# print(u_t.shape)
# v_t = gradient(V, 0)[:,:-1,:] # (51, 30, 30) # ensure size consistency
# print("Shape of du/dt:", u_t.shape) #(51,30,30)

# u, v, p
# u_x = gradient(U,1)
# u_xx = gradient(u_x, 1)
# p_x = gradient(P,1)
# p_xx = gradient(p_x, 1)
# u_y = gradient(U,2)
# u_yy = gradient(u_y, 2)
u_y = np.gradient(U)
u_yy = np.gradient(u_y)
v_y = np.gradient(V)
v_yy = np.gradient(v_y)
p_y = np.gradient(P)
p_yy = np.gradient(p_y)

# v_x = gradient(V,1)
# v_xx = gradient(v_x, 1)
# v_y = gradient(V,2)
# v_yy = gradient(v_y, 1)
# p_y = gradient(P,2)
# p_yy = gradient(p_y, 2)

# print(U.shape, V.shape, u_x.shape,u_xx.shape,v_x.shape, v_xx.shape, p_x.shape, p_xx.shape, u_y.shape, u_yy.shape, v_y.shape, v_yy.shape, p_y.shape, p_yy.shape)
# all_differentials = [U[:,:,:-1], V[:,:-1,:], u_x[:,:,:-1],u_xx[:,:,:-1], p_x[:,:-1,:-1], p_xx[:,:-1,:-1], u_y[:,:,:-1], u_yy[:,:,:-1]]
# all_labels = ['u','v','u_x','u_xx','p_x','p_xx','u_y','u_yy']
all_differentials = [U[:-1], V[:], p_y[:-1], p_yy[:-1], u_y[:-1], u_yy[:-1]]
all_labels = ['u','v','p_y','p_yy','u_y','u_yy']

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

# oper_dict = np.transpose(np.array(oper_dict),(1,0,2,3))
oper_dict = np.transpose(np.array(oper_dict))
print(oper_dict.shape)


# dimensions (length of dictionary)
# d = 36
d= 21
# Print shape information
print("Original operdict shape:", oper_dict.shape)

# Min-max normalization for the whole array
min_val = np.min(oper_dict)
max_val = np.max(oper_dict)
oper_dict = (oper_dict - min_val) / (max_val - min_val)

# Reshaping to ensure compatibility
# y = u_t.reshape(51 * (Nx-1) * (Nx-1))

# flatten arrays
y = np.zeros(((Ny-1))) # du/dt = 0 due to steady flow, add small value to avoid trivial solution
x = oper_dict.reshape((Ny-1), d)



init_weights = np.linalg.lstsq(x.T.dot(x) + 10E-5 * np.eye(d),x.T.dot(y),rcond=None)[0]
print(x.shape)
print(y.shape)
print(init_weights.shape)




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


# stridge1(x, y)
# This stridge removes one element at a time

def stridge2(X, Y, penalty=10**-5, tol=5.0):
    # Define dimensions
    d = X.shape[1]  # Number of features
    y = Y  # Rename for clarity

    # Initialize
    count = 0
    smallinds = []  # List to store indices to set to zero

    # Calculate initial weights
    weights = np.linalg.lstsq(X.T.dot(X) + penalty*np.eye(d), X.T.dot(y), rcond=None)[0]

    # Find initial smallest index
    first_small = np.argmin(np.abs(weights))
    smallinds.append(first_small)
    print("Smallinds:", smallinds)

    # Get big indices (all except the smallest)
    biginds = np.where(np.abs(weights) != np.abs(weights[first_small]))[0]
    print("Biginds:", biginds)

    # Initialize previous weights for comparison
    prev_weights = weights.copy()

    # Main loop - remove one coefficient at a time
    while len(biginds) >= 2 and count <= 1000:
        # Set the small indices to zero
        for idx in smallinds:
            weights[idx] = 0

        print("Weights after zeroing:", weights)

        # Recalculate weights for big indices only
        if len(biginds) > 0:
            X_big = X[:, biginds]
            weights[biginds] = np.linalg.lstsq(
                X_big.T.dot(X_big) + penalty * np.eye(len(biginds)),
                X_big.T.dot(y),
                rcond=None
            )[0]

        # Find the next smallest index among the current big indices
        if len(biginds) > 0:
            # Get absolute weights of big indices and find smallest
            smallest_in_big = np.argmin(np.abs(weights[biginds]))
            # Get the actual index in original array
            next_small = biginds[smallest_in_big]

            # Add to smallinds list
            smallinds.append(next_small)

            # Update biginds - all indices not in smallinds
            biginds = np.array([i for i in range(d) if i not in smallinds])

            print(f"Iteration {count}: biginds={biginds}, smallinds={smallinds}")

        # Check for convergence
        if np.array_equal(prev_weights, weights):
            print("Weights stopped changing - possible convergence")
            break

        prev_weights = weights.copy()
        count += 1

    remaininginds = np.where(weights != 0)[0]
    for i in remaininginds:
        print("Remaining Terms:")
        operator = list(dictionary.keys())[i]
        print(operator)

    print(f"Final: {len(biginds)} big indices, {len(smallinds)} zeroed indices")
    # return weights, smallinds, biginds
    tolerance = 0
    return weights, tolerance



def trainStridge(X, y, tol=3.0, penalty=1e-5, print_best_tol=False):
    d_tol = 3.0
    max_iter = 50

    # initial full ridge
    w_best = np.linalg.lstsq(X, y, rcond=None)[0]
    print(w_best)
    err_best = np.linalg.norm(y - X @ w_best, 2) + penalty * np.count_nonzero(w_best)
    tol_best = 0

    for iter in range(max_iter):
        print(f"iter {iter}, tol={tol}, d_tol={d_tol}")
        # calculate weights with new tolerance
        print("TOL:", tol)
        weights, tolerance = stridge2(X, y, penalty=penalty, tol=tol)
        nnz = np.count_nonzero(weights)
        err = np.linalg.norm(y - X @ weights, 2) + penalty * nnz
        print(f"  flag={tolerance}, nnz={nnz}, err={err}")

        if err <= err_best:
            # Improved: keep direction
            print("error improved", err)
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
        if err_best <= 10**-8:
            break




    if print_best_tol:
        print("Optimal tolerance:", tol_best)

    return w_best

