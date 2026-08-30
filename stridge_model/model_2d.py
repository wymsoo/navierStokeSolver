import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from global_var import dt, Nx, Ny, dx, dy, Re, rho, G, D, max_ts, H, L, viscosity, epsilon
import sys; sys.path.append('../')
import scipy.io as sio
import itertools
import os
import argparse
import linecache


def count_files_scandir(directory):
    count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                count += 1
    return count


def gradient(u, dt, dy, axis=1):
    ut = np.gradient(u, dt, axis=0, edge_order=2)
    uy = np.gradient(u, dy, axis=axis, edge_order=2)
    uyy = np.gradient(uy, dy, axis=axis, edge_order=2)
    
    return ut, uy, uyy

# Construct Operator Dictionary
import itertools

def build_dict(u, uy, u_yy, ux, u_xx):

    all_differentials = [u, uy, u_yy, ux[:,:-1,:], u_xx[:,:-1,:]] # ensure shape consistency
    all_labels = ['u','u_y','u_yy', 'u_x','u_xx']
    for i, val in enumerate(all_differentials):
        print(all_labels[i])
        print(val.shape)

    # Dictionary: for display
    dictionary = {}
    combinations = itertools.combinations(all_differentials, 2)
    label_combinations = itertools.combinations(all_labels, 2)

    # compute all product of combinations
    oper_dict = [(a*b).reshape(-1) for a, b in combinations]


    # Use string keys for combined labels (e.g. 'u*u_y') instead of tuples
    for label, oper in zip(label_combinations, oper_dict):
        if isinstance(label, tuple):
            key = f"{label[0]}*{label[1]}"
        else:
            key = label
        dictionary[key] = oper

    # add back the single terms
    for diff, label in zip(all_differentials, all_labels):
        oper_dict.append(diff.reshape(-1))
        dictionary[label] = diff

    dictionary['1'] = np.ones_like(u)
    # dictionary['u**2'] = (u**2).reshape(-1)
    oper_dict.append(np.ones_like(u).reshape(-1))
    # oper_dict.append((u**2).reshape(-1))

    dictionary_keys = []
    for key in dictionary:
        dictionary_keys.append(key)

    # for row in range(len(oper_dict)):
    #     print(len(oper_dict[row]))
    # oper_dict = np.transpose(np.array(oper_dict),(1,0,2,3))
    oper_dict = np.transpose(np.array(oper_dict))
    print(oper_dict.shape)
    return oper_dict, dictionary_keys



def stridge1(X, Y, tol, lam=10**-8, max_iter = 100):
    d = X.shape[1]
    biginds = np.arange(d)  # initialise array for big indices
    biginds_prev = biginds

    #normalise theta
    norm = np.linalg.norm(X, axis=0)
    norm[norm<1e-16] = 1.0
    X_norm = X/norm

    # Calculate initial weights
    weights = np.linalg.lstsq(X_norm.T @ X_norm + lam * np.eye(d), X_norm.T @ Y, rcond=None)[0]
    # print("weights:",weights)

    if biginds.size == 0:
        return weights

    for iter in range(max_iter):
        biginds = np.where(abs(weights) >= tol)[0]
        # If the weights do not change, break the loop
        if np.array_equal(biginds, biginds_prev):
            break
        if biginds.size==0:
            weights[:]=0
            break
        biginds_prev = biginds.copy()
        weights[:] = 0
        weights[biginds] = np.linalg.lstsq(
            X_norm[:, biginds].T.dot(X[:, biginds]) + lam * np.eye(len(biginds)),
            X_norm[:, biginds].T.dot(Y),
            rcond=None
        )[0]

    biginds = np.where(abs(weights)>= tol)[0]
    if biginds.size > 0:
        weights[:] = 0.0
        weights[biginds] = np.linalg.lstsq(X_norm[:, biginds], Y, rcond=None)[0]
    else:
        weights[:] = 0.0

    return np.array(weights/norm)

def trainStridge(X, y, tol_values, lam=1e-8, l0_penalty=1e-6, seed=0, print_best_tol=False):
    rng = np.random.default_rng(seed)
    n = X.shape[0] # random sample points in the field
    rand_indices = rng.permutation(n)
    X_train = X[rand_indices[:int(n*0.8)]]
    X_validate = X[rand_indices[int(n*0.8):]]
    y_train = y[rand_indices[:int(n*0.8)]]
    y_validate = y[rand_indices[int(n*0.8):]]
    score_best = np.inf
    print(X.shape, y.shape)
    w_best = np.linalg.lstsq(X, y, rcond=None)[0]
    train_err_best = np.linalg.norm(y_train - X_train @ w_best, 2) + l0_penalty * np.count_nonzero(w_best)
    val_err_best = np.linalg.norm(y_validate - X_validate @ w_best, 2) + l0_penalty * np.count_nonzero(w_best)
    tol_best = 0

    for tolerance in tol_values:
        weights = stridge1(X_train, y_train, tolerance)
        train_err = np.linalg.norm(y_train - X_train @ weights, 2) / np.sqrt(y_train.size)
        val_err = np.linalg.norm(y_validate - X_validate @ weights, 2)/np.sqrt(y_validate.size)
        score = val_err + l0_penalty * np.count_nonzero(np.abs(weights) > 1e-12)
        if score <= score_best:
            # Improved: keep direction
            # print("error improved", train_err)
            score_best = score
            w_best = weights
            train_err_best = train_err
            val_err_best = val_err
            tol_best = tolerance
        if val_err <= 10**-8:
            break


    # if print_best_tol:
        # print("Optimal tolerance:", tol_best)

    return w_best, tol_best, train_err_best, val_err_best


def format_equation(coefficients, descriptions, lhs="u_t", coef_cutoff=1e-10):
    # save active coeficcients as tuples in an array
    active = []
    for c, name in zip(coefficients, descriptions):
        if abs(c) > coef_cutoff:
            active.append((c, name))
    if not active:
        return f"{lhs} = 0"

    pieces = []
    for i, (c, name) in enumerate(active):
        sign = "-" if c < 0 else "+"
        mag = abs(c)
        if name == "1":
            term = f"{mag:.8e}*one"
        else:
            term = f"{mag:.8e}*{name}"
        if i == 0:
            pieces.append(term if c >= 0 else f"- {term}")
        else:
            pieces.append(f" {sign} {term}")
    return f"{lhs} =" + "".join(pieces)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Learn the startup Poiseuille PDE from saved data.")
    parser.add_argument("--input", type=str, default=os.path.join(os.path.dirname(__file__), "startup_poiseuille_data.npz"))
    parser.add_argument("--lam", type=float, default=1e-8)
    parser.add_argument("--tol_min", type=float, default=1e-8)
    parser.add_argument("--tol_max", type=float, default=1e-0)
    parser.add_argument("--num_tol", type=int, default=40)
    parser.add_argument("--l0_penalty", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crop_t", type=int, default=1, help="discard this many time layers near each temporal boundary")
    parser.add_argument("--crop_y", type=int, default=2, help="discard this many spatial points near each wall")
    parser.add_argument("--crop_x", type=int, default=1, help="discard this many spatial points near each wall")
    args = parser.parse_args()


    data = np.load(args.input)
    U = data["u"]
    # t = data["t"]
    # y = data["y"]
    # dt = float(data["dt"])
    # dy = float(data["dy"])
    # nu_true = float(data["nu"])
    # g_true = float(data["g"])
    nu_true = 10.0/1000
    g_true = 9.81

    file_path = "u_velocity_field"

    time_len = count_files_scandir(f'{file_path}')
    results = []
    # line_to_read = 15 # mid line of the file
    print(time_len)
    for i in range(150,time_len+1):
        t_res = []
        filepath = f'u_velocity_field/u_velocity_t={i}.txt'
        with open(filepath, 'r') as file:
            for l in range(1,31):
                line = linecache.getline(filepath, l).strip()  # reading one row of the 31-vector field
                if not line:
                    continue
                str_list = line.split()  # split on any whitespace, including tabs and repeated spaces
                floats = [float(value) for value in str_list]
                t_res.append(floats)
            results.append(t_res)

    # U is an array of (t,31) 
    U = np.array(results) # (t, x, y)
    print(U.shape)



    ut, uy, uyy = gradient(U, dt, dy, axis=2)
    ut, ux, uxx = gradient(U, dt, dx, axis=1)
    t_slice = slice(args.crop_t, U.shape[0] - args.crop_t)
    x_slice = slice(args.crop_x, U.shape[1] - args.crop_x)
    y_slice = slice(args.crop_y, U.shape[2] - args.crop_y)

    u_crop = U[t_slice, y_slice]
    ut_crop = ut[t_slice, y_slice]
    uy_crop = uy[t_slice, y_slice]
    uyy_crop = uyy[t_slice, y_slice]
    ux_crop = ux[t_slice, x_slice]
    uxx_crop = uxx[t_slice, x_slice]

    oper_dict, dictionary = build_dict(u_crop,uy_crop,uyy_crop,ux_crop,uxx_crop)

    x = oper_dict[:] # (1963416, 16)
    y = ut_crop.reshape(-1)

    tol_values = np.logspace(np.log10(args.tol_min), np.log10(args.tol_max), args.num_tol)
    w_best, tol_best, err_best, val_error = trainStridge(x,y,tol_values)

    print(dictionary)

    print("=" * 72)
    print("Discovered equation")
    print(format_equation(w_best, dictionary, lhs="u_t"))
    print("=" * 72)
    print(f"Best tolerance      : {tol_best:.8e}")
    print(f"Train RMSE          : {err_best:.8e}")
    print(f"Validation RMSE     : {val_error:.8e}")
    print(f"True equation       : u_t = {g_true:.8e} + {nu_true:.8e}*u_yy")
    print("Estimated coefficients")
    for c, name in zip(w_best, dictionary):
        print(f"  {name:8s} : {c:+.8e}")



