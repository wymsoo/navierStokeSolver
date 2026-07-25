import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys; sys.path.append('../')
import scipy.io as sio
import itertools
import os
import argparse
import linecache
from scipy.io import loadmat
import itertools
from scipy.signal import savgol_filter

def count_files_scandir(directory):
    count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                count += 1
    return count


def gradient(u, dt, dy):
    ut = np.gradient(u, dt, axis=0, edge_order=2)
    uy = np.gradient(u, dy, axis=1, edge_order=2)
    uyy = np.gradient(uy, dy, axis=1, edge_order=2)
    return ut, uy, uyy

def poly_gradient(u,dt,dy):
    ut = np.gradient(u, dt, axis=1, edge_order=2)
    uy = savgol_filter(u, 5, 4, delta=dy, deriv=1, axis=0)
    uyy = savgol_filter(u, 5, 4, delta=dy, deriv=2, axis=0)
    return ut, uy, uyy

def build_dict(u, uy, u_yy):

    all_differentials = [u, uy, u_yy]
    all_labels = ['u','u_y','u_yy']

    dictionary = {}
    oper_dict = np.zeros((8,len(u[0])*len(u[1])))
    combinations = itertools.combinations(all_differentials, 2)
    label_combinations = itertools.combinations(all_labels, 2)

    oper_dict = [(a*b).reshape(-1) for a, b in combinations]

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
    dictionary['u**2'] = (u**2).reshape(-1)
    oper_dict.append(np.ones_like(u).reshape(-1))
    oper_dict.append((u**2).reshape(-1))

    dictionary_keys = []
    for key in dictionary:
        dictionary_keys.append(key)


    # oper_dict = np.transpose(np.array(oper_dict),(1,0,2,3))
    oper_dict = np.transpose(np.array(oper_dict))
    return oper_dict, dictionary_keys



def stridge1(X, Y, tol, lam=1e-10, max_iter = 100):
    d = X.shape[1]
    biginds = np.arange(d)  # initialise array for big indices
    biginds_prev = biginds

    #normalise theta
    norm = np.linalg.norm(X, axis=0)
    norm[norm<1e-16] = 1.0
    X_norm = X/norm

    # Calculate initial weights
    weights = np.linalg.lstsq(X_norm.T @ X_norm + lam * np.eye(d), X_norm.T @ Y, rcond=None)[0]

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

def trainStridge(X, y, tol_values, lam=1e-5, l0_penalty=1e-6, seed=0, print_best_tol=False):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    rand_indices = rng.permutation(n)
    X_train = X[rand_indices[:int(n*0.8)]]
    X_validate = X[rand_indices[int(n*0.8):]]
    y_train = y[rand_indices[:int(n*0.8)]]
    y_validate = y[rand_indices[int(n*0.8):]]
    score_best = np.inf
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

    return w_best, tol_best, train_err_best, val_err_best


def format_equation(coefficients, descriptions, lhs="u_t", coef_cutoff=1e-9):
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

import matplotlib.pyplot as plt

if __name__ == "__main__":
    nu_true=0.01/np.pi #viscosity/rho

    # Load and extract data
    mat_data = loadmat('stridge_model/burgers_shock.mat')
    U = mat_data['usol'] #256,100
    y = mat_data['x'] #256, 1
    t = mat_data['t'] #100, 1
    dt = t[31,0] - t[30,0]
    dy = y[31,0] - y[30,0]

    plt.plot(y,U[:,-1])
    plt.show()

    ut, uy, uyy = poly_gradient(U, dt, dy)
    t_end = 100
    y_end = len(y)

    u_crop = U[:y_end, :t_end]
    ut_crop = ut[:y_end, :t_end]
    uy_crop = uy[:y_end, :t_end]
    uyy_crop = uyy[:y_end, :t_end]
    tol_min = 2.5e-2
    tol_max = 5.0
    num_tol = 50

    oper_dict, dictionary = build_dict(u_crop,uy_crop,uyy_crop)

    x = oper_dict[:]
    y = ut_crop.reshape(-1)

    tol_values = np.logspace(np.log10(tol_min), np.log10(tol_max), num_tol)
    w_best, tol_best, err_best, val_error = trainStridge(x,y,tol_values)


    print(dictionary)
    print("=" * 72)
    print("Discovered equation")
    print(format_equation(w_best, dictionary, lhs="u_t"))
    print("=" * 72)
    print(f"Best tolerance      : {tol_best:.8e}")
    print(f"Train RMSE          : {err_best:.8e}")
    print(f"Validation RMSE     : {val_error:.8e}")
    # print(f"True equation       : u_t = {g_true:.8e} + {nu_true:.8e}*u_yy")
    print(f"True equation       : u_t = -u*u_y")
    print("Estimated coefficients")
    for c, name in zip(w_best, dictionary):
        print(f"  {name:8s} : {c:+.8e}")
