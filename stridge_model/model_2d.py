import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from global_var_2d import dt, Nx, Ny, dx, dy, Re, rho, G, D, max_ts, H, L, viscosity, epsilon
import sys; sys.path.append('../')
import scipy.io as sio
import itertools
import os
import argparse


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

def build_dict(u, v, uy, u_yy, ux, u_xx, p_x):
    all_differentials = [u, v, uy, u_yy, ux, u_xx, p_x]
    all_labels = ['u', 'v', 'u_y','u_yy', 'u_x','u_xx', 'p_x']
    
    for i, val in enumerate(all_differentials):
        print(f"{all_labels[i]}: {val.shape}")

    dictionary = {}
    combinations = itertools.combinations(all_differentials, 2)
    label_combinations = itertools.combinations(all_labels, 2)

    oper_dict = [(a*b).reshape(-1) for a, b in combinations]

    for label, oper in zip(label_combinations, oper_dict):
        if isinstance(label, tuple):
            key = f"{label[0]}*{label[1]}"
        else:
            key = label
        dictionary[key] = oper

    for diff, label in zip(all_differentials, all_labels):
        oper_dict.append(diff.reshape(-1))
        dictionary[label] = diff

    dictionary['1'] = np.ones_like(u)
    oper_dict.append(np.ones_like(u).reshape(-1))

    dictionary_keys = list(dictionary.keys())
    
    oper_dict = np.transpose(np.array(oper_dict))
    print("Dictionary shape:", oper_dict.shape)
    return oper_dict, dictionary_keys

def stridge1(X, Y, tol, lam=10**-4, max_iter=100):
    d = X.shape[1]
    biginds = np.arange(d) 
    biginds_prev = biginds
    n_samples = X.shape[0]
    Y_rms = np.linalg.norm(Y) / np.sqrt(n_samples)
    if Y_rms < 1e-16: Y_rms = 1.0
    Y_norm = Y / Y_rms
    norm = np.linalg.norm(X, axis=0)/np.sqrt(n_samples)
    norm[norm < 1e-16] = 1.0
    X_norm = X / norm

    # 1. Calculate initial weights on NORMALIZED features
    w_norm = np.linalg.lstsq(X_norm.T @ X_norm + lam * np.eye(d), X_norm.T @ Y_norm, rcond=None)[0]

    if biginds.size == 0:
        return w_norm / norm

    for iter in range(max_iter):
        # 2. Apply threshold to NORMALIZED weights!
        biginds = np.where(abs(w_norm) >= tol)[0]
        
        if np.array_equal(biginds, biginds_prev):
            break
        if biginds.size == 0:
            w_norm[:] = 0.0
            break
            
        biginds_prev = biginds.copy()
        w_norm[:] = 0.0
        
        # Recalculate weights for active indices using X_norm
        w_norm[biginds] = np.linalg.lstsq(
            X_norm[:, biginds].T.dot(X_norm[:, biginds]) + lam * np.eye(len(biginds)),
            X_norm[:, biginds].T.dot(Y_norm),
            rcond=None
        )[0]
    
    # Final un-regularized polish on active set
    biginds = np.where(abs(w_norm) >= tol)[0]
    if biginds.size > 0:
        w_norm[:] = 0.0
        w_norm[biginds] = np.linalg.lstsq(X_norm[:, biginds], Y_norm, rcond=None)[0]
    else:
        w_norm[:] = 0.0

    # 3. ONLY NOW do we un-normalize to return the true dimensional coefficients
    return w_norm / norm

def trainStridge(X, y, tol_values, lam=1e-8, l0_penalty=1e-2, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    rand_indices = rng.permutation(n)
    
    train_size = int(n * 0.8)
    X_train, X_validate = X[rand_indices[:train_size]], X[rand_indices[train_size:]]
    y_train, y_validate = y[rand_indices[:train_size]], y[rand_indices[train_size:]]
    
    score_best = np.inf
    w_best = np.linalg.lstsq(X, y, rcond=None)[0]
    train_err_best = np.linalg.norm(y_train - X_train @ w_best, 2) + l0_penalty * np.count_nonzero(w_best)
    val_err_best = np.linalg.norm(y_validate - X_validate @ w_best, 2) + l0_penalty * np.count_nonzero(w_best)
    tol_best = 0

    for tolerance in tol_values:
        weights = stridge1(X_train, y_train, tolerance)
        train_err = np.linalg.norm(y_train - X_train @ weights, 2) / np.sqrt(y_train.size)
        val_err = np.linalg.norm(y_validate - X_validate @ weights, 2) / np.sqrt(y_validate.size)
        score = val_err + l0_penalty * np.count_nonzero(np.abs(weights) > 1e-12)
        
        if score <= score_best:
            print("New High Score!")
            score_best = score
            w_best = weights
            train_err_best = train_err
            val_err_best = val_err
            tol_best = tolerance
        if val_err <= 10**-8:
            break

    return w_best, tol_best, train_err_best, val_err_best

def format_equation(coefficients, descriptions, lhs="u_t", coef_cutoff=1e-10):
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
        term = f"{mag:.8e}*one" if name == "1" else f"{mag:.8e}*{name}"
        if i == 0:
            pieces.append(term if c >= 0 else f"- {term}")
        else:
            pieces.append(f" {sign} {term}")
    return f"{lhs} =" + "".join(pieces)

def load_field_data(filepath_template, start_idx, end_idx):
    """Helper to cleanly load the sequential field data"""
    results = []
    for i in range(start_idx, end_idx + 1):
        filepath = filepath_template.format(i)
        t_res = []
        with open(filepath, 'r') as file:
            lines = file.readlines()
            for l in range(min(30, len(lines))):
                line = lines[l].strip()
                if not line:
                    continue
                floats = [float(value) for value in line.split()]
                t_res.append(floats)
        results.append(t_res)
    # Drop the last column for consistent shape as per original code
    return np.array(results)[:, :, :-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn the startup Poiseuille PDE from saved data.")
    parser.add_argument("--input", type=str, default=os.path.join(os.path.dirname(__file__), "startup_poiseuille_data.npz"))
    parser.add_argument("--lam", type=float, default=1e-8)
    parser.add_argument("--tol_min", type=float, default=1e-7)
    parser.add_argument("--tol_max", type=float, default=1)
    parser.add_argument("--num_tol", type=int, default=40)
    parser.add_argument("--l0_penalty", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crop_t", type=int, default=2, help="discard time layers near each temporal boundary")
    parser.add_argument("--crop_y", type=int, default=2, help="discard spatial points near each wall")
    parser.add_argument("--crop_x", type=int, default=2, help="discard spatial points near each wall")
    args = parser.parse_args()

    nu_true = 10.0 / 1000
    g_true = 9.81

    time_len = count_files_scandir('output_4movingwalls/u_velocity_field')
    
    # Load all arrays properly (Fixes BUG #2 & BUG #4)
    U = load_field_data('output_4movingwalls/u_velocity_field/u_velocity_t={}.txt', 150, time_len)
    V = load_field_data('output_4movingwalls/v_velocity_field/v_velocity_t={}.txt', 150, time_len)
    P = load_field_data('output_4movingwalls/pressure_field/pressure_field_t={}.txt', 150, time_len)

    print("U Shape:", U.shape)
    print("V Shape:", V.shape)
    print("P Shape:", P.shape)

    # Compute Gradients (Fixes BUG #3 - added dx to pressure gradient)
    ut, uy, uyy = gradient(U, dt, dy, axis=2)
    _, ux, uxx = gradient(U, dt, dx, axis=1)  # No need to overwrite ut here
    vt, vy, vyy = gradient(V, dt, dy, axis=2)
    _, vx, vxx = gradient(V, dt, dx, axis=1)
    px = np.gradient(P, dx, axis=1) 

    # Fixes BUG #1 - Apply complete 3D Slicing
    t_slice = slice(args.crop_t, U.shape[0] - args.crop_t)
    x_slice = slice(args.crop_x, U.shape[1] - args.crop_x)
    y_slice = slice(args.crop_y, U.shape[2] - args.crop_y)
    print("SLICES (t, x, y):", t_slice, x_slice, y_slice)

    u_crop = U[t_slice, x_slice, y_slice]
    ut_crop = ut[t_slice, x_slice, y_slice]
    uy_crop = uy[t_slice, x_slice, y_slice]
    uyy_crop = uyy[t_slice, x_slice, y_slice]
    ux_crop = ux[t_slice, x_slice, y_slice]
    uxx_crop = uxx[t_slice, x_slice, y_slice]
    v_crop = V[t_slice, x_slice, y_slice]
    vt_crop = vt[t_slice, x_slice, y_slice]
    vy_crop = vy[t_slice, x_slice, y_slice]
    vyy_crop = vyy[t_slice, x_slice, y_slice]
    vx_crop = vx[t_slice, x_slice, y_slice]
    vxx_crop = vxx[t_slice, x_slice, y_slice]
    px_crop = px[t_slice, x_slice, y_slice]

    oper_dict, dictionary = build_dict(u_crop, v_crop, uy_crop, uyy_crop, ux_crop, uxx_crop, px_crop)

    x = oper_dict[:]
    y = ut_crop.reshape(-1)

    tol_values = np.logspace(np.log10(args.tol_min), np.log10(args.tol_max), args.num_tol)
    w_best, tol_best, err_best, val_error = trainStridge(x, y, tol_values)

    print("=" * 72)
    print("Discovered equation")
    print(format_equation(w_best, dictionary, lhs="u_t"))
    print("=" * 72)
    print(f"Best tolerance      : {tol_best:.8e}")
    print(f"Train RMSE          : {err_best:.8e}")
    print(f"Validation RMSE     : {val_error:.8e}")
    print(f"True equation       : u_t = - 1.00000000e+00*u*u_x - 1.00000000e+00*v*u_y - 1.08695652e-03*p_x + 8.80434800e-05*u_xx + 8.80434800e-05*u_yy")
    print("Estimated coefficients")
    for c, name in zip(w_best, dictionary):
        print(f"  {name:8s} : {c:+.8e}")


