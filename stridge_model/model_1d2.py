import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from format_report import write_markdown_report, format_equation
import sys; sys.path.append('../')
import scipy.io as sio
import itertools
import os
import argparse
import linecache
import matplotlib.pyplot as plt
import itertools


dt = 0.0001
epsilon = 7e-11
# Grid size
Nx = 31
Ny = 31
dx = 1.0 / Nx
dy = 1.0 / Ny
G = 9.81
rho = 1000
u_max = 1
# D = 0.005
D = 1.0
viscosity = 10
Re = (rho * u_max * D) / viscosity
nu = viscosity/rho
c_advective = 1
c_pressure = 1/rho
c_viscous = 1/Re





def count_files_scandir(directory):
    count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                count += 1
    return count


def gradient(u, dt, dy, axis):
    ut = np.gradient(u, dt, axis=0, edge_order=2)
    uy = np.gradient(u, dy, axis=axis, edge_order=2)
    uyy = np.gradient(uy, dy, axis=axis,edge_order=2)
    return ut, uy, uyy



def build_ns_dict(u, v, uy, u_yy, ux, u_xx, p_x):
    dict_keys = []
    oper = []
    ns_dict = {}
    # ns_dict['1'] = np.ones_like(u)
    ns_dict['uu_x'] = u * ux
    ns_dict['vu_y'] = v * uy
    ns_dict['u_xx'] = u_xx
    ns_dict['u_yy'] = u_yy
    ns_dict['p_x'] = p_x

    for key, val in ns_dict.items():
        dict_keys.append(key)
        oper.append(val.reshape(-1))

    oper = np.transpose(np.array(oper))
    print("Dictionary shape:", oper.shape)
    N_features = oper.shape[1]
    print("N_features:", N_features)
    return oper, dict_keys

# def build_dict(u, uy, u_yy):

#     all_differentials = [u, uy, u_yy]
#     all_labels = ['u','u_y','u_yy']

#     # Dictionary: for display
#     dictionary = {}
#     combinations = itertools.combinations(all_differentials, 2)
#     label_combinations = itertools.combinations(all_labels, 2)

#     # compute all product of combinations
#     oper_dict = [(a*b).reshape(-1) for a, b in combinations]


#     # Use string keys for combined labels (e.g. 'u*u_y') instead of tuples
#     for label, oper in zip(label_combinations, oper_dict):
#         if isinstance(label, tuple):
#             key = f"{label[0]}*{label[1]}"
#         else:
#             key = label
#         dictionary[key] = oper

#     # add back the single terms
#     for diff, label in zip(all_differentials, all_labels):
#         oper_dict.append(diff.reshape(-1))
#         dictionary[label] = diff

#     dictionary['1'] = np.ones_like(u)
#     # dictionary['u**2'] = (u**2).reshape(-1)
#     oper_dict.append(np.ones_like(u).reshape(-1))
#     # oper_dict.append((u**2).reshape(-1))

#     dictionary_keys = []
#     for key in dictionary:
#         dictionary_keys.append(key)


#     # oper_dict = np.transpose(np.array(oper_dict),(1,0,2,3))
#     oper_dict = np.transpose(np.array(oper_dict))
#     return oper_dict, dictionary_keys



def stridge1(X, Y, tol, lam, max_iter = 100):
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
        biginds = np.where(abs(weights/norm) >= tol)[0]
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

    biginds = np.where(abs(weights/norm)>= tol)[0]
    if biginds.size > 0:
        weights[:] = 0.0
        weights[biginds] = np.linalg.lstsq(X_norm[:, biginds], Y, rcond=None)[0]
    else:
        weights[:] = 0.0

    return np.array(weights/norm)

def trainStridge(X, y, tol_values, lam, l0_penalty, seed=0):
    rng = np.random.default_rng(seed)

    n_samples = X.shape[0]
    train_size = int(n_samples * 0.8)
    X_train, X_validate = X[:train_size], X[train_size:]
    y_train, y_validate = y[:train_size], y[train_size:]

    score_best = np.inf
    w_best = np.linalg.lstsq(X, y, rcond=None)[0]
    train_err_best = np.linalg.norm(y_train - X_train @ w_best, 2) + l0_penalty * np.count_nonzero(w_best)
    val_err_best = np.linalg.norm(y_validate - X_validate @ w_best, 2) + l0_penalty * np.count_nonzero(w_best)
    tol_best = 0

    for tolerance in tol_values:
        weights = stridge1(X_train, y_train, tolerance, args.lam)
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





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Learn the startup Poiseuille PDE from saved data.")
    # parser.add_argument("--input", type=str, default=os.path.join(os.path.dirname(__file__), "startup_poiseuille_data.npz"))
    parser.add_argument("--lam", type=float, default=1e-9)
    parser.add_argument("--tol_min", type=float, default=1e-6)
    parser.add_argument("--tol_max", type=float, default=1e-2)
    parser.add_argument("--num_tol", type=int, default=40)
    parser.add_argument("--l0_penalty", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crop_t", type=int, default=1200, help="discard this many time layers near each temporal boundary")
    parser.add_argument("--crop_y", type=int, default=2, help="discard this many spatial points near each wall")
    parser.add_argument("--crop_x", type=int, default=4, help="discard this many spatial points near each wall")
    parser.add_argument(
        "--report",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "stridge_report.md"),
        help="path for the generated Markdown report",
    )
    args = parser.parse_args()

    

    file_path = os.path.join(os.path.dirname(__file__), "output_4movingwalls", "u_velocity_field")
    file_path_v = os.path.join(os.path.dirname(__file__), "output_4movingwalls", "v_velocity_field")
    file_path_p = os.path.join(os.path.dirname(__file__), "output_4movingwalls", "pressure_field")

    time_len = count_files_scandir(file_path)

    # Efficient loading using np.loadtxt
    results_u = [
        np.loadtxt(os.path.join(file_path, f"u_velocity_t={i}.txt"))
        for i in range(150, time_len + 1)
    ]

    results_p = [
        np.loadtxt(os.path.join(file_path_p, f"pressure_field_t={i}.txt"))
        for i in range(150, time_len + 1)
    ]

    results_v = [
        np.loadtxt(os.path.join(file_path_v, f"v_velocity_t={i}.txt"))
        for i in range(150, time_len + 1)
    ]

    U = np.array(results_u)
    V = np.array(results_v)
    P = np.array(results_p)

    ut, uy, uyy = gradient(U, dt, dy, axis=2)
    _, ux, uxx = gradient(U, dt, dx, axis=1)
    px = np.gradient(P, dx, axis=1)
    t_slice = slice(args.crop_t, U.shape[0] - args.crop_t)
    x_slice = slice(args.crop_x, U.shape[1] - args.crop_x)
    y_slice = slice(args.crop_y, U.shape[2] - args.crop_y)

    u_crop = U[t_slice, x_slice, y_slice]
    v_crop = V[t_slice, x_slice, y_slice]
    ut_crop = ut[t_slice, x_slice, y_slice]
    uy_crop = uy[t_slice, x_slice, y_slice]
    uyy_crop = uyy[t_slice, x_slice, y_slice]
    ux_crop = ux[t_slice, x_slice, y_slice]
    uxx_crop = uxx[t_slice, x_slice, y_slice]
    px_crop = px[t_slice, x_slice, y_slice]

    oper_dict, dictionary = build_ns_dict(u_crop, v_crop, uy_crop, uyy_crop, ux_crop, uxx_crop, px_crop)

    x = oper_dict[:]
    y = ut_crop.reshape(-1)

    tol_values = np.logspace(np.log10(args.tol_min), np.log10(args.tol_max), args.num_tol)
    w_best, tol_best, err_best, val_error = trainStridge(x,y,tol_values,args.lam, args.l0_penalty)

    write_markdown_report(
        args.report,
        w_best,
        dictionary,
        U.shape,
        u_crop.shape,
        tol_best,
        err_best,
        val_error,
        data_start=150,
        crop_settings={"t": args.crop_t, "x": args.crop_x, "y": args.crop_y},
        lam=args.lam,
        l0_penalty=args.l0_penalty,
    )
    print(f"Markdown report written to: {args.report}")



