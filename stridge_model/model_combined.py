import numpy as np
from format_report import write_2d_markdown_report, format_equation
import sys; sys.path.append('../')
import os
import argparse
import config
import matplotlib.pyplot as plt


def count_files_scandir(directory):
    count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                count += 1
    return count


def load_case_fields(output_directory, data_start):
    if any(
        name.startswith("data_t=") and name.endswith(".npz")
        for name in os.listdir(output_directory)
    ):
        return load_npz_files(output_directory, data_start)

    field_directories = {
        "u": os.path.join(output_directory, "u_velocity_field"),
        "v": os.path.join(output_directory, "v_velocity_field"),
        "p": os.path.join(output_directory, "pressure_field"),
    }
    time_len = count_files_scandir(field_directories["u"])
    snapshot_indices = range(data_start, time_len + 1)

    fields = {}
    for field_name, directory in field_directories.items():
        if field_name == "p":
            filename = lambda index: f"pressure_field_t={index}.txt"
        else:
            filename = lambda index, name=field_name: f"{name}_velocity_t={index}.txt"
        fields[field_name] = np.array([
            np.loadtxt(os.path.join(directory, filename(i)))
            for i in snapshot_indices
        ])

    return fields

def load_npz_files(output_directory, data_start):
    snapshot_files = [
        filename
        for filename in os.listdir(output_directory)
        if filename.startswith("data_t=") and filename.endswith(".npz")
    ]
    snapshot_files.sort(key=lambda filename: int(filename[len("data_t="):-len(".npz")]))

    U = []
    V = []
    P = []
    for filename in snapshot_files:
        index = int(filename[len("data_t="):-len(".npz")])
        if index < data_start:
            continue
        filepath = os.path.join(output_directory, filename)
        with np.load(filepath) as data:
            U.append(data["u"])
            V.append(data["v"])
            P.append(data["p"])
            dt_value = data["dt"]
            dx_value = data["dx"]
            dy_value = data["dy"]

    return {"u": np.array(U), "v": np.array(V), "p": np.array(P), "dt": dt_value, "dx": dx_value, "dy": dy_value}


def find_run_directories(input_directories):
    run_directories = []
    for input_directory in input_directories:
        if not os.path.isdir(input_directory):
            raise FileNotFoundError(f"Input directory does not exist: {input_directory}")

        for directory, _, filenames in os.walk(input_directory):
            if any(
                filename.startswith("data_t=") and filename.endswith(".npz")
                for filename in filenames
            ):
                run_directories.append(directory)

    return list(dict.fromkeys(sorted(run_directories)))


def load_case(output_directory, data_start, crop_t, crop_x, crop_y, fields=None):
    if fields is None:
        # fields = load_case_fields(output_directory, data_start)
        fields = load_npz_files(output_directory, data_start)

    case_dt = fields.get("dt") #or dt
    print("CASE DT", case_dt)
    U, V, P = fields["u"][:,:,:-1], fields["v"][:,:-1,:], fields["p"][:,:-1,:-1]
    print(np.array(U).shape, np.array(V).shape, np.array(P).shape)


    ut, uy, uyy = gradient(U, case_dt, fields["dy"], axis=2)
    _, ux, uxx = gradient(U, case_dt, fields["dx"], axis=1)
    vt, vy, vyy = gradient(V, case_dt, fields["dy"], axis=2)
    _, vx, vxx = gradient(V, case_dt, fields["dx"], axis=1)
    _, px, _ = gradient(P, case_dt, fields["dx"], axis=1)
    _, py, _ = gradient(P, case_dt, fields["dy"], axis=2)

    n_time = U.shape[0]
    if crop_t < 0:
        raise ValueError("crop_t must be non-negative")

    # Keep the transient start and limit tail cropping to at most half the data.
    end_index = max(n_time - crop_t, (n_time + 1) // 2)
    t_slice = slice(0, end_index)
    print(f"Keeping {end_index}/{n_time} time layers; cropping {n_time - end_index} from the end")
    x_slice = slice(crop_x, U.shape[1]-crop_x)
    y_slice = slice(crop_y, U.shape[2]-crop_y)

    u_crop = U[t_slice, x_slice, y_slice]
    v_crop = V[t_slice, x_slice, y_slice]

    # compute operator matrix and dictionary keys
    oper_dict, dictionary = build_ns_dict(
        u_crop,
        v_crop,
        uy[t_slice, x_slice, y_slice],
        uyy[t_slice, x_slice, y_slice],
        ux[t_slice, x_slice, y_slice],
        uxx[t_slice, x_slice, y_slice],
        px[t_slice, x_slice, y_slice],
    )
    # compute operator matrix and dictionary keys
    oper_dict_v, dictionary_v = build_v_ns_dict(
        u_crop,
        v_crop,
        vx[t_slice, x_slice, y_slice],
        vxx[t_slice, x_slice, y_slice],
        vy[t_slice, x_slice, y_slice],
        vyy[t_slice, x_slice, y_slice],
        py[t_slice, x_slice, y_slice],
    )

    return (
        oper_dict, # u operator array
        ut[t_slice, x_slice, y_slice].reshape(-1),
        dictionary, # dictionary keys
        oper_dict_v, # v operator array
        vt[t_slice, x_slice, y_slice].reshape(-1),
        dictionary_v, # v dictionary keys
        U.shape,
        u_crop.shape,
    )



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
    ns_dict['u_px'] = u*p_x
    ns_dict['v_px'] = v*p_x
    ns_dict['uu_xx'] = u * u_xx

    for key, val in ns_dict.items():
        dict_keys.append(key)
        oper.append(val.reshape(-1))

    oper = np.transpose(np.array(oper))
    print("Dictionary shape:", oper.shape)
    N_features = oper.shape[1]
    print("N_features:", N_features)
    return oper, dict_keys


def build_v_ns_dict(u, v, vx, v_xx, vy, v_yy, p_y):
    dict_keys = []
    oper = []
    ns_dict = {
        'uv_x': u * vx,
        'vv_y': v * vy,
        'v_xx': v_xx,
        'v_yy': v_yy,
        'p_y': p_y,
        'vp_y': v*p_y,
        'up_y': u*p_y,
        'vv_yy': v*v_yy
    }

    for key, val in ns_dict.items():
        dict_keys.append(key)
        oper.append(val.reshape(-1))

    oper = np.transpose(np.array(oper))
    print("V dictionary shape:", oper.shape)
    print("V features:", oper.shape[1])
    return oper, dict_keys



def stridge1(X, Y, tol, lam, max_iter = 100):
    d = X.shape[1]
    biginds = np.arange(d)  # initialise array for big indices
    biginds_prev = biginds

    #normalise theta
    norm = np.linalg.norm(X, axis=0)
    norm[norm<1e-16] = 1.0
    normy = np.linalg.norm(Y)
    if normy < 1e-16:
        normy = 1.0
    X_norm = X/norm
    Y_norm = Y/normy

    # Calculate initial weights
    weights = np.linalg.lstsq(X_norm.T @ X_norm + lam * np.eye(d), X_norm.T @ Y_norm, rcond=None)[0]
    # print("weights:",weights)

    if biginds.size == 0:
        return weights

    for iter in range(max_iter):
        physical_weights = weights * normy / norm
        biginds = np.where(abs(physical_weights) >= tol)[0]
        # If the weights do not change, break the loop
        if np.array_equal(biginds, biginds_prev):
            break
        if biginds.size==0:
            weights[:]=0
            break
        biginds_prev = biginds.copy()
        weights[:] = 0
        weights[biginds] = np.linalg.lstsq(
            X_norm[:, biginds].T.dot(X_norm[:, biginds]) + lam * np.eye(len(biginds)),
            X_norm[:, biginds].T.dot(Y_norm),
            rcond=None
        )[0]

    physical_weights = weights * normy / norm
    print("PHYSICAL WEIGHTS", physical_weights)
    biginds = np.where(abs(physical_weights) >= tol)[0]
    if biginds.size > 0:
        weights[:] = 0.0
        weights[biginds] = np.linalg.lstsq(X_norm[:, biginds], Y_norm, rcond=None)[0]
        weights = weights * normy / norm
    else:
        weights[:] = 0.0

    return np.array(weights)

def trainStridge(X, y, tol_values, lam, l0_penalty, seed=0):
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    train_size = int(n_samples * 0.8)
    indices = rng.permutation(n_samples)
    train_indices = indices[:train_size]
    validate_indices = indices[train_size:]
    X_train, X_validate = X[train_indices], X[validate_indices]
    y_train, y_validate = y[train_indices], y[validate_indices]

    w_best = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
    train_residual = y_train - X_train @ w_best
    val_residual = y_validate - X_validate @ w_best
    train_err_best = np.sqrt(np.mean(train_residual**2))
    val_err_best = np.sqrt(np.mean(val_residual**2))

    score_best = np.inf
    tol_best = 0
    train_loss = []
    val_loss = []
    evaluated_tol_values = []

    for tolerance in tol_values:
        weights = stridge1(X_train, y_train, tolerance, lam)
        train_residual = y_train - X_train @ weights
        val_residual = y_validate - X_validate @ weights
        train_err = np.sqrt(np.mean(train_residual**2))
        val_err = np.sqrt(np.mean(val_residual**2))
        train_loss.append(train_err)
        val_loss.append(val_err)
        evaluated_tol_values.append(tolerance)
        print("TOLERANCE:",tolerance)
        print("TRAIN LOSS:", train_err)
        print("VALIDATION LOSS:", val_err)
        score = val_err + l0_penalty * np.count_nonzero(np.abs(weights) > 1e-12)
        if score < score_best:
            # Improved: keep direction
            # print("error improved", train_err)
            score_best = score
            w_best = weights
            train_err_best = train_err
            val_err_best = val_err
            tol_best = tolerance
        if val_err <= 10**-8:
            break



    return (
        w_best,
        tol_best,
        train_err_best,
        val_err_best,
        train_loss,
        val_loss,
        evaluated_tol_values,
    )





if __name__ == "__main__":

    config.initialize_model_config() # initialize model

    parser = argparse.ArgumentParser(description="Learn the startup Poiseuille PDE from saved data.")
    parser.add_argument("--lam", type=float, default=1e-6)
    # parser.add_argument("--tol_min", type=float, default=1e-8)
    # parser.add_argument("--tol_max", type=float, default=1e-2)
    parser.add_argument("--tol_min", type=float, default=1e-2)
    parser.add_argument("--tol_max", type=float, default=0.15)
    parser.add_argument("--num_tol", type=int, default=10)
    parser.add_argument("--l0_penalty", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_start", type=int, default=1)
    parser.add_argument(
        "--input_directories",
        nargs="+",
        default=[
            os.path.join(os.path.dirname(__file__), "output_npz"),
        ],
    )
    parser.add_argument("--crop_t", type=int, default=2000, help="discard this many time layers from the end")
    parser.add_argument("--crop_y", type=int, default=1, help="discard this many spatial points near each wall")
    parser.add_argument("--crop_x", type=int, default=1, help="discard this many spatial points near each wall")
    parser.add_argument(
        "--report",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "stridge_report.md"),
        help="path for the generated Markdown report",
    )
    args = parser.parse_args()

    tol_values = np.logspace(np.log10(args.tol_min), np.log10(args.tol_max), args.num_tol)
    print(tol_values)

    input_directories = find_run_directories(args.input_directories)
    if not input_directories:
        raise FileNotFoundError(
            "No data_t=*.npz snapshots found in the supplied input directories"
        )
    print(f"Loading {len(input_directories)} run directories")

    case_results = [
        load_case(
            output_directory,
            args.data_start,
            args.crop_t,
            args.crop_x,
            args.crop_y,
        )
        for output_directory in input_directories
    ]

    oper_dict = np.concatenate([result[0] for result in case_results])
    y = np.concatenate([result[1] for result in case_results])
    dictionary = case_results[0][2]
    if any(result[2] != dictionary for result in case_results[1:]):
        raise ValueError("All cases must use the same U dictionary ordering")

    x = oper_dict
    (
        w_best,
        tol_best,
        err_best,
        val_error,
        train_loss,
        val_loss,
        evaluated_tol_values_u,
    ) = trainStridge(x, y, tol_values, args.lam, args.l0_penalty)

    oper_dict_v = np.concatenate([result[3] for result in case_results])
    y_v = np.concatenate([result[4] for result in case_results])
    dictionary_v = case_results[0][5]
    if any(result[5] != dictionary_v for result in case_results[1:]):
        raise ValueError("All cases must use the same V dictionary ordering")

    x_v = oper_dict_v
    (
        w_best_v,
        tol_best_v,
        err_best_v,
        val_error_v,
        train_loss_v,
        val_loss_v,
        evaluated_tol_values_v,
    ) = trainStridge(
        x_v, y_v, tol_values, args.lam, args.l0_penalty
    )

    datashape = [case_results[i][6] for i in range(len(input_directories))]
    datashape_cropped = [case_results[i][7] for i in range(len(input_directories))]

    try:
        fig = plt.figure(figsize=(18, 12))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot(evaluated_tol_values_u, train_loss)
        ax1.plot(evaluated_tol_values_u, val_loss)
        ax1.set_xscale("log")
        ax1.legend(["Training", "Validation"])
        u_loss_values = np.concatenate((train_loss, val_loss))
        ax1.set_ylim(np.min(u_loss_values), np.max(u_loss_values))
        ax1.set_title("Training and Validation Loss of Weights for U field")
        ax1.set_xlabel("Tolerance")
        ax1.set_ylabel("Loss")
        ax2.plot(evaluated_tol_values_v, train_loss_v)
        ax2.plot(evaluated_tol_values_v, val_loss_v)
        ax2.set_xscale("log")
        ax2.legend(["Training", "Validation"])
        v_loss_values = np.concatenate((train_loss_v, val_loss_v))
        ax2.set_ylim(np.min(v_loss_values), np.max(v_loss_values))
        ax2.set_title("Training and Validation Loss of Weights for V field")
        ax2.set_xlabel("Tolerance")
        ax2.set_ylabel("Loss")
        plt.show()
    except Exception as error:
        print(f"Warning: loss plots could not be displayed: {error}")
    finally:
        write_2d_markdown_report(
        args.report,
        w_best,
        dictionary,
        w_best_v,
        dictionary_v,
        datashape,
        datashape_cropped,
        tol_best,
        err_best,
        val_error,
        tol_best_v,
        err_best_v,
        val_error_v,
        data_start=args.data_start,
        crop_settings={"t": args.crop_t, "x": args.crop_x, "y": args.crop_y},
        lam=args.lam,
        l0_penalty=args.l0_penalty,
        u_reference_coefficients={
            "uu_x": -config.c_advective,
            "vu_y": -config.c_advective,
            "u_xx": config.c_viscous,
            "u_yy": config.c_viscous,
            "p_x": -config.c_pressure,
        },
        v_reference_coefficients={
            "uv_x": -config.c_advective,
            "vv_y": -config.c_advective,
            "v_xx": config.c_viscous,
            "v_yy": config.c_viscous,
            "p_y": -config.c_pressure,
        },
        input_directory=", ".join(input_directories),
    )
    print(f"Markdown U/V report written to: {args.report}")



