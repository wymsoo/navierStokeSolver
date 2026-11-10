import numpy as np
from format_report import write_2d_markdown_report, format_equation
import sys; sys.path.append('../')
import os
from numerical_solver.models.operators.stagger import stagger_back
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
    time_steps = []
    dx_value = None
    dy_value = None
    field_shapes = {}
    for filename in snapshot_files:
        index = int(filename[len("data_t="):-len(".npz")])
        if index < data_start:
            continue
        filepath = os.path.join(output_directory, filename)
        with np.load(filepath) as data:
            required_fields = {"u", "v", "p", "dt", "dx", "dy"}
            missing_fields = required_fields.difference(data.files)
            if missing_fields:
                raise ValueError(f"Snapshot {filepath} is missing fields: {sorted(missing_fields)}")

            snapshot_fields = {name: np.asarray(data[name]) for name in ("u", "v", "p")}
            for name, field in snapshot_fields.items():
                if field.ndim != 2 or not np.issubdtype(field.dtype, np.number):
                    raise ValueError(f"Snapshot field {name!r} in {filepath} must be a numeric 2D array")
                if name in field_shapes and field.shape != field_shapes[name]:
                    raise ValueError(f"Field {name!r} has inconsistent shapes in {output_directory}")
                field_shapes[name] = field.shape

            dt_value = float(np.asarray(data["dt"]).item())
            snapshot_dx = float(np.asarray(data["dx"]).item())
            snapshot_dy = float(np.asarray(data["dy"]).item())
            if not np.isfinite(dt_value) or dt_value <= 0:
                raise ValueError(f"Snapshot time step must be positive and finite: {filepath}")
            if not np.isfinite(snapshot_dx) or snapshot_dx <= 0 or not np.isfinite(snapshot_dy) or snapshot_dy <= 0:
                raise ValueError(f"Snapshot grid spacing must be positive and finite: {filepath}")
            if dx_value is not None and not np.isclose(snapshot_dx, dx_value):
                raise ValueError(f"Inconsistent dx values in {output_directory}")
            if dy_value is not None and not np.isclose(snapshot_dy, dy_value):
                raise ValueError(f"Inconsistent dy values in {output_directory}")

            U.append(snapshot_fields["u"])
            V.append(snapshot_fields["v"])
            P.append(snapshot_fields["p"])
            time_steps.append(dt_value)
            dx_value = snapshot_dx
            dy_value = snapshot_dy

    if not time_steps:
        raise ValueError(f"No snapshots at or after data_start={data_start} in {output_directory}")

    time = np.cumsum(time_steps) - time_steps[0]
    return {"u": np.array(U), "v": np.array(V), "p": np.array(P), "time": time, "dx": dx_value, "dy": dy_value}


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

    case_time = fields.get("time", fields.get("dt"))
    if case_time is None:
        raise ValueError(f"No time-step metadata found for {output_directory}")
    print("CASE TIME", case_time)
    raw_u, raw_v, P = fields["u"], fields["v"], fields["p"]
    if raw_u.shape == raw_v.shape == P.shape:
        U, V = raw_u, raw_v
    else:
        U, V = stagger_back(raw_u, raw_v)
    if U.shape != P.shape or V.shape != P.shape:
        raise ValueError(
            f"Velocity and pressure grids do not align after staggering in {output_directory}: "
            f"U={U.shape}, V={V.shape}, P={P.shape}"
        )
    print(np.array(U).shape, np.array(V).shape, np.array(P).shape)


    ut, uy, uyy = gradient(U, case_time, fields["dy"], axis=2)
    _, ux, uxx = gradient(U, case_time, fields["dx"], axis=1)
    vt, vy, vyy = gradient(V, case_time, fields["dy"], axis=2)
    _, vx, vxx = gradient(V, case_time, fields["dx"], axis=1)
    _, px, _ = gradient(P, case_time, fields["dx"], axis=1)
    _, py, _ = gradient(P, case_time, fields["dy"], axis=2)

    n_time = U.shape[0]
    if crop_t < 0:
        raise ValueError("crop_t must be non-negative")

    # # Keep the transient start and limit tail cropping to at most half the data.
    # end_index = max(n_time - crop_t, (n_time + 1) // 2)
    # t_slice = slice(0, end_index)
    # print(f"Keeping {end_index}/{n_time} time layers; cropping {n_time - end_index} from the end")

    # instead of using t_slice, sample random points throughout the entire flow
    num_total_rows = ut.shape[0]
    num_desired_rows = 1000            # Choose a manageable number of rows for regression

    row_indices = np.random.choice(min(num_total_rows,n_time), num_desired_rows, replace=False)


    x_slice = slice(crop_x, U.shape[1]-crop_x)
    y_slice = slice(crop_y, U.shape[2]-crop_y)

    u_crop = U[row_indices, x_slice, y_slice]
    v_crop = V[row_indices, x_slice, y_slice]

    # compute operator matrix and dictionary keys
    oper_dict, dictionary = build_ns_dict(
        u_crop,
        v_crop,
        uy[row_indices, x_slice, y_slice],
        uyy[row_indices, x_slice, y_slice],
        ux[row_indices, x_slice, y_slice],
        uxx[row_indices, x_slice, y_slice],
        px[row_indices, x_slice, y_slice],
    )
    # compute operator matrix and dictionary keys
    oper_dict_v, dictionary_v = build_v_ns_dict(
        u_crop,
        v_crop,
        vx[row_indices, x_slice, y_slice],
        vxx[row_indices, x_slice, y_slice],
        vy[row_indices, x_slice, y_slice],
        vyy[row_indices, x_slice, y_slice],
        py[row_indices, x_slice, y_slice],
    )

    return (
        oper_dict, # u operator array
        ut[row_indices, x_slice, y_slice].reshape(-1),
        dictionary, # dictionary keys
        oper_dict_v, # v operator array
        vt[row_indices, x_slice, y_slice].reshape(-1),
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
    # ns_dict['u_px'] = u*p_x
    # ns_dict['v_px'] = v*p_x
    # ns_dict['uu_xx'] = u * u_xx

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
        # 'vp_y': v*p_y,
        # 'up_y': u*p_y,
        # 'vv_yy': v*v_yy
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

