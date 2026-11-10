import numpy as np
from format_report import write_2d_markdown_report, format_equation
import os
import sys; sys.path.append('../')
import config
import matplotlib.pyplot as plt
from model_combined import find_run_directories, load_case, trainStridge


config.initialize_fluid_config() # initialize model
model_config = config.initialize_model_config()


tol_values = np.logspace(np.log10(model_config.tol_min), np.log10(model_config.tol_max), model_config.num_tol)
print(tol_values)

input_directories = find_run_directories(model_config.input_directories)
if not input_directories:
    raise FileNotFoundError(
        "No data_t=*.npz snapshots found in the supplied input directories"
    )
print(f"Loading {len(input_directories)} run directories")

case_results = [
    load_case(
        output_directory,
        model_config.data_start,
        model_config.crop_t,
        model_config.crop_x,
        model_config.crop_y,
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
) = trainStridge(x, y, tol_values, model_config.lam, model_config.l0_penalty, model_config.seed)

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
    x_v, y_v, tol_values, model_config.lam, model_config.l0_penalty, model_config.seed
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
    model_config.report,
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
    data_start=model_config.data_start,
    crop_settings={"t": model_config.crop_t, "x": model_config.crop_x, "y": model_config.crop_y},
    lam=model_config.lam,
    l0_penalty=model_config.l0_penalty,
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
print(f"Markdown U/V report written to: {model_config.report}")
