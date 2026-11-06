# Physics Simulation and STRidge Identification Report

> Numerical identification of the streamwise momentum equation from saved velocity and pressure fields.

## 1. Simulation Parameters

| Quantity | Symbol | Value | Unit |
|---|---:|---:|---|
| Fluid density | `rho` | `1.00000000e+01` | kg m$^{-3}$ |
| Dynamic viscosity | `viscosity` | `1.00000000e+01` | Pa s |
| Kinematic viscosity | `nu` | `1.00000000e+00` | m$^2$ s$^{-1}$ |
| Characteristic velocity | `u_max` | `1.00000000e+00` | m s$^{-1}$ |
| Characteristic length | `D` | `1.00000000e+00` | m |
| Reynolds number | `Re` | `1.00000000e+00` | dimensionless |
| Body force | `Fx_ext` | `0.00000000e+00` | m s$^{-2}$ |
| Grid points in x | `Nx` | `3.10000000e+01` | cells |
| Grid points in y | `Ny` | `3.10000000e+01` | cells |
| Grid spacing in x | `dx` | `3.22580645e-02` | m |
| Grid spacing in y | `dy` | `3.22580645e-02` | m |
| Time step | `dt` | `1.00000000e-04` | s |
| Numerical threshold | `epsilon` | `7.00000000e-11` | - |

## 2. Numerical Discretisation

| Quantity | Symbol | Value | Unit |
|---|---:|---:|---|
| Parameters listed above are the values used by the selected solver. |  |  |  |

## 3. Data and Pre-processing

| Quantity | Value |
|---|---:|
| Loaded field shape $(t, x, y)$ | `[(6678, 80, 80), (1802, 80, 80), (2175, 80, 80), (10000, 80, 80)]` |
| Regression field shape $(t, x, y)$ | `[(4678, 78, 78), (901, 78, 78), (1088, 78, 78), (8000, 78, 78)]` |
| Regression samples | `-8,268,782,793,637,494,784` |
| Dictionary terms | `8` |
| First saved time index | `1` |
| Temporal crop at each boundary | `2000` layers |
| x-boundary crop | `1` points |
| y-boundary crop | `1` points |

## 4. Governing Equation Verification

### Reference equation

`u_t =- 1.00000000e+00*uu_x - 1.00000000e+00*vu_y + 1.00000000e+00*u_xx + 1.00000000e+00*u_yy - 1.00000000e-01*p_x`

### Learned equation

`u_t =2.38343106e+00*uu_x + 3.71408080e+00*vu_y + 9.22373053e-01*u_xx + 1.02369093e+00*u_yy - 9.79729907e-02*p_x + 1.17584552e-02*u_px - 4.02348133e-02*v_px + 3.91600747e-02*uu_xx`

| Term | Reference coefficient | Learned coefficient | Percentage error |
|---|---:|---:|---:|
| `uu_x` | `-1.00000000e+00` | `+2.38343106e+00` | `-3.38343106e+02%` |
| `vu_y` | `-1.00000000e+00` | `+3.71408080e+00` | `-4.71408080e+02%` |
| `u_xx` | `+1.00000000e+00` | `+9.22373053e-01` | `7.76269471e+00%` |
| `u_yy` | `+1.00000000e+00` | `+1.02369093e+00` | `2.36909274e+00%` |
| `p_x` | `-1.00000000e-01` | `-9.79729907e-02` | `-2.02700931e+00%` |
| `u_px` | `+0.00000000e+00` | `+1.17584552e-02` | `inf%` |
| `v_px` | `+0.00000000e+00` | `-4.02348133e-02` | `inf%` |
| `uu_xx` | `+0.00000000e+00` | `+3.91600747e-02` | `inf%` |

## 5. STRidge Selection and Error Metrics

| Metric | Value |
|---|---:|
| Regularisation parameter, $\lambda$ | `1.00000000e-06` |
| Selected tolerance | `1.00000000e-02` |
| L0 penalty | `1.00000000e-06` |
| Training RMSE | `1.04769632e+01` |
| Validation RMSE | `1.04612489e+01` |

## 6. Reproducibility Notes

- Input directory: `/Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/fourmovingwalls_Nx81/run_001, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/fourmovingwalls_Nx81/run_002, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/lidcavity_Nx81/run_001, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/poiseuille_Nx81/run_002`
- Fields were loaded starting at saved time index `1`.
- Derivatives were calculated using second-order edge-aware finite differences via `numpy.gradient`.
- The learned equation is compared term-by-term with the supplied reference coefficients.

## 7. Transverse Momentum Equation

### Reference equation

`v_t =- 1.00000000e+00*uv_x - 1.00000000e+00*vv_y + 1.00000000e+00*v_xx + 1.00000000e+00*v_yy - 1.00000000e-01*p_y`

### Learned equation

`v_t =- 1.05148138e+00*uv_x - 6.89790716e+00*vv_y + 1.03644919e+00*v_xx + 9.86194654e-01*v_yy - 9.99228337e-02*p_y - 1.21816122e-02*vp_y + 4.90858646e-02*up_y + 4.19806811e-02*vv_yy`

| Term | Reference coefficient | Learned coefficient | Percentage error |
|---|---:|---:|---:|
| `uv_x` | `-1.00000000e+00` | `-1.05148138e+00` | `-5.14813844e+00%` |
| `vv_y` | `-1.00000000e+00` | `-6.89790716e+00` | `-5.89790716e+02%` |
| `v_xx` | `+1.00000000e+00` | `+1.03644919e+00` | `3.64491903e+00%` |
| `v_yy` | `+1.00000000e+00` | `+9.86194654e-01` | `1.38053457e+00%` |
| `p_y` | `-1.00000000e-01` | `-9.99228337e-02` | `-7.71663454e-02%` |
| `vp_y` | `+0.00000000e+00` | `-1.21816122e-02` | `inf%` |
| `up_y` | `+0.00000000e+00` | `+4.90858646e-02` | `inf%` |
| `vv_yy` | `+0.00000000e+00` | `+4.19806811e-02` | `inf%` |

### V STRidge Selection and Error Metrics

| Metric | Value |
|---|---:|
| Selected tolerance | `1.00000000e-02` |
| Training RMSE | `7.12635771e+00` |
| Validation RMSE | `7.08158121e+00` |
