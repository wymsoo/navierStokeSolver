# Experimental Report
### Case: Lid Driven Cavity
Description of Flow:
This case models the standard two-dimensional lid-driven cavity, in which the top wall moves tangentially while the remaining three walls remain stationary. The imposed lid motion generates a primary circulating vortex and weaker secondary recirculation regions, creating a strongly nonlinear velocity field. The experiment aims to recover the streamwise momentum equation from the recorded velocity and pressure data.

> Numerical identification of the streamwise momentum equation from saved velocity and pressure fields.

![Lid-cavity velocity field](./lidcav.png)

## 1. Governing Equation Verification

### Reference equation

`u_t =- 1.00000000e+00*uu_x - 1.00000000e+00*vu_y + 1.00000000e-02*u_xx + 1.00000000e-02*u_yy - 1.00000000e-03*p_x`


`v_t =- 1.00000000e+00*uv_x - 1.00000000e+00*vv_y + 1.00000000e-02*v_xx + 1.00000000e-02*v_yy - 1.00000000e-03*p_y`

### Learned equation

`u_t =- 1.13842289e+00*uu_x - 1.03055812e+00*vu_y + 8.40649079e-03*u_xx + 1.03461437e-02*u_yy - 1.04792514e-03*p_x`


`v_t =- 1.06616627e+00*uv_x - 1.25928856e+00*vv_y + 3.54865766e-03*v_xx + 1.23468066e-02*v_yy - 1.05083164e-03*p_y`

| Term | Reference coefficient | Learned coefficient | Absolute error |
|---|---:|---:|---:|
| `uu_x` | `-1.00000000e+00` | `-1.13842289e+00` | `1.38422889e-01` |
| `vu_y` | `-1.00000000e+00` | `-1.03055812e+00` | `3.05581219e-02` |
| `u_xx` | `+1.00000000e-02` | `+8.40649079e-03` | `1.59350921e-03` |
| `u_yy` | `+1.00000000e-02` | `+1.03461437e-02` | `3.46143665e-04` |
| `p_x` | `-1.00000000e-03` | `-1.04792514e-03` | `4.79251383e-05` |

| Term | Reference coefficient | Learned coefficient | Absolute error |
|---|---:|---:|---:|
| `uv_x` | `-1.00000000e+00` | `-1.06616627e+00` | `6.61662724e-02` |
| `vv_y` | `-1.00000000e+00` | `-1.25928856e+00` | `2.59288563e-01` |
| `v_xx` | `+1.00000000e-02` | `+3.54865766e-03` | `6.45134234e-03` |
| `v_yy` | `+1.00000000e-02` | `+1.23468066e-02` | `2.34680658e-03` |
| `p_y` | `-1.00000000e-03` | `-1.05083164e-03` | `5.08316415e-05` |


## 2. Simulation Parameters

| Quantity | Symbol | Value | Unit |
|---|---:|---:|---|
| Fluid density | `rho` | `1.00000000e+03` | kg m$^{-3}$ |
| Dynamic viscosity | `viscosity` | `1.00000000e+01` | Pa s |
| Kinematic viscosity | `nu` | `1.00000000e-02` | m$^2$ s$^{-1}$ |
| Characteristic velocity | `u_max` | `1.00000000e+00` | m s$^{-1}$ |
| Characteristic length | `D` | `1.00000000e+00` | m |
| Reynolds number | `Re` | `1.00000000e+02` | dimensionless |
| Body force | `G` | `9.81000000e+00` | m s$^{-2}$ |
| Grid points in x | `Nx` | `3.10000000e+01` | cells |
| Grid points in y | `Ny` | `3.10000000e+01` | cells |
| Grid spacing in x | `dx` | `3.22580645e-02` | m |
| Grid spacing in y | `dy` | `3.22580645e-02` | m |
| Time step | `dt` | `1.00000000e-04` | s |
| Numerical threshold | `epsilon` | `7.00000000e-11` | - |

### 3. STRidge Selection and Error Metrics

| Regularisation parameter, $\lambda$ | `1.00000000e-09` |
| L0 penalty | `1.00000000e-06` |

| Metric | Value |
|---|---:|
| Selected tolerance | `9.42668455e-04` |
| Training RMSE | `4.35889502e-03` |
| Validation RMSE | `6.52899072e-03` |

| Metric | Value |
|---|---:|
| Selected tolerance | `9.42668455e-04` |
| Training RMSE | `1.10494359e-02` |
| Validation RMSE | `6.16076275e-03` |

## 4. Data and Pre-processing

| Quantity | Value |
|---|---:|
| Loaded field shape $(t, x, y)$ | `(6655, 31, 30)` |
| Regression field shape $(t, x, y)$ | `(4255, 22, 27)` |
| Regression samples | `2,527,470` |
| Dictionary terms | `5` |
| First saved time index | `150` |
| Temporal crop at each boundary | `1200` layers |
| x-boundary crop | `4` points |
| y-boundary crop | `2` points |




