# Experimental Report
### Case: Lid Cavity with 4 Moving Walls
Description of Flow:
This experiment studies a two-dimensional cavity flow in which all four walls move at prescribed speeds, setting the fluid into a strong recirculating motion. The moving boundaries inject momentum into the fluid and generate a nonlinear vortex structure with both shear-driven and pressure-driven effects. The objective is to identify the governing streamwise momentum equation directly from the evolving velocity and pressure fields.

> Numerical identification of the streamwise momentum equation from saved velocity and pressure fields.

![4-moving-walls cavity velocity field](./4movingwalls_cavity.png)
> 
## 1. Governing Equation Verification

### Reference equation

`u_t =- 1.00000000e+00*uu_x - 1.00000000e+00*vu_y + 1.00000000e-02*u_xx + 1.00000000e-02*u_yy - 1.00000000e-03*p_x`

`v_t =- 1.00000000e+00*uv_x - 1.00000000e+00*vv_y + 1.00000000e-02*v_xx + 1.00000000e-02*v_yy - 1.00000000e-03*p_y`

### Learned equation

`u_t =- 9.26171495e-01*uu_x - 5.86478455e-01*vu_y + 1.17845623e-02*u_xx + 7.83468588e-03*u_yy - 6.20912385e-04*p_x`

`v_t =- 8.34780067e-01*uv_x - 1.20858250e+00*vv_y + 8.07767889e-03*v_xx + 7.56703159e-03*v_yy - 1.01070735e-03*p_y`


| Term | Reference coefficient | Learned coefficient | Absolute error |
|---|---:|---:|---:|
| `uu_x` | `-1.00000000e+00` | `-9.26171495e-01` | `7.38285050e-02` |
| `vu_y` | `-1.00000000e+00` | `-5.86478455e-01` | `4.13521545e-01` |
| `u_xx` | `+1.00000000e-02` | `+1.17845623e-02` | `1.78456232e-03` |
| `u_yy` | `+1.00000000e-02` | `+7.83468588e-03` | `2.16531412e-03` |
| `p_x` | `-1.00000000e-03` | `-6.20912385e-04` | `3.79087615e-04` |

| Term | Reference coefficient | Learned coefficient | Absolute error |
|---|---:|---:|---:|
| `uv_x` | `-1.00000000e+00` | `-8.34780067e-01` | `1.65219933e-01` |
| `vv_y` | `-1.00000000e+00` | `-1.20858250e+00` | `2.08582497e-01` |
| `v_xx` | `+1.00000000e-02` | `+8.07767889e-03` | `1.92232111e-03` |
| `v_yy` | `+1.00000000e-02` | `+7.56703159e-03` | `2.43296841e-03` |
| `p_y` | `-1.00000000e-03` | `-1.01070735e-03` | `1.07073471e-05` |


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


## 3. Data and Pre-processing

| Quantity | Value |
|---|---:|
| Loaded field shape $(t, x, y)$ | `(9851, 31, 30)` |
| Regression field shape $(t, x, y)$ | `(7851, 28, 29)` |
| Regression samples | `6,375,012` |
| Dictionary terms | `5` |
| First saved time index | `150` |
| Temporal crop at each boundary | `2000` layers |
| x-boundary crop | `1` points |
| y-boundary crop | `1` points |


## 4. STRidge Selection and Error Metrics
| Metric | Value |
|---|---:|
| L0 penalty | `1.00000000e-06` |
| Regularisation parameter, $\lambda$ | `1.00000000e-09` |

| Metric | Value |
|---|---:|
| Selected tolerance | `5.87801607e-04` |
| Training RMSE | `9.15047149e-02` |
| Validation RMSE | `9.39825989e-02` |


| Metric | Value |
|---|---:|
| Selected tolerance | `9.42668455e-04` |
| Training RMSE | `8.05213761e-02` |
| Validation RMSE | `8.02861086e-02` |
