# Physics Simulation and STRidge Identification Report

> Numerical identification of the streamwise momentum equation from saved velocity and pressure fields.

## 1. Simulation Parameters

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

## 2. Numerical Discretisation

| Quantity | Symbol | Value | Unit |
|---|---:|---:|---|
| Parameters listed above are the values used by the selected solver. |  |  |  |

## 3. Data and Pre-processing

| Quantity | Value |
|---|---:|
| Loaded field shape $(t, x, y)$ | `[(5435, 80, 80), (6613, 80, 80)]` |
| Regression field shape $(t, x, y)$ | `[(3435, 76, 76), (4613, 76, 76)]` |
| Regression samples | `528,645,530,945,280` |
| Dictionary terms | `5` |
| First saved time index | `150` |
| Temporal crop at each boundary | `1000` layers |
| x-boundary crop | `2` points |
| y-boundary crop | `2` points |

## 4. Governing Equation Verification

### Reference equation

`u_t =- 1.00000000e+00*uu_x - 1.00000000e+00*vu_y + 1.00000000e-02*u_xx + 1.00000000e-02*u_yy - 1.00000000e-03*p_x`

### Learned equation

`u_t =- 1.10882883e+00*uu_x - 9.00486192e-01*vu_y + 1.17132003e-02*u_xx + 9.54792967e-03*u_yy - 9.61052126e-04*p_x`

| Term | Reference coefficient | Learned coefficient | Percentage error |
|---|---:|---:|---:|
| `uu_x` | `-1.00000000e+00` | `-1.10882883e+00` | `-1.08828833e+01%` |
| `vu_y` | `-1.00000000e+00` | `-9.00486192e-01` | `-9.95138077e+00%` |
| `u_xx` | `+1.00000000e-02` | `+1.17132003e-02` | `1.71320035e+01%` |
| `u_yy` | `+1.00000000e-02` | `+9.54792967e-03` | `4.52070326e+00%` |
| `p_x` | `-1.00000000e-03` | `-9.61052126e-04` | `-3.89478738e+00%` |

## 5. STRidge Selection and Error Metrics

| Metric | Value |
|---|---:|
| Regularisation parameter, $\lambda$ | `1.00000000e-08` |
| Selected tolerance | `9.42668455e-04` |
| L0 penalty | `1.00000000e-06` |
| Training RMSE | `5.54856252e-02` |
| Validation RMSE | `2.76317905e-02` |

## 6. Reproducibility Notes

- Input directory: `/Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output/4movingwalls_80, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output/lidcavity_81`
- Fields were loaded starting at saved time index `150`.
- Derivatives were calculated using second-order edge-aware finite differences via `numpy.gradient`.
- The learned equation is compared term-by-term with the supplied reference coefficients.

## 7. Transverse Momentum Equation

### Reference equation

`v_t =- 1.00000000e+00*uv_x - 1.00000000e+00*vv_y + 1.00000000e-02*v_xx + 1.00000000e-02*v_yy - 1.00000000e-03*p_y`

### Learned equation

`v_t =- 1.03471651e+00*uv_x - 1.11093148e+00*vv_y + 9.53136692e-03*v_xx + 1.38013045e-02*v_yy - 1.05880897e-03*p_y`

| Term | Reference coefficient | Learned coefficient | Percentage error |
|---|---:|---:|---:|
| `uv_x` | `-1.00000000e+00` | `-1.03471651e+00` | `-3.47165138e+00%` |
| `vv_y` | `-1.00000000e+00` | `-1.11093148e+00` | `-1.10931478e+01%` |
| `v_xx` | `+1.00000000e-02` | `+9.53136692e-03` | `4.68633079e+00%` |
| `v_yy` | `+1.00000000e-02` | `+1.38013045e-02` | `3.80130446e+01%` |
| `p_y` | `-1.00000000e-03` | `-1.05880897e-03` | `-5.88089681e+00%` |

### V STRidge Selection and Error Metrics

| Metric | Value |
|---|---:|
| Selected tolerance | `9.42668455e-04` |
| Training RMSE | `4.52023912e-02` |
| Validation RMSE | `1.64496948e-02` |
