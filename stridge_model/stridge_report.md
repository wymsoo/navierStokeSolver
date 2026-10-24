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
| Loaded field shape $(t, x, y)$ | `[(381, 30, 30), (404, 30, 30), (10000, 30, 30)]` |
| Regression field shape $(t, x, y)$ | `[(181, 28, 28), (204, 28, 28), (9800, 28, 28)]` |
| Regression samples | `174,374,512,331,980,800` |
| Dictionary terms | `5` |
| First saved time index | `1` |
| Temporal crop at each boundary | `100` layers |
| x-boundary crop | `1` points |
| y-boundary crop | `1` points |

## 4. Governing Equation Verification

### Reference equation

`u_t =- 1.00000000e+00*uu_x - 1.00000000e+00*vu_y + 1.00000000e-01*u_xx + 1.00000000e-01*u_yy - 1.00000000e-02*p_x`

### Learned equation

`u_t =- 3.74631824e+01*uu_x - 5.57292686e+01*vu_y + 1.77326125e+00*u_xx + 2.14085336e+00*u_yy - 6.44914508e-01*p_x`

| Term | Reference coefficient | Learned coefficient | Percentage error |
|---|---:|---:|---:|
| `uu_x` | `-1.00000000e+00` | `-3.74631824e+01` | `-3.64631824e+03%` |
| `vu_y` | `-1.00000000e+00` | `-5.57292686e+01` | `-5.47292686e+03%` |
| `u_xx` | `+1.00000000e-01` | `+1.77326125e+00` | `1.67326125e+03%` |
| `u_yy` | `+1.00000000e-01` | `+2.14085336e+00` | `2.04085336e+03%` |
| `p_x` | `-1.00000000e-02` | `-6.44914508e-01` | `-6.34914508e+03%` |

## 5. STRidge Selection and Error Metrics

| Metric | Value |
|---|---:|
| Regularisation parameter, $\lambda$ | `1.00000000e-08` |
| Selected tolerance | `1.00000000e-06` |
| L0 penalty | `1.00000000e-06` |
| Training RMSE | `7.49377819e+00` |
| Validation RMSE | `0.00000000e+00` |

## 6. Reproducibility Notes

- Input directory: `/Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/4movingwalls_31, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/lidcavity_31, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/poiseuille_vertical_31`
- Fields were loaded starting at saved time index `1`.
- Derivatives were calculated using second-order edge-aware finite differences via `numpy.gradient`.
- The learned equation is compared term-by-term with the supplied reference coefficients.

## 7. Transverse Momentum Equation

### Reference equation

`v_t =- 1.00000000e+00*uv_x - 1.00000000e+00*vv_y + 1.00000000e-01*v_xx + 1.00000000e-01*v_yy - 1.00000000e-02*p_y`

### Learned equation

`v_t =- 3.08640973e+01*uv_x + 5.26329314e+01*vv_y - 4.46319146e-03*v_xx - 5.14729092e-01*v_yy + 1.84585000e-02*p_y`

| Term | Reference coefficient | Learned coefficient | Percentage error |
|---|---:|---:|---:|
| `uv_x` | `-1.00000000e+00` | `-3.08640973e+01` | `-2.98640973e+03%` |
| `vv_y` | `-1.00000000e+00` | `+5.26329314e+01` | `-5.36329314e+03%` |
| `v_xx` | `+1.00000000e-01` | `-4.46319146e-03` | `1.04463191e+02%` |
| `v_yy` | `+1.00000000e-01` | `-5.14729092e-01` | `6.14729092e+02%` |
| `p_y` | `-1.00000000e-02` | `+1.84585000e-02` | `-2.84585000e+02%` |

### V STRidge Selection and Error Metrics

| Metric | Value |
|---|---:|
| Selected tolerance | `3.88815518e-03` |
| Training RMSE | `1.46637989e+01` |
| Validation RMSE | `2.93472466e+00` |
