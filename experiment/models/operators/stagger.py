import numpy as np

# def stagger_back(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
def stagger_back(U: np.ndarray, V: np.ndarray):
    if U.ndim < 2 or V.ndim < 2:
        raise ValueError("U and V must have at least two spatial dimensions")
    if U.shape[:-2] != V.shape[:-2]:
        raise ValueError("U and V must have matching leading dimensions")
    if U.shape[-2] + 1 != V.shape[-2] or U.shape[-1] != V.shape[-1] + 1:
        raise ValueError("U and V do not have the expected staggered grid shapes")

    u_padding = [(0, 0)] * U.ndim
    u_padding[-2] = (1, 1)
    U = np.pad(U, u_padding, mode="constant")
    U = 0.5 * (U[..., 1:, :] + U[..., :-1, :])

    v_padding = [(0, 0)] * V.ndim
    v_padding[-1] = (1, 1)
    V = np.pad(V, v_padding, mode="constant")
    V = 0.5 * (V[..., 1:] + V[..., :-1])

    return U, V
