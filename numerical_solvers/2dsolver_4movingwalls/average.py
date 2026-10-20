import numpy as np 

def average(array: np.ndarray, axis: int) -> np.ndarray:

    if axis == 0:  # x-direction
        return 0.5 * (array[1:, :] + array[:-1, :])
    elif axis == 1:  # y-direction
        return 0.5 * (array[:, 1:] + array[:, :-1])
    else:
        raise ValueError("axis must be 0, or 1")
    
