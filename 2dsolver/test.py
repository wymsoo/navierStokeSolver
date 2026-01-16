from scipy.sparse import diags, eye, kron, csr_matrix
import numpy as np

A = np.array([[1,1,1],
     [1,1,1],
     [1,1,1],
     [1,2,1]])

B = np.array([[1],
     [2],
     [3],
     [1]])

zero = np.zeros((A.shape[0],1))
print(np.concatenate((A,zero),axis=1))