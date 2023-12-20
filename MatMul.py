import numpy as np

def matmul(A : np.ndarray, B : np.ndarray):
    max_i, max_k = A.shape
    max_k2, max_j = B.shape
    if max_k != max_k2:
        raise ValueError(f"Shape of A ({A.shape}) and B ({B.shape}) do not match!")
    
    dst = np.zeros((max_i, max_j), np.float64)
    for i in range(max_i):
        for j in range(max_j):
            for k in range(max_k):
                dst[i, j] += A[i, k] * B[k, j]
    return dst
