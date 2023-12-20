import numpy as np
from MyMatrixWrapper import Matrix
from MatMul import matmul

A = np.arange(0, 64 * 64, dtype="d").reshape((64, 64))
B = Matrix.from_ndarray(A)

if __name__ == "__main__":
    mat1 = Matrix.from_ndarray(np.ones((2, 3), dtype=float))
    print(mat1)
    print(repr(1 + mat1))
    print((1.3 + mat1).to_ndarray())
    # print([]+mat1)
    print((A @ A - (B @ B).to_ndarray()).sum())
    print((A @ A - matmul(A, A)).sum())
