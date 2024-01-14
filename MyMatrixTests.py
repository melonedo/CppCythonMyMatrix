import numpy as np
from MyMatrixWrapper import Matrix
from MatMul import matmul

size = 64
A = np.arange(0, size * size, dtype=np.float32).reshape((size, size))
B = Matrix.from_ndarray(A)

if __name__ == "__main__":
    mat1 = Matrix.from_ndarray(np.ones((2, 3), dtype=np.float32))
    print(mat1)
    print(repr(1 + mat1))
    print((1.3 + mat1).to_ndarray())
    # print([]+mat1)
    print(np.abs(A @ A - (B @ B).to_ndarray()).sum())
    print(np.abs(A @ A - matmul(A, A)).sum())
    kernel = Matrix.from_ndarray(np.array([[-1, 2, 1]], dtype=np.float32))
    print(kernel.conv(kernel))
