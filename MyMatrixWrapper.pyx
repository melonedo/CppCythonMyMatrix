#cython: language_level=3
#distutils: language = c++
#distutils: define_macros = NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from MyMatrixWrapper cimport MyMatrix
import numpy as np
cimport numpy as cnp
cimport cython

ctypedef fused NumOrMat:
    cython.integral
    cython.floating
    Matrix

cdef class Matrix:
    cdef MyMatrix[double] mat

    def __init__(self, int rows = 0, int cols = 0):
        self.mat = MyMatrix[double](rows, cols)

    @staticmethod
    def from_ndarray(cnp.ndarray[double, ndim=2] arr):
        cdef Matrix mat
        mat = Matrix()
        arr = np.ascontiguousarray(arr)
        mat.mat = MyMatrix[double](&arr[0, 0], arr.shape[0], arr.shape[1])
        return mat

    def to_ndarray(self):
        view = <double[:self.mat.getRows(), :self.mat.getCols()]> self.mat.getData()
        return np.asarray(view).copy()

    def __str__(self):
        # [:-1] to get rid of '\n' in the end
        # return self.mat.toString().decode()[:-1]
        return str(self.to_ndarray())

    def __repr__(self):
        return f"Matrix.from_ndarray(\n{repr(self.to_ndarray())})"

    def __getitem__(self, (size_t, size_t) i):
        return self.mat.at(i[0], i[1])

    def __eq__(self, Matrix other):
        return self.mat == other.mat

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def add(self, NumOrMat other):
        ret = Matrix()
        if NumOrMat in cython.numeric:
            ret.mat = self.mat + other
        elif NumOrMat is Matrix:
            ret.mat = self.mat + other.mat
        return ret

    def __sub__(self, other):
        return self.sub(other)

    def sub(self, NumOrMat other):
        ret = Matrix()
        if NumOrMat in cython.numeric:
            ret.mat = self.mat - other
        elif NumOrMat is Matrix:
            ret.mat = self.mat - other.mat
        return ret

    def __rsub__(self, other):
        return self.rsub(other)

    def rsub(self, NumOrMat other):
        ret = Matrix()
        if NumOrMat in cython.numeric:
            ret.mat = other - self.mat
        elif NumOrMat is Matrix:
            ret.mat = other.mat - self.mat
        return ret

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)
    
    def __matmul__(self, other):
        return self.mul(other)

    def mul(self, NumOrMat other):
        ret = Matrix()
        if NumOrMat in cython.numeric:
            ret.mat = self.mat * other
        elif NumOrMat is Matrix:
            ret.mat = self.mat * other.mat
        return ret

    def slow_mul(self, Matrix other):
        ret = Matrix()
        ret.mat = self.mat.mul(other.mat)
        return ret
