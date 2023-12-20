#cython: language_level=3

from libcpp cimport bool
from libcpp.string cimport string

cdef extern from "MyMatrix.hpp" nogil:
    cdef cppclass MyMatrix[T]:
        MyMatrix() noexcept
        MyMatrix(MyMatrix[T]) noexcept
        MyMatrix(int, int) noexcept
        MyMatrix(T*, int, int) noexcept
        size_t getRows() noexcept
        size_t getCols() noexcept
        T *getData() noexcept
        string toString() noexcept

        T at(size_t, size_t) except +

        bool operator==(MyMatrix[T]) noexcept
        MyMatrix[T] operator+(MyMatrix[T]) except +
        MyMatrix[T] operator-(MyMatrix[T]) except +
        MyMatrix[T] operator*(MyMatrix[T]) except +
        MyMatrix[T] mul(MyMatrix[T]) except +
        MyMatrix[T] operator+(double) except +
        MyMatrix[T] operator-(double) except +
        MyMatrix[T] operator*(double) except +
    
    MyMatrix[double] operator-(double, MyMatrix[double]) except +
