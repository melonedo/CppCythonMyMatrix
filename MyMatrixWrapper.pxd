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
        MyMatrix[T] operator+(T) except +
        MyMatrix[T] operator-(T) except +
        MyMatrix[T] operator*(T) except +
        MyMatrix[T] inv(T) except +
        MyMatrix[T] conv(MyMatrix[T]) noexcept
        MyMatrix[T] conv_slow(MyMatrix[T]) noexcept
    
    MyMatrix[double] operator-(double, MyMatrix[double]) except +
    MyMatrix[float] operator-(float, MyMatrix[float]) except +
