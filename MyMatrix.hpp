#pragma once

#include <immintrin.h>
#include <stdlib.h>

#include <cassert>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef _MSC_VER
#define ALWAYS_INLINE __forceinline
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#define NOALIAS
#endif

// 乘法和加法在维度不匹配，或访问了不存在的元素时抛出
class ShapeError : public std::exception {
 public:
  ShapeError(const std::string &msg_) : msg(msg_) {}
  const char *what() const noexcept override { return msg.c_str(); }

 private:
  std::string msg;
};

class SingularMatrixError : public std::exception {
 public:
  SingularMatrixError(const std::string &msg_) : msg(msg_) {}
  const char *what() const noexcept override { return msg.c_str(); }

 private:
  std::string msg;
};

const static size_t cache_size = 32;

template <typename T>
class MatMulKernel {
 public:
  MatMulKernel(T *dst_, const T *A_, const T *B_, size_t max_i_, size_t max_j_,
               size_t max_k_)
      : dst(dst_), A(A_), B(B_), max_i(max_i_), max_j(max_j_), max_k(max_k_) {}

  void matmul(size_t i, size_t j, size_t k) const { kkernel(i, j, k); }

 private:
  void ALWAYS_INLINE kernel(size_t i, size_t j, size_t k) const {
    dst[i * max_j + j] += A[i * max_k + k] * B[k * max_j + j];
  }

  void jkernel(size_t i, size_t j, size_t k) const {
    if (Cj > max_j - j) {
      for (size_t j0 = 0; j0 < max_j - j; j0++) {
        kernel(i, j + j0, k);
      }
    } else {
      for (size_t j0 = 0; j0 < Cj; j0++) {
        kernel(i, j + j0, k);
      }
    }
  }

  void ikernel(size_t i, size_t j, size_t k) const {
    if (Ci > max_i - i) {
      for (size_t i0 = 0; i0 < max_i - i; i0++) {
        jkernel(i + i0, j, k);
      }
    } else {
      for (size_t i0 = 0; i0 < Ci; i0++) {
        jkernel(i + i0, j, k);
      }
    }
  }

  void kkernel(size_t i, size_t j, size_t k) const {
    if (Ck > max_k - k) {
      for (size_t k0 = 0; k0 < max_k - k; k0++) {
        ikernel(i, j, k + k0);
      }
    } else {
      for (size_t k0 = 0; k0 < Ck; k0++) {
        ikernel(i, j, k + k0);
      }
    }
  }

  T *const dst;
  const T *const A;
  const T *const B;
  const size_t max_i, max_j, max_k;
  const static size_t Ci = cache_size / sizeof(T);
  const static size_t Cj = Ci, Ck = Ci;
};

template <>
ALWAYS_INLINE void MatMulKernel<double>::jkernel(size_t i, size_t j,
                                                 size_t k) const {
  // max_j must also be a multiple of Cj
  if (Cj > max_j - j || max_j & (Cj - 1)) {
    const size_t Cj = cache_size / sizeof(double);  // ???
    for (size_t j0 = 0; j0 < std::min(Cj, max_j - j); j0++) {
      kernel(i, j + j0, k);
    }
  } else {
    __m256d a = _mm256_set1_pd(A[i * max_k + k]);
    __m256d b = _mm256_load_pd(&B[k * max_j + j]);
    __m256d c = _mm256_load_pd(&dst[i * max_j + j]);
    __m256d prod = _mm256_fmadd_pd(a, b, c);
    _mm256_store_pd(&dst[i * max_j + j], prod);
  }
}

template <>
ALWAYS_INLINE void MatMulKernel<float>::jkernel(size_t i, size_t j,
                                                size_t k) const {
  // max_j must also be a multiple of Cj
  if (Cj > max_j - j || max_j & (Cj - 1)) {
    const size_t Cj = cache_size / sizeof(float);  // ???
    for (size_t j0 = 0; j0 < std::min(Cj, max_j - j); j0++) {
      kernel(i, j + j0, k);
    }
  } else {
    __m256 a = _mm256_set1_ps(A[i * max_k + k]);
    __m256 b = _mm256_load_ps(&B[k * max_j + j]);
    __m256 c = _mm256_load_ps(&dst[i * max_j + j]);
    __m256 prod = _mm256_fmadd_ps(a, b, c);
    _mm256_store_ps(&dst[i * max_j + j], prod);
  }
}

template <typename T>
class MyMatrix final {
 private:
  size_t nrow, ncol;
  T *data;

 public:
  // 构造和析构
  MyMatrix() : MyMatrix(0, 0) {}
  ~MyMatrix() { dealloc(data); }

  // (复制|右值)(构造|赋值)
  MyMatrix(const MyMatrix &other) : MyMatrix(other.nrow, other.ncol) {
    for (int i = 0; i < nrow * ncol; i++) {
      data[i] = other.data[i];
    }
  }

  MyMatrix &operator=(const MyMatrix &other) {
    if (this == &other) return *this;
    nrow = other.nrow;
    ncol = other.ncol;
    dealloc(data);
    data = alloc(nrow * ncol);
    for (int i = 0; i < nrow * ncol; i++) {
      data[i] = other.data[i];
    }
    return *this;
  }

  MyMatrix(MyMatrix &&other) noexcept {
    nrow = other.nrow;
    ncol = other.ncol;
    data = other.data;
    other.nrow = other.ncol = 0;
    other.data = nullptr;
  }

  MyMatrix &operator=(MyMatrix &&other) noexcept {
    dealloc(data);
    nrow = other.nrow;
    ncol = other.ncol;
    data = other.data;
    other.nrow = other.ncol = 0;
    other.data = nullptr;
    return *this;
  }

  // 初始化列表构造: MyMatrix<T> m{{1,2}, {3,4}};
  MyMatrix(std::initializer_list<std::initializer_list<T>> ilist) {
    // 设定行数和列数
    nrow = ilist.size();
    ncol = ilist.begin()->size();
    data = alloc(nrow * ncol);

    auto &&row = ilist.begin();
    for (size_t r = 0; r < nrow; r++) {
      if (row->size() != ncol) {
        std::ostringstream msg;
        msg << "Row 0 has " << ncol << " elements, but row " << r << " has "
            << row->size() << " elements.";
        throw ShapeError(msg.str());
      }
      size_t c = 0;
      for (auto &&d : *row) get(r, c++) = d;
      ++row;
    }
  }

  // 方阵构造
  MyMatrix(size_t nrow_, size_t ncol_)
      : nrow(nrow_),
        ncol(ncol_),
        data(nrow_ && ncol_ ? alloc(nrow_ * ncol_) : nullptr) {}

  // 一般矩阵构造
  MyMatrix(const T *data_, size_t nrow_, size_t ncol_)
      : MyMatrix(nrow_, ncol_) {
    size_t size = nrow * ncol;
    for (size_t i = 0; i < size; i++) {
      data[i] = data_[i];
    }
  }

  // 运算符

  bool operator==(const MyMatrix &other) const noexcept {
    return isapprox(*this, other, {});
  }

  friend bool isapprox(const MyMatrix &a, const MyMatrix &b, T tol = {}) {
    if (a.ncol == b.ncol && a.nrow == b.nrow) {
      size_t size = a.ncol * a.nrow;
      if (tol != T{}) {
        for (size_t i = 0; i < size; i++) {
          if (abs(a.data[i] - b.data[i]) > tol) return false;
        }
      } else {
        for (size_t i = 0; i < size; i++) {
          if (a.data[i] != b.data[i]) return false;
        }
      }

      return true;
    } else {
      return false;
    }
  }

  MyMatrix &operator+=(const MyMatrix &other) {
    if (nrow == other.nrow && ncol == other.ncol) {
      for (size_t i = 0; i < nrow * ncol; i++) {
        data[i] += other.data[i];
      }
      return *this;
    } else {
      std::ostringstream msg;
      msg << "Adding (" << nrow << " x " << ncol << ") matrix by ("
          << other.nrow << " x " << other.ncol << ") matrix.";
      throw ShapeError(msg.str());
    }
  }

  MyMatrix &operator+=(T x) {
    for (size_t i = 0; i < nrow * ncol; i++) {
      data[i] += x;
    }
    return *this;
  }

  MyMatrix &operator-=(const MyMatrix &other) {
    if (nrow == other.nrow && ncol == other.ncol) {
      for (size_t i = 0; i < nrow * ncol; i++) {
        data[i] -= other.data[i];
      }
      return *this;
    } else {
      std::ostringstream msg;
      msg << "Subtracting (" << nrow << " x " << ncol << ") matrix by ("
          << other.nrow << " x " << other.ncol << ") matrix.";
      throw ShapeError(msg.str());
    }
  }

  MyMatrix &operator-=(T x) {
    for (size_t i = 0; i < nrow * ncol; i++) {
      data[i] -= x;
    }
    return *this;
  }

  MyMatrix &operator*=(T x) {
    for (size_t i = 0; i < nrow * ncol; i++) {
      data[i] *= x;
    }
    return *this;
  }

  MyMatrix operator-() const { return *this * (-1); }

  MyMatrix &operator*=(const MyMatrix &other) {
    *this = *this * other;
    return *this;
  }

  MyMatrix mul(const MyMatrix &other) const {
    if (ncol == other.nrow) {
      MyMatrix res{nrow, other.ncol};
      size_t size = nrow * other.ncol;
      for (size_t i = 0; i < size; i++) {
        res.data[i] = 0;
      }
      for (size_t i = 0; i < nrow; i++) {
        for (size_t j = 0; j < other.ncol; j++) {
          for (size_t k = 0; k < ncol; k++) {
            res.get(i, j) += this->get(i, k) * other.get(k, j);
          }
        }
      }
      return res;
    } else {
      std::ostringstream msg;
      msg << "Multiplying (" << nrow << " x " << ncol << ") matrix by ("
          << other.nrow << " x " << other.ncol << ") matrix.";
      throw ShapeError(msg.str());
    }
  }

  MyMatrix operator*(const MyMatrix &other) const {
    if (ncol == other.nrow) {
      MyMatrix res{nrow, other.ncol};
      size_t size = nrow * other.ncol;
      for (size_t i = 0; i < size; i++) {
        res.data[i] = 0;
      }
      const size_t C0 = cache_size / sizeof(T);
      const size_t C1 = C0, C2 = C0;

      const MatMulKernel<T> kernel(res.data, this->data, other.data, this->nrow,
                                   other.ncol, this->ncol);
      for (size_t j1 = 0; j1 < other.ncol; j1 += C2) {
        for (size_t k1 = 0; k1 < ncol; k1 += C1) {
          for (size_t i1 = 0; i1 < nrow; i1 += C0) {
            kernel.matmul(i1, j1, k1);
          }
        }
      }
      return res;
    } else {
      std::ostringstream msg;
      msg << "Multiplying (" << nrow << " x " << ncol << ") matrix by ("
          << other.nrow << " x " << other.ncol << ") matrix.";
      throw ShapeError(msg.str());
    }
  }

  // LU分解，对角线及以上为U，对角线以下加上全为1的对角线为L
  // inplace
  void lu(size_t *pivots, T tol = {}) {
    if (getRows() != getCols()) {
      std::ostringstream msg;
      msg << "LU decomposing (" << nrow << " x " << ncol << ") matrix";
      throw ShapeError(msg.str());
    }
    size_t m = getRows();
    for (size_t i = 0; i < m; i++) pivots[i] = i;

    for (size_t i = 0; i < m; i++) {
      // 寻找主元
      T max{};
      size_t p;
      for (size_t j = i; j < m; j++) {
        if (abs(get(j, i)) > max) {
          p = j;
          max = abs(get(j, i));
        }
      }
      if (max < tol) {
        throw SingularMatrixError(
            "Computing LU decompostion of singular matrix");
      }

      // 交换行
      if (i != p) {
        std::swap(pivots[i], pivots[p]);
        for (size_t j = 0; j < m; j++) {
          std::swap(get(i, j), get(p, j));
        }
      }

      // LU分解
      for (size_t j = i + 1; j < m; j++) {
        T f = get(j, i) / get(i, i);
        get(j, i) = f;
        for (size_t k = i + 1; k < m; k++) {
          get(j, k) -= f * get(i, k);
        }
      }
    }
  }

  MyMatrix inv(T tol = {}) const {
    MyMatrix lu = *this;
    std::vector<size_t> row_pivots(getRows());
    lu.lu(row_pivots.data(), tol);

    // 在lu()里已经检查是否方阵

    std::vector<size_t> col_pivots(getRows());
    for (size_t i = 0; i < getRows(); i++) {
      size_t j = row_pivots[i];
      col_pivots[j] = i;
    }

    MyMatrix res(this->getRows(), this->getCols());
    for (size_t i = 0; i < getRows() * getCols(); i++) {
      res.data[i] = 0;
    }

    // L \ I
    for (size_t j = 0; j < getCols(); j++) {
      size_t p = col_pivots[j];
      assert(p < getCols());

      res.get(p, j) = 1;

      // i < p 全是0
      for (size_t i = p + 1; i < getRows(); i++) {
        for (size_t k = p; k < i; k++) {
          res.get(i, j) -= lu.get(i, k) * res.get(k, j);
        }
      }
    }

    // U \ (L \ I)
    for (size_t j = 0; j < getCols(); j++) {
      for (size_t i = getRows() - 1; i != -1; i--) {
        for (size_t k = i + 1; k < getCols(); k++) {
          res.get(i, j) -= lu.get(i, k) * res.get(k, j);
        }
        res.get(i, j) /= lu.get(i, i);
      }
    }

    return res;
  }

  MyMatrix pinv() const {
    auto transposed = transpose();

    if (getRows() > getCols()) {
      return (transposed * *this).inv() * transposed;
    } else {
      return transposed * (*this * transposed).inv();
    }
  }

  MyMatrix transpose() const {
    MyMatrix res(getCols(), getRows());

    for (size_t i = 0; i < getRows(); i++) {
      for (size_t j = 0; j < getCols(); j++) {
        res.get(j, i) = this->get(i, j);
      }
    }

    return res;
  }

  MyMatrix conv_slow(const MyMatrix &kernel) const {
    ssize_t rows = getRows() - kernel.getRows() + 1;
    ssize_t cols = getCols() - kernel.getCols() + 1;
    if (rows <= 0 || cols <= 0) {
      return {};
    }
    MyMatrix res(rows, cols);

    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        T &x = res.get(i, j);
        x = 0;
        for (size_t ki = 0; ki < kernel.nrow; ki++) {
          for (size_t kj = 0; kj < kernel.ncol; kj++) {
            x += this->get(i + ki, j + kj) * kernel.get(ki, kj);
          }
        }
      }
    }

    return res;
  }

  MyMatrix conv(const MyMatrix &kernel) const {
    ssize_t rows = getRows() - kernel.getRows() + 1;
    ssize_t cols = getCols() - kernel.getCols() + 1;
    if (rows <= 0 || cols <= 0) {
      return {};
    }
    MyMatrix res(rows, cols);

    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        res.get(i, j) = 0;
      }
    }

    for (size_t i = 0; i < rows; i++) {
      for (size_t ki = 0; ki < kernel.nrow; ki++) {
        for (size_t kj = 0; kj < kernel.ncol; kj++) {
          convKernel(*this, res, kernel.get(ki, kj), ki, kj, i, cols);
        }
      }
    }

    return res;
  }

  // 二元转换为前缀运算
  MyMatrix operator+(const MyMatrix &other) const {
    return MyMatrix(*this) += other;
  }
  MyMatrix operator-(const MyMatrix &other) const {
    return MyMatrix(*this) -= other;
  }
  MyMatrix operator+(T x) const { return MyMatrix(*this) += x; }
  MyMatrix operator-(T x) const { return MyMatrix(*this) -= x; }
  MyMatrix operator*(T x) const { return MyMatrix(*this) *= x; }

  // 获取成员
  size_t getRows() const { return nrow; }
  size_t getCols() const { return ncol; }
  T *getData() const { return data; }

  T &at(size_t r, size_t c) {
    return const_cast<T &>(const_cast<const MyMatrix *>(this)->at(r, c));
  }
  const T &at(size_t r, size_t c) const {
    if (r < nrow && c < ncol) {
      return get(r, c);
    } else {
      std::ostringstream msg;
      msg << "Accessing (" << nrow << " x " << ncol << ") matrix at (" << r
          << ", " << c << ").";
      throw ShapeError(msg.str());
    }
  }

  // 输出流运算符 <<
  friend std::ostream &operator<<(std::ostream &os, const MyMatrix &mat) {
    size_t nrows = mat.getRows();
    size_t ncols = mat.getCols();
    for (size_t r = 0; r < nrows; r++) {
      os << "[ ";
      for (size_t c = 0; c < ncols; c++) {
        os << mat.get(r, c) << '\t';
      }
      os << "]\n";
    }
    return os;
  }

  // 转换为字符串
  std::string toString() const {
    std::ostringstream oss;
    oss << *this;
    return oss.str();
  }

  // 数字与矩阵运算
  template <typename S>
  friend MyMatrix operator*(S x, const MyMatrix &mat) {
    return mat * static_cast<T>(x);
  }

  template <typename S>
  friend MyMatrix operator+(S x, const MyMatrix &mat) {
    return mat + static_cast<T>(x);
  }

  template <typename S>
  friend MyMatrix operator-(S x, const MyMatrix &mat) {
    return -mat + static_cast<T>(x);
  }

#ifdef _MSC_VER
  static T *alloc(size_t len) {
    return static_cast<T *>(_aligned_malloc(sizeof(T) * len, cache_size));
  }

  static void dealloc(T *ptr) { _aligned_free(ptr); }
#else
  static T *alloc(size_t len) {
    return static_cast<T *>(aligned_alloc(cache_size, sizeof(T) * len));
  }

  static void dealloc(T *ptr) { free(ptr); }
#endif

 private:
  const T &get(size_t r, size_t c) const {
    assert(r < nrow && c < ncol);
    return data[r * ncol + c];
  }

  T &get(size_t r, size_t c) {
    return const_cast<T &>(const_cast<const MyMatrix *>(this)->get(r, c));
  }

  static void convKernel(const MyMatrix<T> &src, MyMatrix<T> &dst, T k,
                         size_t ki, size_t kj, size_t i, size_t cols) {
    for (size_t j = 0; j < cols; j++) {
      dst.get(i, j) += src.get(i + ki, j + kj) * k;
    }
  }
};

template <>
ALWAYS_INLINE void MyMatrix<float>::convKernel(const MyMatrix<float> &src,
                                               MyMatrix<float> &dst, float k,
                                               size_t ki, size_t kj, size_t i,
                                               size_t cols) {
  size_t j = 0;
  const size_t stride = 32 / sizeof(float);
  const float *src_base = &src.get(i + ki, kj);
  float *dst_base = &dst.get(i, 0);
  for (; j + stride <= cols; j += stride) {
    __m256 k8 = _mm256_set1_ps(k);
    __m256 a8 = _mm256_loadu_ps(src_base + j);
    __m256 b8 = _mm256_loadu_ps(dst_base + j);
    __m256 prod = _mm256_fmadd_ps(a8, k8, b8);
    _mm256_storeu_ps(&dst.get(i, j), prod);
  }
  for (; j < cols; j++) {
    dst.get(i, j) += src.get(i + ki, j + kj) * k;
  }
}

template <>
ALWAYS_INLINE void MyMatrix<double>::convKernel(const MyMatrix<double> &src,
                                                MyMatrix<double> &dst, double k,
                                                size_t ki, size_t kj, size_t i,
                                                size_t cols) {
  size_t j = 0;
  const size_t stride = 32 / sizeof(double);
  const double *src_base = &src.get(i + ki, kj);
  double *dst_base = &dst.get(i, 0);
  for (; j + stride <= cols; j += stride) {
    __m256d k8 = _mm256_set1_pd(k);
    __m256d a8 = _mm256_loadu_pd(src_base + j);
    __m256d b8 = _mm256_loadu_pd(dst_base + j);
    __m256d prod = _mm256_fmadd_pd(a8, k8, b8);
    _mm256_storeu_pd(&dst.get(i, j), prod);
  }
  for (; j < cols; j++) {
    dst.get(i, j) += src.get(i + ki, j + kj) * k;
  }
}

extern template class MyMatrix<int>;
extern template class MyMatrix<float>;
extern template class MyMatrix<double>;
MyMatrix<float> matmul55(const MyMatrix<float> &a, const MyMatrix<float> &b);
