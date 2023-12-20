#include <cassert>

#include "MyMatrix.hpp"

template class MyMatrix<int>;
template class MyMatrix<double>;
template class MyMatrix<float>;

#ifdef _MSC_VER
#define _m256_extract_ps(v, i) v.m256_f32[i]
#define _mm_extract_ps(v, i) v.m128_f32[i]
#else
#define _m256_extract_ps(v, i) v[i]
// GCC的实现是错的，WTF？
#ifdef _mm_extract_ps
#undef _mm_extract_ps
#endif
#define _mm_extract_ps(v, i) v[i]
#endif

static ALWAYS_INLINE void scatter(float* dst, __m256i vindex, __m256 v) {
#if defined(__AVX512F__) && defined(__AVX512VL__)
  _mm256_i32scatter_ps(dst, vindex, v, sizeof(float));
#elif _MSC_VER
  for (int i = 0; i < 8; i++) {
    dst[vindex.m256i_i32[i]] = v.m256_f32[i];
  }
#else
  union {
    __m256i v;
    int i32[8];
  } vindex_;
  vindex_.v = vindex;
  for (int i = 0; i < 8; i++) {
    dst[vindex_.i32[i]] = v[i];
  }
#endif
}

static ALWAYS_INLINE __m256 repeat(__m128 v) {
  return _mm256_insertf128_ps(_mm256_castps128_ps256(v), v, 1);
}

MyMatrix<float> matmul55(const MyMatrix<float>& a, const MyMatrix<float>& b) {
  assert(a.getRows() == 5 && a.getCols() == 5 && b.getRows() == 5 &&
         b.getCols() == 5);

  MyMatrix<float> dst(5, 5);

  const __m128i col0index = _mm_set_epi32(15, 10, 5, 0);
  const __m256i row01index = _mm256_set_epi32(8, 7, 6, 5, 3, 2, 1, 0);
  const uint8_t dp_mask = 0b11110001;

  const float* pa = a.getData();
  const float* pb = b.getData();
  float* pd = dst.getData();

  // a[0:4, 0:4] * b[0:4, 0:5]
  for (size_t i = 0; i < 4; i += 2) {
    for (size_t j = 0; j < 5; j++) {
      __m256 a01 = _mm256_i32gather_ps(&pa[i * 5], row01index, sizeof(float));
      __m128 b0 = _mm_i32gather_ps(&pb[j], col0index, sizeof(float));
      __m256 b00 = repeat(b0);
      __m256 dp = _mm256_dp_ps(a01, b00, dp_mask);
      pd[i * 5 + j] = _m256_extract_ps(dp, 0);
      pd[(i + 1) * 5 + j] = _m256_extract_ps(dp, 4);
    }
  }

  // a[4, 0:4] * b[0:4, 0:5]
  {
    size_t i = 4;
    __m128 a4 = _mm_loadu_ps(&pa[i * 5]);
    for (size_t j = 0; j < 4; j += 2) {
      __m256i col01index = _mm256_set_epi32(16, 11, 6, 1, 15, 10, 5, 0);
      __m256 b01 = _mm256_i32gather_ps(&pb[j], col01index, sizeof(float));
      __m256 a44 = repeat(a4);
      __m256 dp = _mm256_dp_ps(a44, b01, dp_mask);
      pd[i * 5 + j] = _m256_extract_ps(dp, 0);
      pd[i * 5 + j + 1] = _m256_extract_ps(dp, 4);
    }
    {
      size_t j = 4;
      __m128 b4 = _mm_i32gather_ps(&pb[j], col0index, sizeof(float));
      __m128 dp = _mm_dp_ps(a4, b4, dp_mask);
      pd[i * 5 + j] = _mm_extract_ps(dp, 0);
    }
  }

  // a[0:4, 4] * b[4, 0:4]
  {
    size_t k = 4;
    __m128 b0 = _mm_loadu_ps(&pb[k * 5]);
    __m256 b00 = repeat(b0);
    for (size_t i = 0; i < 4; i += 2) {
      __m128 a0 = _mm_set1_ps(pa[i * 5 + k]);
      __m128 a1 = _mm_set1_ps(pa[(i + 1) * 5 + k]);
      __m256 a01 = _mm256_insertf128_ps(_mm256_castps128_ps256(a0), a1, 1);
      __m256 d01 = _mm256_i32gather_ps(&pd[i * 5], row01index, sizeof(float));
      __m256 prod = _mm256_fmadd_ps(a01, b00, d01);
      scatter(&pd[i * 5], row01index, prod);
      // _mm_storeu_ps(&pd[i * 5], _mm256_extractf128_ps(prod, 0));
      // _mm_storeu_ps(&pd[(i + 1) * 5], _mm256_extractf128_ps(prod, 1));
    }
  }

  // a[4, 4] * b[4, 0:4], a[0:4, 4] * b[4, 4]
  {
    __m256i aindex = _mm256_set_epi32(19, 14, 9, 4, 24, 24, 24, 24);
    __m256i bindex = _mm256_set_epi32(24, 24, 24, 24, 23, 22, 21, 20);
    __m256i dindex = _mm256_set_epi32(19, 14, 9, 4, 23, 22, 21, 20);
    __m256 aa = _mm256_i32gather_ps(pa, aindex, sizeof(float));
    __m256 bb = _mm256_i32gather_ps(pb, bindex, sizeof(float));
    __m256 dd = _mm256_i32gather_ps(pd, dindex, sizeof(float));
    __m256 prod = _mm256_fmadd_ps(aa, bb, dd);
    scatter(pd, dindex, prod);
  }

  pd[24] += pa[24] * pb[24];

  return dst;
}
