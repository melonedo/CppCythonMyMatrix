#include <nanobench.h>
#include <chrono>

#include "MyMatrix.hpp"

int main() {
  const size_t size = 64;
  double init[size][size];
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) init[i][j] = i;
  }
  MyMatrix<double> mat(&init[0][0], size, size);

  ankerl::nanobench::Bench b;
  b.relative(true);
  b.minEpochTime(std::chrono::milliseconds{100});
  b.warmup(100);

  b.title("MatMul 64x64");
  b.run("matmul 64x64 slow",
        [&] { ankerl::nanobench::doNotOptimizeAway(mat.mul(mat)); });
  b.run("matmul 64x64 fast",
        [&] { ankerl::nanobench::doNotOptimizeAway(mat * mat); });

  float init55[5][5];
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) init55[i][j] = i;
  }
  MyMatrix<float> mat55(&init55[0][0], 5, 5);

  b.title("MatMul 5x5");
  b.run("matmul 5x5 fast",
        [&] { ankerl::nanobench::doNotOptimizeAway(mat55 * mat55); });
  b.run("matmul 5x5 slow",
        [&] { ankerl::nanobench::doNotOptimizeAway(mat55.mul(mat55)); });
  b.run("matmul 5x5 simd",
        [&] { ankerl::nanobench::doNotOptimizeAway(matmul55(mat55, mat55)); });
}
