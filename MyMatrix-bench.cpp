#include <nanobench.h>

#include <chrono>

#include "MyMatrix.hpp"

using ankerl::nanobench::doNotOptimizeAway;

int main() {
  const size_t size = 64;
  float init[size][size];
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) init[i][j] = i;
  }
  MyMatrix<float> mat64(&init[0][0], size, size);

  ankerl::nanobench::Bench b;
  b.relative(true);
  b.minEpochTime(std::chrono::milliseconds{100});
  b.warmup(100);

  b.title("MatMul 64x64");
  b.run("matmul 64x64 slow", [&] { doNotOptimizeAway(mat64.mul(mat64)); });
  b.run("matmul 64x64 fast", [&] { doNotOptimizeAway(mat64 * mat64); });

  float init55[5][5];
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) init55[i][j] = i;
  }
  MyMatrix<float> mat55(&init55[0][0], 5, 5);

  b.title("MatMul 5x5");
  b.run("matmul 5x5 fast", [&] { doNotOptimizeAway(mat55 * mat55); });
  b.run("matmul 5x5 slow", [&] { doNotOptimizeAway(mat55.mul(mat55)); });
  b.run("matmul 5x5 simd", [&] { doNotOptimizeAway(matmul55(mat55, mat55)); });

  MyMatrix<float> kernel{{0, -1, 0}, {-1, 2, -1}, {0, -1, 0}};
  auto init512 = new float[512][512];
  for (int i = 0; i < 512; i++) {
    for (int j = 0; j < 512; j++) init512[i][j] = i;
  }
  MyMatrix<float> mat512(&init512[0][0], 512, 512);
  b.title("Conv 64x64 with 3x3 kernel");
  b.run("Conv 64x64 slow", [&] { doNotOptimizeAway(mat64.conv_slow(kernel)); });
  b.run("Conv 64x64 fast", [&] { doNotOptimizeAway(mat64.conv(kernel)); });

  b.title("Conv 512x512 with 3x3 kernel");
  b.run("Conv 512x512 slow",
        [&] { doNotOptimizeAway(mat512.conv_slow(kernel)); });
  b.run("Conv 512x512 fast", [&] { doNotOptimizeAway(mat512.conv(kernel)); });
}
