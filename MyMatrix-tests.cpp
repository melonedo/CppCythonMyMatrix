#include "MyMatrix.hpp"
#include "gtest/gtest.h"

template <typename T>
class ConstructorTest : public ::testing::Test {};

using TestTypes = ::testing::Types<int, float, double>;

TYPED_TEST_SUITE(ConstructorTest, TestTypes);

TYPED_TEST(ConstructorTest, DefaultConstructor) {
  MyMatrix<TypeParam> M;
  EXPECT_EQ(M.getCols(), 0);
  EXPECT_EQ(M.getRows(), 0);
}

TYPED_TEST(ConstructorTest, SizeConstructor) {
  MyMatrix<TypeParam> M{2, 3};
  EXPECT_EQ(M.getCols(), 3);
  EXPECT_EQ(M.getRows(), 2);
}

TYPED_TEST(ConstructorTest, BufferConstructor) {
  TypeParam buffer[] = {1, 2, 3, 4, 5, 6};
  MyMatrix<TypeParam> M{buffer, 2, 3};
  for (size_t r = 0; r < M.getRows(); r++) {
    for (size_t c = 0; c < M.getCols(); c++) {
      EXPECT_EQ(M.at(r, c), r * 3 + c + 1);
    }
  }
}

TYPED_TEST(ConstructorTest, InitListConstructor) {
  MyMatrix<TypeParam> m1{{1, 2, 3}, {4, 5, 6}};
  TypeParam buffer[] = {1, 2, 3, 4, 5, 6};
  MyMatrix<TypeParam> m2{buffer, 2, 3};
  EXPECT_EQ(m1, m2);
  EXPECT_THROW((MyMatrix<TypeParam>{{1, 2, 3}, {4, 6}}), ShapeError);
}

TYPED_TEST(ConstructorTest, CopyConstructor) {
  TypeParam buffer[] = {1, 2, 3, 4, 5, 6};
  MyMatrix<TypeParam> m1{buffer, 2, 3};
  MyMatrix<TypeParam> m2{m1};
  EXPECT_EQ(m2, m1);
}

TYPED_TEST(ConstructorTest, MoveConstructor) {
  TypeParam buffer[] = {1, 2, 3, 4, 5, 6};
  MyMatrix<TypeParam> m1{buffer, 2, 3};
  MyMatrix<TypeParam> m2{m1};
  MyMatrix<TypeParam> m3{std::move(m1)};
  EXPECT_EQ(m2, m3);
}

template <typename T>
class OperatorTest : public ::testing::Test {
 public:
  static auto constexpr approx = [](const MyMatrix<T>& a, const MyMatrix<T>& b,
                                    T tol) { return isapprox(a, b, tol); };
};

TYPED_TEST_SUITE(OperatorTest, TestTypes);

TYPED_TEST(OperatorTest, Assign) {
  MyMatrix<TypeParam> m1{{1, 2, 3}, {4, 5, 6}};
  MyMatrix<TypeParam> m2{{1, 2, 3}, {4, 5, 6}};
  auto m3 = m1;
  auto m4 = std::move(m1);
  EXPECT_EQ(m2, m3);
  EXPECT_EQ(m2, m4);
}

TYPED_TEST(OperatorTest, Add) {
  MyMatrix<TypeParam> m1{{1, 2, 3}, {4, 5, 6}};
  MyMatrix<TypeParam> m2{{6, 5, 4}, {3, 2, 1}};
  MyMatrix<TypeParam> m3{{7, 7, 7}, {7, 7, 7}};

  auto sum = m1 + m2;
  EXPECT_EQ(sum, m3);
  MyMatrix<TypeParam> m4 = m1;
  m4 += m2;
  EXPECT_EQ(sum, m4);
}

TYPED_TEST(OperatorTest, ScalerAdd) {
  MyMatrix<TypeParam> m1{{1, 2, 3}, {4, 5, 6}};
  MyMatrix<TypeParam> m5{{2, 3, 4}, {5, 6, 7}};
  auto m4 = m1;
  EXPECT_EQ(1 + m1, m5);
  EXPECT_EQ(m1 + 1, m5);
  m4 = m1;
  m4 += 1;
  EXPECT_EQ(m4, m5);
}

TYPED_TEST(OperatorTest, Sub) {
  MyMatrix<TypeParam> m1{{1, 2, 3}, {4, 5, 6}};
  MyMatrix<TypeParam> m2{{6, 5, 4}, {3, 2, 1}};
  MyMatrix<TypeParam> m3{{7, 7, 7}, {7, 7, 7}};
  MyMatrix<TypeParam> m4 = m3;
  auto diff = m3 - m1;
  EXPECT_EQ(diff, m2);
  m4 -= m2;
  EXPECT_EQ(m4, m1);
}

TYPED_TEST(OperatorTest, ScalarSub) {
  MyMatrix<TypeParam> m1{{1, 2, 3}, {4, 5, 6}};
  MyMatrix<TypeParam> m2{{6, 5, 4}, {3, 2, 1}};
  MyMatrix<TypeParam> m5{{2, 3, 4}, {5, 6, 7}};
  auto m4 = m1;
  EXPECT_EQ(m5 - 1, m1);
  EXPECT_EQ(7 - m1, m2);
  m4 = m5;
  m4 -= 1;
  EXPECT_EQ(m4, m1);
}

TYPED_TEST(OperatorTest, Negate) {
  MyMatrix<TypeParam> m1{{1, 2, 3}, {4, 5, 6}};
  MyMatrix<TypeParam> m2{{-1, -2, -3}, {-4, -5, -6}};
  EXPECT_EQ(-m1, m2);
}

TYPED_TEST(OperatorTest, Mul) {
  MyMatrix<TypeParam> m1{{1, 2, 3}, {4, 5, 6}};
  MyMatrix<TypeParam> m2{{6, 5}, {4, 3}, {2, 1}};
  MyMatrix<TypeParam> m3{{20, 14}, {56, 41}};
  MyMatrix<TypeParam> m5{{26, 37, 48}, {16, 23, 30}, {6, 9, 12}};
  auto prod1 = m1 * m2;
  auto prod2 = m1.mul(m2);
  auto m4 = m2;
  EXPECT_EQ(prod1, m3);
  EXPECT_EQ(prod2, m3);
  m4 *= m1;
  EXPECT_EQ(m5, m4);
}


TYPED_TEST(OperatorTest, Mul234) {
  MyMatrix<TypeParam> A{{-120, -104, 126}, {44, 43, 17}};
  MyMatrix<TypeParam> B{
      {60, -128, -64, -14}, {-127, -74, 66, -50}, {66, -122, -58, -25}};
  MyMatrix<TypeParam> RES{{14324, 7684, -6492, 3730},
                          {-1699, -10888, -964, -3191}};

  EXPECT_EQ(RES, A * B);
  EXPECT_EQ(RES, A.mul(B));
}

TYPED_TEST(OperatorTest, FastMul) {
  TypeParam init[81];
  for (int i = 0; i < 20; i++) {
    init[i * 4 + 0] = 1;
    init[i * 4 + 1] = 2;
    init[i * 4 + 2] = -1;
    init[i * 4 + 3] = 1;
  }
  init[80] = -1;
  MyMatrix<TypeParam> A{init, 9, 9};

  for (int i = 0; i < 20; i++) {
    init[i * 4 + 1] = -2;
    init[i * 4 + 2] = 1;
  }
  MyMatrix<TypeParam> B{init, 9, 9};

  TypeParam res_[81] = {
      -5, -2, 1,  13, -5, -2, 1,  13, -7, 14, -10, 2,  2,   14, -10, 2,  2,
      10, -1, 14, -7, -1, -1, 14, -7, -1, 1,  1,   -2, 13,  -5, 1,   -2, 13,
      -5, -1, -5, -2, 1,  13, -5, -2, 1,  13, -7,  14, -10, 2,  2,   14, -10,
      2,  2,  10, -1, 14, -7, -1, -1, 14, -7, -1,  1,  1,   -2, 13,  -5, 1,
      -2, 13, -5, -1, -7, 2,  -1, 11, -7, 2,  -1,  11, -5};
  MyMatrix<TypeParam> res{res_, 9, 9};

  auto prod1 = A.mul(B);
  auto prod2 = A * B;
  EXPECT_EQ(prod1, res);
  EXPECT_EQ(prod2, res);
}

TYPED_TEST(OperatorTest, ScalarMul) {
  MyMatrix<TypeParam> m1{{1, 2, 3}, {4, 5, 6}};
  MyMatrix<TypeParam> m6{{2, 4, 6}, {8, 10, 12}};
  auto prod2 = 2 * m1;
  auto prod3 = m1 * 2;
  EXPECT_EQ(prod2, m6);
  EXPECT_EQ(prod3, m6);
  m1 *= 2;
  EXPECT_EQ(m1, m6);
}

TYPED_TEST(OperatorTest, inverse) {
  if (!std::is_floating_point_v<TypeParam>) return;

  MyMatrix<TypeParam> m{{1, 0, 0}, {1, 2, 3}, {6, 5, 4}};
  MyMatrix<TypeParam> res{{7, 0, 0}, {-14, -4, 3}, {7, 5, -2}};
  res *= 1. / 7;
  auto inv = m.inv();

  EXPECT_PRED3(this->approx, res, inv, 0.001);
}

TYPED_TEST(OperatorTest, pseudoinverse) {
  if (!std::is_floating_point_v<TypeParam>) return;

  MyMatrix<TypeParam> m1{{1, 2, 3}, {4, 5, 6}};
  MyMatrix<TypeParam> m2 = m1.transpose();
  MyMatrix<TypeParam> I{{1, 0}, {0, 1}};

  EXPECT_PRED3(this->approx, m1 * m1.pinv(), I, 0.001);
  EXPECT_PRED3(this->approx, m2.pinv() * m2, I, 0.001);
}

TYPED_TEST(OperatorTest, transpose) {
  MyMatrix<TypeParam> m1{{1, 2, 3}, {4, 5, 6}};
  MyMatrix<TypeParam> m2{{1, 4}, {2, 5}, {3, 6}};
  EXPECT_EQ(m1.transpose(), m2);
}

TYPED_TEST(OperatorTest, conv) {
  MyMatrix<TypeParam> kernel{{-1, 2, -1}};
  MyMatrix<TypeParam> m2{{6}};
  EXPECT_EQ(kernel.conv(kernel), m2);
}

TYPED_TEST(OperatorTest, Exception) {
  MyMatrix<TypeParam> m1{{1, 2, 3}, {4, 5, 6}};
  MyMatrix<TypeParam> m8{{7, 7, 7}};

  EXPECT_THROW({ m1 + m8; }, ShapeError);
  EXPECT_THROW({ m1 - m8; }, ShapeError);
  EXPECT_THROW({ m1* m8; }, ShapeError);
}

TEST(SIMDTest, MatMul55) {
  float a_[25] = {39, -101, -105, 15,  -4, -104, -28, -1, 24,  54, 29,  70, 87,
                  99, 115,  -49,  -18, 82, -66,  65,  -7, -73, 58, -80, -77};
  float b_[25] = {31,  -4,  30,  59, 19,   27, -50, -37, -10, 77, -94, -72, 54,
                  -71, -65, -49, 70, -127, -3, 123, 118, -78, 72, 118, -13};
  float res_[25] = {7145,  13816, -2956,  10249, 1686,   1310,   -644,
                    -1298, 515,   -1817,  3330,  -11920, -1315,  8107,
                    10968, 1191,  -14498, 16686, -665,   -16610, -12806,
                    -92,   10239, -12647, -18363};
  MyMatrix<float> A(a_, 5, 5);
  MyMatrix<float> B(b_, 5, 5);
  MyMatrix<float> RES(res_, 5, 5);

  EXPECT_EQ(RES, A * B);
  EXPECT_EQ(RES, A.mul(B));
  EXPECT_EQ(RES, matmul55(A, B));
}
