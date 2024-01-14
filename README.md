# 矩阵乘法加速作业

系统编程2023作业，基本的矩阵类，包括SIMD乘法及Cython生成的Python接口。

## 编译及运行

```shell
# 安装
pip install 'Cython>=3.0' numpy torch jupyter Pillow
## 从GitHub下载gtest和nanobench并配置
cmake -B build

# C++部分
cmake --build build --config Release
## 功能测试
build/Release/MyMatrix-tests
## 速度测试
build/Release/MyMatrix-bench

# Python部分
cmake --build build --config Release -t CopyMyMatrixWrapper
## 速度测试
./test-python.ps1
```

## 测试配置

- CPU: AMD Ryzen 7 7735HS
- RAM: 32GB 6400MHz

## C++矩阵乘法加速

```shell
build/Release/MyMatrix-bench
```

|   relative  |               ns/op |                op/s |    err% |     total | benchmark
|------------:|--------------------:|--------------------:|--------:|----------:|:----------
|     100.0%  |          663,624.57 |            1,506.88 |    0.3% |      1.22 | `matmul 64x64 slow`
| **1,664.2%**|           39,875.88 |           25,077.82 |    0.4% |      1.21 | `matmul 64x64 fast`

C++中加速实现相比简单实现加速比为16.6。

## 与Python和NumPy比较

```shell
PS C:\Code\MyMatrix> ./test-python.ps1
Difference:
19712.0 19712.0 84432.0
NumPy:
50000 loops, best of 5: 8.31 usec per loop
Cpp-Accelerated:
10000 loops, best of 5: 39.1 usec per loop
Cpp-Slow:
500 loops, best of 5: 651 usec per loop
Python:
5 loops, best of 5: 84.2 msec per loop
```
由于使用的矩阵中最大项的大小已经达到5x10^8，32位浮点数有一定误差，因此C++结果和NumPy不完全一致。
可见C++实现和纯Python实现均和NumPy结果相同，以Python为基准，各方法速度：

| 实现    | NumPy | Cpp加速 | Cpp简单 | Python |
|---------|-------|---------|---------|--------|
| 用时/μs | 8.31  | 39.1    | 651     | 84200  |
| 相对速度 | 10132 | 2153    | 129     | 1      |

## 5x5矩阵加速

| relative |               ns/op |                op/s |    err% |     total | benchmark
|---------:|--------------------:|--------------------:|--------:|----------:|:-----------
|   100.0% |              193.06 |        5,179,750.13 |    0.4% |      1.21 | `matmul 5x5 fast`
|    89.2% |              216.40 |        4,621,121.39 |    1.0% |      1.22 | `matmul 5x5 slow`
|   200.1% |               96.48 |       10,364,738.77 |    0.6% |      1.15 | `matmul 5x5 simd`

专门设计的5x5 SIMD 矩阵加速比通用实现快大约两倍。
