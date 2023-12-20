# 矩阵乘法加速作业

系统编程2023作业，基本的矩阵类，包括SIMD乘法及Cython生成的Python接口。

## 编译及运行

```shell
# 安装
## 从GitHub下载gtest和nanobench并配置
pip install 'Cython>=3.0' numpy
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
|    100.0%   |          643,806.74 |            1,553.26 |    0.2% |     12.10 | `matmul 64x64 slow`
|  **735.4%** |           87,548.42 |           11,422.25 |    0.2% |     11.81 | `matmul 64x64 fast`

C++中加速实现相比简单实现加速比为7.35。

## 与Python和NumPy比较

```shell
PS C:\Code\MyMatrix> ./test-python.ps1
Difference:
0.0 0.0 0.0
NumPy:
20000 loops, best of 5: 12.4 usec per loop
Cpp-Accelerated:
5000 loops, best of 5: 91.8 usec per loop
Cpp-Slow:
500 loops, best of 5: 651 usec per loop
Python:
5 loops, best of 5: 84.2 msec per loop
```

可见C++实现和纯Python实现均和NumPy结果相同，以Python为基准，各方法速度：

| 实现    | NumPy | Cpp加速 | Cpp简单 | Python |
|---------|-------|---------|---------|--------|
| 用时/μs | 12.4  | 91.8    | 651     | 84200  |
| 相对速度 | 6740  | 917     | 129     | 1      |
