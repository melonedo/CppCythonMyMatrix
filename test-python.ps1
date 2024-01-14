echo Difference:
python -c "from MyMatrixTests import *; print(np.abs(A @ A - (B @ B).to_ndarray()).sum(), np.abs(A @ A - (B.mul(B)).to_ndarray()).sum(), np.abs(A @ A - matmul(A, A)).sum())"
echo NumPy:
python -m timeit -s "from MyMatrixTests import A" "A@A"
echo Cpp-Accelerated:
python -m timeit -s "from MyMatrixTests import B" "B@B"
echo Cpp-Slow:
python -m timeit -s "from MyMatrixTests import B" "B.mul_slow(B)"
echo Python:
python -m timeit -s "from MyMatrixTests import A, matmul" "matmul(A, A)"
