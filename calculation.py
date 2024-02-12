import numpy as np
from sympy import symbols, diff

def invert_matrices(matrices):
    # 用于计算一个或多个矩阵的逆
    inverses = [np.linalg.inv(matrix) for matrix in matrices]
    return inverses

def differentiate_function(function, symbol):
    # 对给定的函数求导
    derivative = diff(function, symbol)
    return derivative

def main():
    # 定义两个矩阵
    A = np.array([[3, -1],
                  [1, -2]])
    B = np.array([[1, 1, 2],
                  [-2, 3, 3],
                  [1, 2, 5]])
    
    # 计算矩阵的逆
    A_inv, B_inv = invert_matrices([A, B])
    
    # 打印矩阵的逆
    print("Inverse of A:")
    print(A_inv)
    print("\nInverse of B:")
    print(B_inv)
    
    # 求导
    x = symbols('x')
    f = (x**2 + 2*x) / x**3
    df = differentiate_function(f, x)
    
    # 打印求导结果
    print("\nDerivative of fx with respect to x:")
    print(df)

if __name__ == "__main__":
    main()
