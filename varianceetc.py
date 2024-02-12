import numpy as np

def calculate_mean(data):
    return np.mean(data)

def calculate_variance(data):
    return np.var(data)

def calculate_covariance(data1, data2):
    covariance_matrix = np.cov(data1, data2)
    return covariance_matrix[0, 1]

def calculate_correlation(data1, data2):
    correlation_matrix = np.corrcoef(data1, data2)
    return correlation_matrix[0, 1]

def main():
    # 定义两组数据
    data1 = np.array([1, 2, 3])
    data2 = np.array([2, 3, 4])

    # 计算均值
    mean1 = calculate_mean(data1)
    mean2 = calculate_mean(data2)
    print(f"Mean of data1: {mean1}")
    print(f"Mean of data2: {mean2}")

    # 计算方差
    variance1 = calculate_variance(data1)
    variance2 = calculate_variance(data2)
    print(f"Variance of data1: {variance1}")
    print(f"Variance of data2: {variance2}")

    # 计算协方差
    covariance = calculate_covariance(data1, data2)
    print(f"Covariance between data1 and data2: {covariance}")

    # 计算相关系数
    correlation = calculate_correlation(data1, data2)
    print(f"Correlation between data1 and data2: {correlation}")

# 运行主函数
if __name__ == "__main__":
    main()