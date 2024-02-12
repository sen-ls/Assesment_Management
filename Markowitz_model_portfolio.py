import numpy as np

# Define functions for the calculations
def portfolio_expected_return(weights, returns):
    return np.dot(weights, returns)

def portfolio_variance(weights, cov_matrix):
    return np.dot(weights, np.dot(cov_matrix, weights))

def minimum_variance_portfolio(variances, cov):
    inv_cov_matrix = np.linalg.inv(np.array([[variances[0], cov], [cov, variances[1]]]))
    one_vector = np.array([1, 1])
    weights = np.dot(inv_cov_matrix, one_vector) / np.dot(one_vector, np.dot(inv_cov_matrix, one_vector))
    return weights

def given_return_portfolio_variance(target_return, returns, variances, cov):
    weights = minimum_variance_portfolio(variances, cov)
    min_return = portfolio_expected_return(weights, returns)
    if target_return == min_return:
        return portfolio_variance(weights, np.array([[variances[0], cov], [cov, variances[1]]]))
    
    excess_returns = returns - min_return
    inv_cov_matrix = np.linalg.inv(np.array([[variances[0], cov], [cov, variances[1]]]))
    a = np.dot(excess_returns, np.dot(inv_cov_matrix, excess_returns))
    b = np.dot(excess_returns, np.dot(inv_cov_matrix, np.ones(2)))
    c = np.dot(np.ones(2), np.dot(inv_cov_matrix, np.ones(2)))
    
    lambda_1 = (c * target_return - b) / (a * c - b ** 2)
    lambda_2 = (a - b * target_return) / (a * c - b ** 2)
    
    weights = lambda_1 * np.dot(inv_cov_matrix, excess_returns) + lambda_2 * np.dot(inv_cov_matrix, np.ones(2))
    return portfolio_variance(weights, np.array([[variances[0], cov], [cov, variances[1]]]))

def risk_efficient_frontier(returns, variances, cov, num_points=100):
    min_return = np.min(returns)
    max_return = np.max(returns)
    target_returns = np.linspace(min_return, max_return, num_points)
    variances = [given_return_portfolio_variance(target_return, returns, variances, cov) for target_return in target_returns]
    return target_returns, variances

# Main function for executing defined functions
def main():
    # Expected returns, variances, and covariance
    returns = np.array([0.2, 0.3])
    variances = np.array([0.04, 0.08])
    cov = 0.02

    # 1. Minimum variance portfolio
    weights_min_var = minimum_variance_portfolio(variances, cov)
    min_var_return = portfolio_expected_return(weights_min_var, returns)
    min_var_variance = portfolio_variance(weights_min_var, np.array([[variances[0], cov], [cov, variances[1]]]))
    print(f"Minimum Variance Portfolio Weights: {weights_min_var}")
    print(f"Expected Return: {min_var_return}")
    print(f"Variance: {min_var_variance}")

    # 2. Portfolio variance given a target return
    target_return = 0.25  # Example target return
    target_var = given_return_portfolio_variance(target_return, returns, variances, cov)
    print(f"Variance for a portfolio with expected return {target_return}: {target_var}")

    # 3. Risk-efficient frontier
    efficient_returns, efficient_variances = risk_efficient_frontier(returns, variances, cov)
    print(f"Efficient Frontier Returns: {efficient_returns}")
    print(f"Efficient Frontier Variances: {efficient_variances}")

if __name__ == "__main__":
    main()
