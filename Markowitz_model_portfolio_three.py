import numpy as np

def invert_covariance_matrix(cov_matrix):
    return np.linalg.inv(cov_matrix)

def minimum_variance_portfolio(cov_matrix):
    inv_cov_matrix = invert_covariance_matrix(cov_matrix)
    one_vector = np.ones(len(cov_matrix))
    weights = np.dot(inv_cov_matrix, one_vector) / np.dot(one_vector, np.dot(inv_cov_matrix, one_vector))
    return weights

def portfolio_standard_deviation(variance):
    # Calculate the standard deviation, which is the square root of the variance
    return np.sqrt(variance)


def portfolio_return(weights, returns):
    return np.dot(weights, returns)

def portfolio_variance(weights, cov_matrix):
    return np.dot(weights, np.dot(cov_matrix, weights))

def given_return_portfolio_variance(target_return, returns, cov_matrix):
    inv_cov_matrix = invert_covariance_matrix(cov_matrix)
    one_vector = np.ones(len(returns))
    A = np.dot(one_vector, np.dot(inv_cov_matrix, one_vector))
    B = np.dot(returns, np.dot(inv_cov_matrix, one_vector))
    C = np.dot(returns, np.dot(inv_cov_matrix, returns))
    lambda_val = (C - target_return * B) / (A * C - B ** 2)
    gamma_val = (target_return - A * lambda_val) / B
    weights = lambda_val * np.dot(inv_cov_matrix, one_vector) + gamma_val * np.dot(inv_cov_matrix, returns)
    return portfolio_variance(weights, cov_matrix)

def tangent_portfolio(cov_matrix, risk_free_rate, returns):
    inv_cov_matrix = invert_covariance_matrix(cov_matrix)
    excess_returns = returns - risk_free_rate
    weights = np.dot(inv_cov_matrix, excess_returns) / np.dot(np.ones(len(cov_matrix)), np.dot(inv_cov_matrix, excess_returns))
    return weights, portfolio_return(weights, returns), portfolio_variance(weights, cov_matrix)

def main():
    # Given data
    returns = np.array([0.15, 0.10, 0.08])
    cov_matrix = np.array([[0.6400, -0.100, -0.1800], [-0.100, 0.3000, 0.0500], [-0.1800, 0.0500, 0.1200 ]])
    risk_free_rate = 0.05  # Example risk-free rate
    target_portfolio_return = 0.15  # Given expected return

    # 1. Calculate the inverse of the variance-covariance matrix
    inv_cov_matrix = invert_covariance_matrix(cov_matrix)

    # 2. Calculate the expected return and variance of the portfolio with the minimum variance
    min_var_weights = minimum_variance_portfolio(cov_matrix)
    min_var_portfolio_return = portfolio_return(min_var_weights, returns)
    min_var_portfolio_variance = portfolio_variance(min_var_weights, cov_matrix)
    
    # Calculate and print the standard deviation for the minimum variance portfolio
    min_var_portfolio_std_dev = portfolio_standard_deviation(min_var_portfolio_variance)


    # 3. Calculate the variance given the expected return of the portfolio
    given_return_variance = given_return_portfolio_variance(target_portfolio_return, returns, cov_matrix)

    # 4. Calculate the Tangent Portfolio (TPF)
    tp_weights, tp_return, tp_variance = tangent_portfolio(cov_matrix, risk_free_rate, returns)

    # Output results
    print("Inverse Covariance Matrix:", inv_cov_matrix)
    print("Minimum Variance Portfolio Weights:", min_var_weights)
    print("Minimum Variance Portfolio Expected Return:", min_var_portfolio_return)
    print("Minimum Variance Portfolio Variance:", min_var_portfolio_variance)
    print("Minimum Variance Portfolio Standard Deviation:", min_var_portfolio_std_dev)
    print("Given Return Portfolio Variance:", given_return_variance)
    print("Tangent Portfolio Weights:", tp_weights)
    print("Tangent Portfolio Expected Return:", tp_return)
    print("Tangent Portfolio Variance:", tp_variance)

if __name__ == "__main__":
    main()
