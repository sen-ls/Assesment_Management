import numpy as np

def calculate_proportions_and_total_worth(num_stocks_A, price_A, num_stocks_B, price_B):
    total_worth_A = num_stocks_A * price_A
    total_worth_B = num_stocks_B * price_B
    total_worth = total_worth_A + total_worth_B
    proportion_A = total_worth_A / total_worth
    proportion_B = total_worth_B / total_worth
    return proportion_A, proportion_B, total_worth

def calculate_market_portfolio_return(proportion_A, return_A, proportion_B, return_B):
    return proportion_A * return_A + proportion_B * return_B

def calculate_market_portfolio_std_dev(proportion_A, std_dev_A, proportion_B, std_dev_B, correlation_AB):
    return np.sqrt((proportion_A * std_dev_A) ** 2 + (proportion_B * std_dev_B) ** 2 + 2 * proportion_A * proportion_B * std_dev_A * std_dev_B * correlation_AB)

def calculate_covariance_with_market(std_dev_stock, std_dev_other, correlation):
    return std_dev_stock * std_dev_other * correlation

def calculate_beta(covariance_with_market, std_dev_market):
    return covariance_with_market / std_dev_market ** 2

def main():
    num_stocks_A, price_A, return_A, std_dev_A = 5000, 12, 0.17, 0.16
    num_stocks_B, price_B, return_B, std_dev_B = 8000, 15, 0.13, 0.10
    correlation_AB = 0.5

    proportion_A, proportion_B, total_worth = calculate_proportions_and_total_worth(num_stocks_A, price_A, num_stocks_B, price_B)
    market_return = calculate_market_portfolio_return(proportion_A, return_A, proportion_B, return_B)
    market_std_dev = calculate_market_portfolio_std_dev(proportion_A, std_dev_A, proportion_B, std_dev_B, correlation_AB)
    
    covariance_with_market_A = calculate_covariance_with_market(std_dev_A, market_std_dev, correlation_AB)
    covariance_with_market_B = calculate_covariance_with_market(std_dev_B, market_std_dev, correlation_AB)
    
    beta_A = calculate_beta(covariance_with_market_A, market_std_dev)
    beta_B = calculate_beta(covariance_with_market_B, market_std_dev)

    # 输出结果
    print(f"Proportions of Stock A: {proportion_A}, Stock B: {proportion_B}, Total Worth: {total_worth}")
    print(f"Expected Return of Market Portfolio: {market_return}, Standard Deviation: {market_std_dev}")
    print(f"Betas - Stock A: {beta_A}, Stock B: {beta_B}")

if __name__ == "__main__":
    main()
