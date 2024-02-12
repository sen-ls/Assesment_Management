import numpy as np

# 定义计算函数
def calculate_sharpe_ratio(fund_return, std_dev, risk_free_rate):
    return (fund_return - risk_free_rate) / std_dev

def calculate_treynor_ratio(fund_return, beta, risk_free_rate):
    return (fund_return - risk_free_rate) / beta

def calculate_jensens_alpha(fund_return, beta, risk_free_rate, market_return):
    # 根据提供的公式计算 Jensen's Alpha
    alpha = fund_return - risk_free_rate - beta * (market_return - risk_free_rate)
    return alpha

# 排名函数
def rank_funds(funds, metric):
    return sorted(funds.items(), key=lambda x: x[1][metric], reverse=True)

# 主函数
def main():
    # 输入数据
    funds = {
        'Fund A': {'return': 0.085, 'std_dev': 0.175, 'beta': 0.97},
        'Fund B': {'return': 0.12, 'std_dev': 0.04, 'beta': 1.2},
        'Fund C': {'return': 0.18, 'std_dev': 0.08, 'beta': 1.6}
    }
    risk_free_rate = 0.5
    market_return = 0.12  # 市场回报率

    # 计算指标
    for name, fund in funds.items():
        fund['sharpe'] = calculate_sharpe_ratio(fund['return'], fund['std_dev'], risk_free_rate)
        fund['treynor'] = calculate_treynor_ratio(fund['return'], fund['beta'], risk_free_rate)
        fund['jensen'] = calculate_jensens_alpha(fund['return'], fund['beta'], risk_free_rate, market_return)

    # 输出排名
    print_rankings(funds)

def print_rankings(funds):
    for metric in ['sharpe', 'treynor', 'jensen']:
        ranked = rank_funds(funds, metric)
        print(f"\n{metric.capitalize()} Ratio Ranking:")
        for fund in ranked:
            print(f"{fund[0]}: {fund[1][metric]:.4f}")

if __name__ == "__main__":
    main()
