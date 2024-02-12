import numpy as np

def calculate_expected_payoff(probabilities, payoffs):
    return np.sum(probabilities * payoffs)

def calculate_expected_utility(wealth, probabilities, payoffs, utility_function):
    if utility_function == 'sqrt':
        utilities = np.sqrt(wealth + payoffs)
    elif utility_function == 'linear':
        utilities = wealth + payoffs
    elif utility_function == 'square':
        utilities = (wealth + payoffs) ** 2
    return np.dot(probabilities, utilities)

def calculate_certainty_equivalent(expected_utility, utility_function):
    if utility_function == 'sqrt':
        return expected_utility ** 2
    elif utility_function == 'linear':
        return expected_utility
    elif utility_function == 'square':
        return np.sqrt(expected_utility)

def calculate_risk_premium(initial_wealth, expected_payoff, certainty_equivalent):
    return (initial_wealth + expected_payoff) - certainty_equivalent

def calculate_max_price(expected_payoff, risk_premium):
    return expected_payoff - risk_premium

def calculate_expected_endofperiod_wealth(wealth, riskless_rate, payoffs):
    return wealth * (1 + riskless_rate) + payoffs

def main():
    # Given data
    W0 = 20
    riskless_rate = 0.0
    probabilities = np.array([0.35, 0.65])
    payoffs = np.array([-15, 20])
    expected_payoff = calculate_expected_payoff(probabilities, payoffs)

    utility_functions = ['sqrt', 'linear', 'square']

    for utility_function in utility_functions:
        expected_utility = calculate_expected_utility(W0, probabilities, payoffs, utility_function)
        cert_equivalent = calculate_certainty_equivalent(expected_utility, utility_function)
        risk_premium = calculate_risk_premium(W0, expected_payoff, cert_equivalent)
        max_price = calculate_max_price(expected_payoff, risk_premium)
        expected_eop_wealth = np.sum(probabilities * calculate_expected_endofperiod_wealth(W0, riskless_rate, payoffs))

        print(f"\nUtility Function = {utility_function}:")
        print(f"Expected Utility: {expected_utility}")
        print(f"Certainty Equivalent: {cert_equivalent}")
        print(f"Risk Premium: {risk_premium}")
        print(f"Maximum Price of the Asset: {max_price}")
        print(f"Expected End-of-Period Wealth: {expected_eop_wealth}")

if __name__ == "__main__":
    main()
