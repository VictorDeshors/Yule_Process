import numpy as np
import matplotlib.pyplot as plt
import numba

@numba.njit()
def Y_t_2(t, lambda_):
    """
    Simulation of the distribution of Y_t for fixed t: Geometric distribution (for k >= 1) with parameter np.exp(-lambda_ * t)
    """
    return np.random.geometric(np.exp(-lambda_ * t))

def theoretical_probability_less_than_A(lambda_, t, A):
    """ Theoretical calculation of P(Y_t < A | Y_0 = 1) using (B). """
    theoretical_probability = 0
    for k in range(1, A):
        theoretical_probability += np.exp(-lambda_*t) * (1 - np.exp(-lambda_*t))**(k - 1)
    return theoretical_probability

def theoretical_probability_greater_than_A(t, A, lambda_):
    """ Calculation of P(Y_t > A | Y_0 = 1) using (B). """
    probability = 1 - theoretical_probability_less_than_A(lambda_, t, A)
    probability -= np.exp(-lambda_*t) * (1 - np.exp(-lambda_*t))**(A - 1)
    return probability

@numba.njit()
def Monte_Carlo_Y_t_greater_than_A(t, A, lam, N):
    cnt_greater_than_A = 0
    for _ in range(N):
        Y_t = Y_t_2(t, lam)
        if Y_t > A:
            cnt_greater_than_A += 1
            
    probability = cnt_greater_than_A / N
    std = np.sqrt(probability * (1 - probability) / N)
    confidence_interval = (probability - 1.96 * std, probability + 1.96 * std)
    return probability, confidence_interval

def plot_probability_Y_t_greater_than_A_A(t, lambda_, N, A_list):
    """ Plot of the probability of Y_t > A as a function of A """
    empirical_probabilities = []
    confidence_intervals = []
    theoretical_probabilities = []
    
    for A in A_list:
        proba, confidence_interval = Monte_Carlo_Y_t_greater_than_A(t, A, lambda_, N)
        empirical_probabilities.append(proba)
        confidence_intervals.append(confidence_interval)
        theoretical_probabilities.append(theoretical_probability_greater_than_A(t, A, lambda_))
        
    plt.plot(A_list, empirical_probabilities, marker='o', linestyle='--', label="Empirical")
    plt.fill_between(A_list, [ic[0] for ic in confidence_intervals], [ic[1] for ic in confidence_intervals], color='b', alpha=.1, label="95% CI")
    plt.plot(A_list, theoretical_probabilities, marker='o', linestyle='--', label="Theoretical")
    plt.legend()
    plt.xlabel("A")
    plt.ylabel("P(Y_t > A)")
    title = f"P(Y_t > A) by naive Monte Carlo as a function of A\n for t = {t}, lambda = {lambda_}, N = {N}"
    plt.title(title)
    plt.grid()
    plt.show()
    
def plot_probability_Y_t_greater_than_A_t(lambda_, N, A, t_max):
    """ Plot of the probability of Y_t > A as a function of t """
    empirical_probabilities = []
    confidence_intervals = []
    theoretical_probabilities = []
    ts = np.arange(1, t_max + 1)
    for t in ts:
        proba, confidence_interval = Monte_Carlo_Y_t_greater_than_A(t, A, lambda_, N)
        empirical_probabilities.append(proba)
        confidence_intervals.append(confidence_interval)
        theoretical_probabilities.append(theoretical_probability_greater_than_A(t, A, lambda_))
    
    plt.plot(ts, empirical_probabilities, marker='o', linestyle='--', label="Empirical")
    plt.fill_between(ts, [ic[0] for ic in confidence_intervals], [ic[1] for ic in confidence_intervals], color='b', alpha=.1, label="95% CI")
    plt.plot(ts, theoretical_probabilities, marker='o', linestyle='--', label="Theoretical")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("P(Y_t > A)")
    plt.title(f"P(Y_t > A) by naive Monte Carlo as a function of t\n for A = {A}, lambda = {lambda_}, N = {N}")
    title = plt.gca().get_title()
    plt.grid()
    plt.show()