import numpy as np
import matplotlib.pyplot as plt
import numba
import scipy.special

def theoretical_value_An_tilde(lambda_, n):
    p = lambda_ / (1 + lambda_)
    sum_value = 0
    for k in range(1, n):
        binom1 = scipy.special.comb(2*n - 3, n - 2 + k)
        binom2 = scipy.special.comb(2*n - 3, n + k - 1)
        sum_value += (binom1 - binom2) * (p**(n-2+k)) * ((1-p)**(n-1-k))
    return 1 - sum_value

# model change 
@numba.jit()
def TEST_SITE_REACHED(lambda_, n):
    prey_count = 1 
    site_number = 1  # number of the furthest prey site
    N = 0  # number of steps
    while (prey_count > 0 and N < 2*n - 3 and prey_count < n):
        # If N = 2*n - 3, it implies there is necessarily one prey that has passed the site n.
        a = np.random.exponential(1, 1)
        b = np.random.exponential(1/lambda_, 1)
        if a < b:
            prey_count -= 1
        else:
            prey_count += 1
            site_number += 1
        N += 1
    if (prey_count == 0):
        return False, site_number  # we are dead before reaching the target site
    else:
        return True, site_number
    
@numba.jit()   
def empirical_An_tilde(lambda_, n, N):
    count = 0
    for i in range(N):
        if TEST_SITE_REACHED(lambda_, n)[0]:
            count += 1
    return 1 - count / N

def plot_An_tilde_fixed_n(lambda_list, n, N):    
    proba_lambda = [empirical_An_tilde(lambda_, n, N) for lambda_ in lambda_list]
    plt.plot(lambda_list, proba_lambda, marker='o', linestyle='-', label="Empirical An_tilde")
    plt.xlabel('lambda')
    plt.ylabel('An_tilde')
    
    # Display theoretical An_tilde
    An_tilde_theoretical = [theoretical_value_An_tilde(lambda_, n) for lambda_ in lambda_list]
    plt.plot(lambda_list, An_tilde_theoretical, marker='o', label='Theoretical A_n_tilde', markersize=1)
    
    # Display p(lambda)
    p_lambda = 1 / np.array(lambda_list)
    plt.plot(lambda_list, p_lambda, marker='o', label='Theoretical p(lambda)', markersize=1)
    
    plt.title(f'An_tilde as a function of lambda for n = {n}, N = {N}')
    plt.grid(True)
    plt.legend()
    plt.show()
    
def plot_An_tilde_fixed_lambda(lambda_, n_list, N):
    proba_n = [empirical_An_tilde(lambda_, n, N) for n in n_list]
    
    plt.plot(n_list, proba_n, marker='o', linestyle='-', label="Empirical An_tilde")
    plt.xlabel('Value of n')
    plt.ylabel('An_tilde')
    plt.title(f'An_tilde as a function of n for lambda = {lambda_}')
    plt.grid(True)
    
    # Display theoretical An_tilde
    An_tilde_theoretical = [theoretical_value_An_tilde(lambda_, n) for n in n_list]
    plt.plot(n_list, An_tilde_theoretical, marker='o', label='Theoretical A_n_tilde', markersize=1)
    
    # Display 1 - p(lambda)
    p_lambda_ = 1 / lambda_
    plt.axhline(y=p_lambda_, color='r', linestyle='--', label=f'p(lambda_ = {lambda_}) theoretical')
    plt.legend()
    plt.show()