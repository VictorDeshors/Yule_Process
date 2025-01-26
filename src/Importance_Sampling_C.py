import numpy as np
import matplotlib.pyplot as plt
import numba
import scipy.stats as sps

@numba.njit()
def Y_t_1(t, lambda_, Y_0):
    """Simulation of Y_t using Method A) starting from Y_0: returns the first k such that time > t
    """
    time = 0
    k = Y_0
    while time < t:
        time += np.random.exponential(1 / (lambda_ * k))
        if time < t:
            k += 1
    return k

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
def F(n, C):
    total = 0.0
    for i in range(1, n + 1):
        total += np.log((i + C) / i)
    return total

@numba.njit()
def importance_sampling(t, A, C, lambda_, N):
    """ Calculate P(Y_t > A) using importance sampling. Also returns the list of weights associated with each simulation to compute uncertainties """
    proba = 0.0
    list_std = []
    for _ in range(N):
        Y_t = 0
        for i in range(C + 1):
            Y_t += Y_t_1(t, lambda_, 1)
        weight = 1 / (np.exp(F(Y_t - (C + 1), C) - lambda_ * C * t))
        if (Y_t - C) > A:
            proba += weight
            list_std.append(weight)
    proba /= N
    return proba, list_std

def display_importance_sampling(t, A, C, lambda_, N):
    proba_emp, list_std = importance_sampling(t, A, C, lambda_, N)

    # theoretical probability
    proba_theorique = sps.geom.sf(A, 1 / np.exp(lambda_ * t))
    # empirical standard deviation
    std_emp = np.std(list_std)
    # theoretical standard deviation
    std_th = np.sqrt(proba_theorique * (1 - proba_theorique))

    proba_low_emp = max(proba_emp - 1.96 * std_emp / np.sqrt(N), 0)
    proba_high_emp = min(proba_emp + 1.96 * std_emp / np.sqrt(N), 1)
    proba_low_th = max(proba_emp - 1.96 * std_th / np.sqrt(N), 0)
    proba_high_th = min(proba_emp + 1.96 * std_th / np.sqrt(N), 1)

    print("The empirical standard deviation is:", std_emp)
    print("The length of the empirical confidence interval is:", 2 * std_emp / np.sqrt(N), "\n")
    print("Theoretical value of P(Y_t > A) under the measure P:", proba_theorique)
    print("Estimation of P(Y_t > A) under the measure P:", proba_emp, "\n")
    print("The confidence interval with empirical standard deviation is:", [proba_low_emp, proba_high_emp])
    print("The confidence interval with theoretical standard deviation is:", [proba_low_th, proba_high_th], "\n")

def plot_proba_importance_sampling_t(liste_t, A, C, lambda_, N):
    """ plot P(Y_t > A) by importance sampling as a function of t"""
    probas_empiriques = []
    intervalles_confiance = []
    probas_theoriques = []
    
    for t in liste_t:
        proba, list_std = importance_sampling(t, A, C, lambda_, N)
        probas_empiriques.append(proba)
        std_emp = np.std(list_std) 
        probas_theoriques.append(theoretical_probability_greater_than_A(t, A, lambda_))
        intervalle_confiance = (max(0, proba - 1.96 * std_emp / np.sqrt(N)), proba + 1.96 * std_emp / np.sqrt(N))
        intervalles_confiance.append(intervalle_confiance)
        
    plt.plot(liste_t, probas_empiriques, marker='o', linestyle='--', label="Empirical")
    plt.fill_between(liste_t, [ic[0] for ic in intervalles_confiance], [ic[1] for ic in intervalles_confiance], color='b', alpha=.1, label="95% CI")
    plt.plot(liste_t, probas_theoriques, marker='o', linestyle='--', label="Theoretical")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("P(Y_t > A)")
    plt.title("P(Y_t > A) by importance sampling as a function of t\n for A = {}, lambda = {}, N = {}, C = {}".format(A, lambda_, N, C))
    plt.grid()
    plt.show()
    
def plot_proba_importance_sampling_A(liste_A, t, C, lambda_, N):
    """ plot P(Y_t > A) by importance sampling as a function of A"""
    probas_empiriques = []
    intervalles_confiance = []
    probas_theoriques = []
    for A in liste_A:
        proba, list_std = importance_sampling(t, A, C, lambda_, N)
        probas_empiriques.append(proba)
        std_emp = np.std(list_std)
        probas_theoriques.append(theoretical_probability_greater_than_A(t, A, lambda_))
        intervalle_confiance = (max(0, proba - 1.96 * std_emp / np.sqrt(N)), proba + 1.96 * std_emp / np.sqrt(N))
        intervalles_confiance.append(intervalle_confiance)
        
    plt.plot(liste_A, probas_empiriques, marker='o', linestyle='--', label="Empirical")
    plt.fill_between(liste_A, [ic[0] for ic in intervalles_confiance], [ic[1] for ic in intervalles_confiance], color='b', alpha=.1, label="95% CI")
    plt.plot(liste_A, probas_theoriques, marker='o', linestyle='--', label="Theoretical")
    plt.xlabel("A")
    plt.ylabel("P(Y_t > A)")
    plt.legend()
    plt.grid()
    plt.title("P(Y_t > A) by importance sampling as a function of A\n for t = {}, lambda = {}, N = {}, C = {}".format(t, lambda_, N, C))
    plt.show()

def plot_proba_importance_sampling_lambda_(liste_lambda_, t, A, C, N):
    """ plot P(Y_t > A) by importance sampling as a function of lambda"""
    probas_empiriques = []
    intervalles_confiance = []
    probas_theoriques = []
    
    for lambda_ in liste_lambda_:
        proba, list_std = importance_sampling(t, A, C, lambda_, N)
        probas_empiriques.append(proba)
        std_emp = np.std(list_std)
        probas_theoriques.append(theoretical_probability_greater_than_A(t, A, lambda_))
        intervalle_confiance = (max(0, proba - 1.96 * std_emp / np.sqrt(N)), proba + 1.96 * std_emp / np.sqrt(N))
        intervalles_confiance.append(intervalle_confiance)
        
    plt.plot(liste_lambda_, probas_empiriques, marker='o', linestyle='--', label="Empirical")
    plt.fill_between(liste_lambda_, [ic[0] for ic in intervalles_confiance], [ic[1] for ic in intervalles_confiance], color='b', alpha=.1, label="95% CI")
    plt.plot(liste_lambda_, probas_theoriques, marker='o', linestyle='--', label="Theoretical")
    plt.xlabel("lambda")
    plt.ylabel("P(Y_t > A)")
    plt.grid()
    plt.legend()
    plt.title("P(Y_t > A) by importance sampling as a function of lambda\n for t = {}, A = {}, N = {}, C = {}".format(t, A, N, C))
    plt.show()

def plot_proba_importance_sampling_C(liste_C, t, A, lambda_, N):
    """ plot P(Y_t > A) by importance sampling as a function of C"""
    probas_empiriques = []
    intervalles_confiance = []
    probas_theoriques = []
    
    for C in liste_C:
        proba, list_std = importance_sampling(t, A, C, lambda_, N)
        probas_empiriques.append(proba)
        std_emp = np.std(list_std)
        probas_theoriques.append(theoretical_probability_greater_than_A(t, A, lambda_))
        intervalle_confiance = (max(0, proba - 1.96 * std_emp / np.sqrt(N)), proba + 1.96 * std_emp / np.sqrt(N))
        intervalles_confiance.append(intervalle_confiance)
        
    plt.plot(liste_C, probas_empiriques, marker='o', linestyle='--', label="Empirical")
    plt.fill_between(liste_C, [ic[0] for ic in intervalles_confiance], [ic[1] for ic in intervalles_confiance], color='b', alpha=.1, label="95% CI")
    plt.plot(liste_C, probas_theoriques, marker='o', linestyle='--', label="Theoretical")
    plt.xlabel("C")
    plt.ylabel("P(Y_t > A)")
    plt.title("P(Y_t > A) by importance sampling as a function of C\n for t = {}, A = {}, N = {}, lambda = {}".format(t, A, N, lambda_))
    plt.grid()
    plt.legend()
    plt.show()