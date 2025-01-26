import numpy as np 
import matplotlib.pyplot as plt

### E) Backward Simulation ###

def yule_backward(lambda_, t):
    """Calculation of a backward process:
    Fix t and lambda_, and simulate the number of Poisson events up to t
    """
    eps = np.random.exponential(1, 1)
    T = eps * (np.exp(lambda_ * t) - 1)
    N_t = np.random.poisson(T, 1)
    return N_t


def N_yule_backward_process(lambda_, t, N):
    """Calculation of N backward processes:
    """
    Y_t = np.zeros(N)
    for i in range(N):
        Y_t[i] = yule_backward(lambda_, t) + 1
    return Y_t


def display(lambda_, t, N):
    Y_t = N_yule_backward_process(lambda_, t, N)
    # Convert Y_t to integers
    Y_t = Y_t.astype(int)
    # Display the distribution of Y_t
    counts = np.bincount(Y_t)
    M = np.max(Y_t)
    plt.bar(np.arange(M + 1), counts / N, width=0.5, label="Simulation")
    
    # Display the theoretical distribution of Y_t: geometric distribution with parameter np.exp(-lambda_*t) for k >= 1
    p = np.exp(-lambda_ * t)
    X = np.arange(1, M + 1)
    Y = p * (1 - p)**(X - 1)
    plt.plot(X, Y, color='red', label="Theoretical Distribution: Geometric with parameter np.exp(-lambda_*t)")
    
    plt.xlabel("k")
    plt.ylabel("P(Y_t = k)")
    plt.title("Distribution of Y_t for lambda = " + str(lambda_) + ", t = " + str(t) + " and N = " + str(N))
    plt.legend()
    plt.show()