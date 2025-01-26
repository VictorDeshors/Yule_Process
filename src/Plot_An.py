import numpy as np
import matplotlib.pyplot as plt
import numba

def theoretical_value_An(lambda_, n):
    p = lambda_ / (1 + lambda_)
    return (1 - (1 - p) / p) / (1 - ((1 - p) / p) ** n)

@numba.njit()
def calculate_A_n_lambdaV1(lambda_1, lambda_2, T, n, N):
    p_1 = float(lambda_1 / (1.0 + lambda_1))  # Probability that Exp(lambda_1) > Exp(1)
    p_2 = float(lambda_2 / (1.0 + lambda_2))  # Probability that Exp(lambda_2) > Exp(1)
    coeff_1 = p_1 / p_2
    coeff_2 = ((1 - p_1) / (1 - p_2))
    probability = 0.0
    weight_list = []
    random_numbers = np.random.binomial(1, p_2, (N, T))

    for i in range(N):
        num_prey = 1
        M = 1.0
        for j in range(T):
            X_i = random_numbers[i, j]
            num_prey += (2 * X_i - 1)

            if X_i == 1:
                M *= coeff_1
            else:
                M *= coeff_2

            if num_prey >= n:
                probability += M
                weight_list.append(M)
                break

            if num_prey <= 0:
                break

    probability /= N
    weight_list = np.array(weight_list)
    return probability, weight_list

### VERSION 3 ###
@numba.njit()
def calculate_A_n_lambdaV2(lambda_1, lambda_2, n, N):
    p_1 = float(lambda_1 / (1.0 + lambda_1))  # Probability that Exp(lambda_1) > Exp(1)
    p_2 = float(lambda_2 / (1.0 + lambda_2))  # Probability that Exp(lambda_2) > Exp(1)
    coeff_1 = p_1 / p_2
    coeff_2 = ((1 - p_1) / (1 - p_2))
    probability = 0.0
    weight_list = []

    batch_size = 10000000  # size of a batch of random variables to save time
    batch_i = batch_size  # index of the next random variable in the batch (= batch_size means creating a new batch)

    for i in range(N):
        num_prey = 1
        M = 1.0
        while True:
            if batch_i >= batch_size:
                # Remove a batch of random variables
                random_numbers = np.random.binomial(1, p_2, batch_size)
                batch_i = 0  # reinitialize the index

            X_i = random_numbers[batch_i]
            batch_i += 1
            num_prey += (2 * X_i - 1)

            if X_i == 1:
                M *= coeff_1
            else:
                M *= coeff_2

            if num_prey >= n:
                probability += M
                weight_list.append(M)
                break

            if num_prey <= 0:
                break

    probability /= N
    weight_list = np.array(weight_list)
    return probability, weight_list

def plot_probability_function_of_lambda_V21(lambda_1, lambda_list_2, T, n, N):
    probability = []
    list_of_weight_lists = []

    for i in range(len(lambda_list_2)):
        p, weight_list = calculate_A_n_lambdaV1(lambda_1, lambda_list_2[i], T, n, N)
        probability.append(p)
        list_of_weight_lists.append(weight_list)

    # Display theoretical values
    A_n_theoretical = theoretical_value_An(lambda_1, n)
    plt.axhline(y=A_n_theoretical, color='r', linestyle='-', label="theoretical value A_n")

    # Calculate standard deviations
    list_of_stds = [np.std(list_of_weight_lists[i]) for i in range(len(lambda_list_2))]

    # Display confidence intervals
    plt.fill_between(lambda_list_2, [probability[i] - 1.96 * list_of_stds[i] / np.sqrt(N) for i in range(len(lambda_list_2))], [probability[i] + 1.96 * list_of_stds[i] / np.sqrt(N) for i in range(len(lambda_list_2))], color='gray', alpha=0.5)

    plt.plot(lambda_list_2, probability, marker='o', linestyle='-')
    plt.xlabel('lambda_2')
    plt.ylabel('p(lambda_)')
    plt.title(' p(lambda_) calculated by IS_V1 as a function of lambda_2')
    plt.grid(True)
    plt.show()
    return np.array(probability)

def plot_probability_function_of_lambda_V2(lambda_1, lambda_list_2, n, N):
    probability = []
    list_of_weight_lists = []

    for i in range(len(lambda_list_2)):
        p, weight_list = calculate_A_n_lambdaV2(lambda_1, lambda_list_2[i], n, N)
        probability.append(p)
        list_of_weight_lists.append(weight_list)

    A_n_theoretical = theoretical_value_An(lambda_1, n)
    plt.axhline(y=A_n_theoretical, color='r', linestyle='-', label="theoretical value A_n")

    list_of_stds = [np.std(list_of_weight_lists[i]) for i in range(len(lambda_list_2))]

    plt.fill_between(lambda_list_2, [probability[i] - 1.96 * list_of_stds[i] / np.sqrt(N) for i in range(len(lambda_list_2))], [probability[i] + 1.96 * list_of_stds[i] / np.sqrt(N) for i in range(len(lambda_list_2))], color='gray', alpha=0.5)

    plt.plot(lambda_list_2, probability, marker='o', linestyle='-')
    plt.xlabel('lambda_2')
    plt.ylabel('p(lambda_)')
    plt.title(' p(lambda_) calculated by IS_V2 as a function of lambda_2')
    plt.grid(True)
    plt.show()
                                     
    return np.array(probability)

def plot_proba_function_of_lambda_V1_V2(lambda_1, list_lambda_2, T, n, N):
    proba_V2 = []
    proba_V3 = []
    list_list_weights_V2 = []
    list_list_weights_V3 = []
    
    for i in range(len(list_lambda_2)):
        p_V2, list_weights_V2 = calculate_A_n_lambdaV1(lambda_1, list_lambda_2[i], T, n, N)
        p_V3, list_weights_V3 = calculate_A_n_lambdaV2(lambda_1, list_lambda_2[i], n, N)
        proba_V2.append(p_V2)
        proba_V3.append(p_V3)
        list_list_weights_V2.append(list_weights_V2)
        list_list_weights_V3.append(list_weights_V3)
            
    A_n_theoretical = theoretical_value_An(lambda_1, n)
    plt.axhline(y=A_n_theoretical, color='r', linestyle='-', label="Theoretical value A_n")
    
    list_stds_V2 = [np.std(list_list_weights_V2[i]) for i in range(len(list_lambda_2))]
    list_stds_V3 = [np.std(list_list_weights_V3[i]) for i in range(len(list_lambda_2))]
                     
    plt.fill_between(list_lambda_2, 
                     [proba_V2[i] - 1.96 * list_stds_V2[i] / np.sqrt(N) for i in range(len(list_lambda_2))], 
                     [proba_V2[i] + 1.96 * list_stds_V2[i] / np.sqrt(N) for i in range(len(list_lambda_2))], 
                     color='skyblue', alpha=0.5, label="Confidence Interval IS_V1")
    plt.fill_between(list_lambda_2, 
                     [proba_V3[i] - 1.96 * list_stds_V3[i] / np.sqrt(N) for i in range(len(list_lambda_2))], 
                     [proba_V3[i] + 1.96 * list_stds_V3[i] / np.sqrt(N) for i in range(len(list_lambda_2))], 
                     color=(1, 0.5, 0, 0.5), alpha=0.5, label="Confidence Interval IS_V2")
    
    plt.plot(list_lambda_2, proba_V2, marker='o', linestyle='-', label="IS_V1")
    plt.plot(list_lambda_2, proba_V3, marker='o', linestyle='-', label="IS_V2")
    plt.xlabel('lambda_2')
    plt.ylabel('p(lambda_)')
    plt.title('p(lambda_) calculated by IS_V1 and IS_V2 as a function of lambda_2')
    plt.legend()
    plt.grid(True)
    
    return np.array(proba_V2), np.array(proba_V3)