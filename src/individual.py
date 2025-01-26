import numpy as np
import matplotlib.pyplot as plt


### Display of the population of individuals based on their birth date using a binary tree

class Individual:
    def __init__(self, id, birth_time, parent=None, eps=None, generation=0):
        self.id = id
        self.birth_time = birth_time
        self.parent = parent
        self.eps = eps
        self.children = []
        self.generation = generation

def simulate_population_tree(t_final, lambda_):
    t = 0
    k = 0
    population = [Individual(0, 0)]  # Creation of the initial individual
    while t < t_final:
        k += 1
        t += np.random.exponential(1 / (lambda_ * k))
        parents = [ind for ind in population if not ind.children]  # Select individuals who haven't had children yet
        parent = np.random.choice(parents)  # Randomly choose a parent from those who haven't had children yet
        child1 = Individual(k, t, parent=parent, eps=1, generation=parent.generation + 1)  # Creation of the first child
        child2 = Individual(k+1, t, parent=parent, eps=-1, generation=parent.generation + 1)  # Creation of the second child
        parent.children.extend([child1, child2])  # Add the children to the parent
        population.extend([child1, child2])  # Add the children to the population
    return population

def plot_population_tree(population):
    plt.figure(figsize=(10, 8))
    for ind in population:
        x = ind.birth_time
        y = 0 if ind.parent is None else ind.parent.y + 2**(-ind.generation) * ind.eps
        ind.y = y
        plt.plot(x, y, marker='o', markersize=5, color='black')  # Representation of the individual in the plane
        if ind.parent is not None:
            plt.plot([ind.parent.birth_time, x], [ind.parent.y, y], color='black')  # Draw a branch between the parent and child
    plt.xlabel('Birth Time')
    plt.title('Representation of Individuals in the Plane Based on Birth Time')
    title = plt.gca().get_title()
    plt.grid(True)
    plt.show()