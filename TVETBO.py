import time
import numpy as np


def TVETBO(population, obj_func, lb, ub, max_iter):
    """
    Technical and Vocational Education and Training-Based Optimizer (TVETBO)
    Returns the best solution, its fitness, and convergence history.

    Parameters:
    - obj_func: function to minimize
    - dim: number of decision variables
    - bounds: list of tuples specifying (min, max) for each dimension
    - population_size: number of members in the population
    - max_iter: maximum number of iterations

    Returns:
    - best_solution: position of the best member (Instructor)
    - best_fitness: objective value of the best member
    - convergence: list of best fitness per iteration
    """

    # Initialization
    population_size, dim = population.shape[0], population.shape[1]
    fitness = np.array([obj_func(ind) for ind in population])

    best_idx = np.argmin(fitness)
    instructor = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence = []
    ct = time.time()
    for t in range(max_iter):
        # Phase 1: Theory Education (Exploration)
        for i in range(population_size):
            r = np.random.rand(dim)
            new_pos = population[i] + r * (instructor - np.random.rand(dim) * population[i])
            new_pos = np.clip(new_pos, lb, ub)
            new_fitness = obj_func(new_pos)
            if new_fitness < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fitness

        # Phase 2: Practical Education (Exploitation)
        for i in range(population_size):
            j = np.random.randint(0, population_size)
            while j == i:
                j = np.random.randint(0, population_size)
            r = np.random.rand(dim)
            new_pos = population[i] + r * (population[j] - population[i])
            new_pos = np.clip(new_pos, lb, ub)
            new_fitness = obj_func(new_pos)
            if new_fitness < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fitness

        # Phase 3: Individual Skills Development (Refinement)
        for i in range(population_size):
            r = np.random.uniform(-0.1, 0.1, dim)  # small perturbation
            new_pos = population[i] + r * (population[i])
            new_pos = np.clip(new_pos, lb, ub)
            new_fitness = obj_func(new_pos)
            if new_fitness < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fitness

        # Update Instructor
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            instructor = population[best_idx].copy()
            best_fitness = fitness[best_idx]

        convergence.append(best_fitness)
        print(f"Iteration {t + 1}/{max_iter}, Best Fitness: {best_fitness}")

    ct = time.time() - ct
    return best_fitness, convergence, instructor, ct
