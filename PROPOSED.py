import time
import numpy as np

def PROPOSED(population, obj_func, lb, ub, max_iter):
    """
    Updated Technical and Vocational Education and Training-Based Optimizer (TVETBO)
    with fitness-based phase selection.
    """

    population_size, dim = population.shape[0], population.shape[1]
    fitness = np.array([obj_func(ind) for ind in population])

    best_idx = np.argmin(fitness)
    instructor = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence = []
    ct = time.time()

    for t in range(max_iter):
        worst_fitness = np.max(fitness)
        for i in range(population_size):
            r = fitness[i] / (best_fitness + worst_fitness + 1e-12)  # avoid division by zero  # PROPOSED updation

            if r > 0.5:
                # Phase 1: Theory Education (Exploration)
                r1 = np.random.rand(dim)
                new_pos = population[i] + r1 * (instructor - np.random.rand(dim) * population[i])
                new_pos = np.clip(new_pos, lb, ub)
                new_fitness = obj_func(new_pos)
                if new_fitness < fitness[i]:
                    population[i] = new_pos
                    fitness[i] = new_fitness

                # Phase 2: Practical Education (Exploration)
                j = np.random.randint(0, population_size)
                while j == i:
                    j = np.random.randint(0, population_size)
                r2 = np.random.rand(dim)
                new_pos = population[i] + r2 * (population[j] - population[i])
                new_pos = np.clip(new_pos, lb, ub)
                new_fitness = obj_func(new_pos)
                if new_fitness < fitness[i]:
                    population[i] = new_pos
                    fitness[i] = new_fitness
            else:
                # Phase 3: Improving Individual Skills (Exploitation)
                r3 = np.random.uniform(-0.1, 0.1, dim)
                new_pos = population[i] + r3 * population[i]
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
