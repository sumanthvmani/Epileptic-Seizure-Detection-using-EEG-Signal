import time

import numpy as np


def POA(X,  fitness, lowerbound, upperbound, Max_iterations):
    SearchAgents, dimension = X.shape[0], X.shape[1]
    lowerbound = np.ones(dimension) * lowerbound  # Lower limit for variables
    upperbound = np.ones(dimension) * upperbound  # Upper limit for variables


    fit = np.zeros(SearchAgents)
    for i in range(SearchAgents):
        L = X[i, :]
        fit[i] = fitness(L)

    best_so_far = np.zeros(Max_iterations)
    average = np.zeros(Max_iterations)
    ct = time.time()
    for t in range(Max_iterations):
        # update the best candidate solution
        best = np.min(fit)
        location = np.argmin(fit)
        if t == 0:
            Xbest = X[location, :]  # Optimal location
            fbest = best  # The optimization objective function
        elif best < fbest:
            fbest = best
            Xbest = X[location, :]

        # UPDATE location of food
        k = np.random.permutation(SearchAgents)[0]
        X_FOOD = X[k, :]
        F_FOOD = fit[k]

        for i in range(SearchAgents):
            # PHASE 1: Moving towards prey (exploration phase)
            I = round(1 + np.random.rand(1))
            if fit[i] > F_FOOD:
                X_new = X[i, :] + np.random.rand(1) * (X_FOOD - I * X[i, :])  # Eq(4)
            else:
                X_new = X[i, :] + np.random.rand(1) * (X[i, :] - 1 * X_FOOD)  # Eq(4)
            X_new = np.maximum(X_new, lowerbound)
            X_new = np.minimum(X_new, upperbound)

            # Updating X_i using (5)
            f_new = fitness(X_new)
            if f_new <= fit[i]:
                X[i, :] = X_new
                fit[i] = f_new

            # END PHASE 1: Moving towards prey (exploration phase)

            # PHASE 2: Winging on the water surface (exploitation phase)
            X_new = X[i, :] + 0.2 * (1 - t / Max_iterations) * (2 * np.random.rand(dimension) - 1) * X[i, :]  # Eq(6)
            X_new = np.maximum(X_new, lowerbound)
            X_new = np.minimum(X_new, upperbound)

            # Updating X_i using (7)
            f_new = fitness(X_new)
            if f_new <= fit[i]:
                X[i, :] = X_new
                fit[i] = f_new

            # END PHASE 2: Winging on the water surface (exploitation phase)

        best_so_far[t] = fbest
        average[t] = np.mean(fit)
    ct = time.time() - ct
    return fbest, best_so_far ,Xbest,ct