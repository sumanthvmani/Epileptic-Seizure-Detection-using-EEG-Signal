import time

import numpy as np


def CO(X,  lowerbound, upperbound, fitness, Max_iterations,):
    SearchAgents, dimension = X.shape[0], X.shape[1]
    for i in range(dimension):
        X[:, i] = lowerbound[i] + np.random.rand(SearchAgents) * (upperbound[i] - lowerbound[i])  # Initial population

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

        for i in range(SearchAgents // 2):
            # Phase1: Hunting and attacking strategy on iguana (Exploration Phase)
            iguana = Xbest
            I = round(1 + np.random.rand())
            X_P1 = X[i, :] + np.random.rand() * (iguana - I * X[i, :])  # Eq. (4)
            X_P1 = np.clip(X_P1, lowerbound, upperbound)

            # update position based on Eq (7)
            L = X_P1
            F_P1 = fitness(L)
            if F_P1 < fit[i]:
                X[i, :] = X_P1
                fit[i] = F_P1

        for i in range(SearchAgents // 2, SearchAgents):
            iguana = lowerbound + np.random.rand() * (upperbound - lowerbound)  # Eq(5)
            L = iguana
            F_HL = fitness(L)
            I = round(1 + np.random.rand())

            if fit[i] > F_HL:
                X_P1 = X[i, :] + np.random.rand() * (iguana - I * X[i, :])  # Eq. (6)
            else:
                X_P1 = X[i, :] + np.random.rand() * (X[i, :] - iguana)  # Eq. (6)
            X_P1 = np.clip(X_P1, lowerbound, upperbound)

            # update position based on Eq (7)
            L = X_P1
            F_P1 = fitness(L)
            if F_P1 < fit[i]:
                X[i, :] = X_P1
                fit[i] = F_P1

        # Phase2: The process of escaping from predators (Exploitation Phase)
        for i in range(SearchAgents):
            LO_LOCAL = lowerbound / (t + 1)  # Eq(9)
            HI_LOCAL = upperbound / (t + 1)  # Eq(10)

            X_P2 = X[i, :] + (1 - 2 * np.random.rand()) * (
                        LO_LOCAL + np.random.rand() * (HI_LOCAL - LO_LOCAL))  # Eq. (8)
            X_P2 = np.clip(X_P2, LO_LOCAL, HI_LOCAL)

            # update position based on Eq (11)
            L = X_P2
            F_P2 = fitness(L)
            if F_P2 < fit[i]:
                X[i, :] = X_P2
                fit[i] = F_P2

        best_so_far[t] = fbest
        average[t] = np.mean(fit)

    ct = time.time() - ct
    return fbest,  best_so_far, Xbest, ct