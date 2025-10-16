import time
import numpy as np


def OOA(func, lb, ub, dim=None, n_pop=30, max_iter=500, seed=None, verbose=False):
    """
    Orangutan Optimization Algorithm (OOA) - minimization.

    Parameters
    ----------
    func : callable
        Objective function: f(x) -> scalar. x is 1D numpy array of length dim.
    lb : array-like or scalar
        Lower bound(s). If scalar, broadcast to all dimensions.
    ub : array-like or scalar
        Upper bound(s). If scalar, broadcast to all dimensions.
    dim : int, optional
        Dimensionality. If None, inferred from lb/ub arrays.
    n_pop : int
        Population size (number of orangutans).
    max_iter : int
        Maximum number of iterations (T in the paper).
    seed : int or None
        Random seed for reproducibility.
    verbose : bool
        If True, prints iteration progress.

    Returns
    -------
    best_x : np.ndarray
        Best-found solution (dim,)
    best_f : float
        Best fitness (minimum).
    history : dict
        { 'best_f_per_iter': list, 'best_x_per_iter': list }

    Notes
    -----
    - Initialization uses Eq.(2): x_i,d = lb_d + r*(ub_d - lb_d). :contentReference[oaicite:3]{index=3}
    - Phase 1 (Foraging) uses Eq.(5)-(6): x_p1 = x + r*(SFS - I*x). Accept if improved. :contentReference[oaicite:4]{index=4}
    - Phase 2 (Nesting) uses Eq.(7)-(8): x_p2 = x + (1-2*r_ij) * ((ub-lb) / t). Accept if improved. Iteration counter t starts at 1. :contentReference[oaicite:5]{index=5}
    - The paper suggests r with a normal distribution in [0,1] for Eq.(5); here we sample a clipped normal centered at 0.5 (can be changed).
    """
    if seed is not None:
        np.random.seed(seed)
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)
    if dim is None:
        if lb.ndim == 0:
            assert ub.ndim == 0
            raise ValueError("Please pass dim when lb and ub are scalars.")
        dim = lb.size
    # handle scalar lb/ub
    if lb.size == 1:
        lb = np.full(dim, float(lb.item()))
    if ub.size == 1:
        ub = np.full(dim, float(ub.item()))
    assert lb.shape[0] == dim and ub.shape[0] == dim

    # Initialization (Eq.2)
    X = np.zeros((n_pop, dim))
    for i in range(n_pop):
        r = np.random.rand(dim)  # uniform for initialization
        X[i, :] = lb + r * (ub - lb)

    # Evaluate objective for each orangutan
    F = np.array([func(X[i, :]) for i in range(n_pop)])
    best_idx = np.argmin(F)
    best_x = X[best_idx].copy()
    best_f = F[best_idx]

    best_f_history = [best_f]
    best_x_history = [best_x.copy()]
    ct = time.time()
    # Main loop
    for t in range(1, max_iter + 1):  # iteration counter t starts at 1
        # Phase 1: Foraging (exploration)
        for i in range(n_pop):
            Fi = F[i]
            # FS_i: indices of individuals with strictly better fitness
            FS_indices = np.where(F < Fi)[0]
            if FS_indices.size == 0:
                # No better individual: choose a random individual (exclude self)
                k = np.random.choice([j for j in range(n_pop) if j != i])
                SFS = X[k]
            else:
                # choose one food source randomly among better ones
                k = np.random.choice(FS_indices)
                SFS = X[k]

            # r ~ approx normal clipped to [0,1] as paper suggests normal in [0,1]
            r_scalar = np.random.normal(loc=0.5, scale=0.15)
            r_scalar = float(np.clip(r_scalar, 0.0, 1.0))
            I = np.random.choice([1, 2])  # I in {1,2}
            x_p1 = X[i] + r_scalar * (SFS - I * X[i])  # Eq.(5)
            # keep within bounds
            x_p1 = np.minimum(np.maximum(x_p1, lb), ub)
            f_p1 = func(x_p1)
            # Eq.(6): accept if improved (minimization)
            if f_p1 <= Fi:
                X[i] = x_p1
                F[i] = f_p1
                if f_p1 < best_f:
                    best_f = f_p1
                    best_x = x_p1.copy()

        # Phase 2: Nesting (exploitation)
        for i in range(n_pop):
            # create x_p2 dimension-wise per Eq.(7)
            # r_ij is independent for each dimension
            r_ij = np.random.rand(dim)
            # compute step factor: (ub_j - lb_j) / t  (paper uses division by t)
            step = (ub - lb) / float(t)
            x_p2 = X[i] + (1.0 - 2.0 * r_ij) * step  # Eq.(7)
            x_p2 = np.minimum(np.maximum(x_p2, lb), ub)
            f_p2 = func(x_p2)
            if f_p2 <= F[i]:  # Eq.(8) accept if improved
                X[i] = x_p2
                F[i] = f_p2
                if f_p2 < best_f:
                    best_f = f_p2
                    best_x = x_p2.copy()

        # end of iteration: update best trackers
        best_f_history.append(best_f)
        best_x_history.append(best_x.copy())
        if verbose and (t % max(1, (max_iter // 10)) == 0):
            print(f"[OOA] iter {t}/{max_iter} best_f = {best_f:.6e}")

    ct = time.time() - ct
    return best_f, best_f_history,  best_x, ct

