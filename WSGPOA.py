# White Shark Optimization + Golden Eagle Optimization + Pelican Optimization Algorithm


import numpy as np


def WSGPOA(obj_func, dim, n_sharks, max_iter, lb, ub):

    ub = np.array(ub)
    lb = np.array(lb)

    # Initialize the population of sharks
    sharks = lb + np.random.rand(n_sharks, dim) * (ub - lb)
    velocities = np.zeros((n_sharks, dim))
    best_shark = sharks[np.argmin([obj_func(s) for s in sharks])]

    # Constants
    pmin, pmax = 0.5, 1.5
    tau = 4.125
    mu = 2 / abs(2 - tau - np.sqrt(tau ** 2 - 4 * tau))

    for k in range(max_iter):
        fitness = np.array([obj_func(s) for s in sharks])
        best_idx = np.argmin(fitness)
        gbest = sharks[best_idx]

        # Compute parameters
        p1 = pmax + (pmax - pmin) * np.exp(-((4 * k / max_iter) ** 2))
        p2 = pmin + (pmax - pmin) * np.exp(-((4 * k / max_iter) ** 2))

        for i in range(n_sharks):
            v_best_idx = np.random.randint(0, n_sharks)
            v_best = sharks[v_best_idx]
            c1, c2 = np.random.rand(), np.random.rand()

            # Update velocity
            velocities[i] = mu * (velocities[i] + p1 * c1 * (gbest - sharks[i]) + p2 * c2 * (v_best - sharks[i]))

            # Update position
            if np.random.rand() < (1 / (1 + np.exp((max_iter / 2 - k) / 5))):
                sharks[i] += velocities[i]
            else:
                q = np.sign(sharks[i] - ub)
                a = (sharks[i] - lb) > 0
                b = (sharks[i] - ub) < 0
                w0 = np.logical_or(a, b)
                sharks[i] = sharks[i] * q * w0 + lb + np.random.rand(dim) * (ub - lb)

                # GEO - Compute Attack Vector (AV) (Guided attack towards prey)
                prey_index = np.random.randint(0, n_sharks)
                prey = sharks[prey_index]
                AV = prey - sharks[i]

                spiral_factor = np.random.uniform(-1, 1, dim)  # Spiral movement factor
                CV = AV - spiral_factor  # Tangential movement (mimicking GEO's cruise motion)

                # WSO - Adaptive Attack & Cruise using GEO-inspired transition coefficients
                rc = np.random.uniform(0, 1)  # Attack coefficient
                rd = np.random.uniform(0, 1)  # Cruise coefficient

                # White Shark Updated Position (Combining WSO + GEO + POA)
                R = 0.2
                poa_update = R * (1-(k/max_iter))
                sharks[i] = sharks[i] + rc * AV + rd * CV + poa_update

            # Boundaries enforcement
            sharks[i] = np.clip(sharks[i], lb, ub)

    best_shark = sharks[np.argmin([obj_func(s) for s in sharks])]
    return best_shark, obj_func(best_shark)


