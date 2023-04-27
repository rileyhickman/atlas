import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def gram_matrix(X, k1, k2):
    dim = len(X)
    K = np.zeros((dim, dim))
    for i in range(0, dim):
        for j in range(0, dim):
            K[i, j] = k1*np.exp(-1/(2*k2**2)*np.linalg.norm(X[i]-X[j])**2)

    return K


# Compute the covariance vector
def cov_vect(x, X, k1, k2):
    dim = len(X)
    kt = np.zeros((dim, 1))
    for i in range(0, dim):
        kt[i, 0] = k1*np.exp(-1/(2*k2**2)*np.linalg.norm(x-X[i])**2)

    return kt


# Defind acq functions
def acq_gp(x, gp, b_n):
    if len(x.shape) == 1:
        y_mean, y_std = gp.predict(np.expand_dims(x, axis=0), return_std=True)
    else:
        y_mean, y_std = gp.predict(x, return_std=True)

    return (y_mean.ravel() + b_n*y_std.ravel())


def acq_lcb(x, gp, b_n):
    if len(x.shape) == 1:
        y_mean, y_std = gp.predict(np.expand_dims(x, axis=0), return_std=True)
    else:
        y_mean, y_std = gp.predict(x, return_std=True)

    return (y_mean.ravel() - b_n*y_std.ravel())


def acq_ei(x, gp, y_max, xi):
    if len(x.shape) == 1:
        y_mean, y_std = gp.predict(np.expand_dims(x, axis=0), return_std=True)
    else:
        y_mean, y_std = gp.predict(x, return_std=True)

    y_mean = y_mean.ravel()
    y_std = y_std.ravel()
    z = np.divide((y_mean - y_max - xi), y_std)
    acq_value = ((y_mean - y_max - xi) * norm.cdf(z) +
                 y_std * norm.pdf(z))

    return acq_value


# This function is for the reguralized BO - Shahriari et al (AISTATS 2016)
def acq_eih(x, gp, y_max, reg_type, x_bar, beta, R, w_matrix):

    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)

    x_dist_xi = x - np.transpose(x_bar)
    if reg_type == 'EIH':
        x_dist_xi_norm = np.linalg.norm(x_dist_xi, axis=1)
        x_xi_rel = np.square((x_dist_xi_norm - R)/(beta*R))

        xi = (x_dist_xi_norm > R)*x_xi_rel
    elif reg_type == 'EIQ':
        temp = np.matmul(x_dist_xi, w_matrix)
        xi = np.diagonal(np.matmul(temp, np.transpose(x_dist_xi)))
    else:
        print('This code only supports regularizer EIQ and EIH')

    y_mean, y_std = gp.predict(x, return_std=True)
    y_mean = y_mean.ravel()
    y_std = y_std.ravel()
    z = np.divide((y_mean - y_max - xi), y_std)
    acq_value = ((y_mean - y_max - xi) * norm.cdf(z) +
                 y_std * norm.pdf(z))

    return acq_value


# Function for Nguyen et al paper (ICDM 2017)
def compute_utility_score_for_maximizing_volume(x_tries, gp, dim, max_lcb,
                                                bounds_n, bounds_o, b_n,
                                                X_invasion, Y_invasion):

    # This function only works when len(x_tries) = 1
    new_bounds = bounds_n
    kappa = b_n
    mean, std = gp.predict(x_tries, return_std=True)
    std[std < 1e-10] = 0

    myucb = mean + kappa*std
    myucb = np.ravel(myucb)

    if np.asscalar(myucb) < np.asscalar(max_lcb):
        return bounds_n, myucb, X_invasion, Y_invasion

    # Check if it is outside the old bound
    x_tries = x_tries.ravel()
    flagOutside = 0
    for d in range(dim):
        if (x_tries[d] > bounds_o[d, 1]) | (x_tries[d] < bounds_o[d, 0]):
            flagOutside = 1
            break

    if flagOutside == 1:  # append to the invasion set
        if len(X_invasion) == 0:
            X_invasion = x_tries
            Y_invasion = myucb
        else:
            X_invasion = np.vstack((X_invasion, x_tries))
            Y_invasion = np.vstack((Y_invasion, myucb))

    # Expand the bound
    for d in range(dim):
        # expand lower bound
        if x_tries[d] < new_bounds[d, 0]:
            new_bounds[d, 0] = x_tries[d]

        if x_tries[d] > new_bounds[d, 1]:
            new_bounds[d, 1] = x_tries[d]

    bounds = new_bounds

    return bounds, myucb, X_invasion, Y_invasion


# This function is for GPUCB-FBO - Nguyen et al paper (ICDM 2017)
def compute_utility_score_for_max_vol_wrapper(x_tries, gp, dim,
                                              max_lcb, bounds_n,
                                              bounds, b_n,
                                              X_invasion,
                                              Y_invasion):
    if len(x_tries.shape) == 1:
        return compute_utility_score_for_maximizing_volume(x_tries, gp, dim,
                                                           max_lcb, bounds_n,
                                                           bounds, b_n,
                                                           X_invasion,
                                                           Y_invasion)

    return np.apply_along_axis(compute_utility_score_for_maximizing_volume, 1,
                               x_tries, gp, dim, max_lcb, bounds_n, bounds,
                               b_n, X_invasion, Y_invasion)


# This function is for GPUCB-FBO - Nguyen et al paper (ICDM 2017)
def max_volume(gp, bounds_n, bounds_o, max_lcb, b_n):

    bounds_temp = bounds_o.copy()
    dim = bounds_n.shape[0]
    # multi start
    for i in range(dim):
        x_tries = np.random.uniform(bounds_n[:, 0], bounds_n[:, 1],
                                    size=(500, dim))

        mean, std = gp.predict(x_tries, return_std=True)
        myucb = mean.ravel() + b_n*std.ravel()

        if np.max(myucb) < np.asscalar(max_lcb):
            return bounds_temp

        x_tries = x_tries[myucb >= max_lcb]

        # expanse the bound
        for i in range(len(x_tries)):
            for d in range(dim):
                # expand lower bound
                if x_tries[i, d] < bounds_temp[d, 0]:
                    bounds_temp[d, 0] = x_tries[i, d]

                if x_tries[i, d] > bounds_temp[d, 1]:
                    bounds_temp[d, 1] = x_tries[i, d]

    return bounds_temp


def acq_maximize_fixopt(gp, b_n, bounds, acq_type='ucb'):

    # Optimization with the rectangles as bounding box
    # 1st: Warm up with random points
    x_tries = [np.random.uniform(x[0], x[1], size=5000) for x in bounds]
    x_tries = np.asarray(x_tries).T
    if acq_type == 'ucb':
        ys = acq_gp(x_tries, gp, b_n)
    elif acq_type == 'lcb':
        ys = acq_lcb(x_tries, gp, b_n)
    else:
        raise AssertionError("Acquisition function not supported!")

    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()


    # Running L-BFGS-B from x_max
    if acq_type == 'ucb':
        res = minimize(lambda x: -acq_gp(x, gp, b_n),
                       x_max, bounds=bounds, method="L-BFGS-B")
    elif acq_type == 'lcb':
        res = minimize(lambda x: -acq_lcb(x, gp, b_n),
                       x_max, bounds=bounds, method="L-BFGS-B")
    else:
        raise AssertionError("Acquisition function not supported!")

    # Store it if better than previous minimum(maximum).
    if max_acq is None or -res.fun[0] >= max_acq:
        x_max = res.x
        max_acq = -res.fun[0]


    # 2nd: Running L-BFGS-B from (250) random starting points
    x_seeds = [np.random.uniform(x[0], x[1], size=100) for x in bounds]
    x_seeds = np.asarray(x_seeds).T

    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        if acq_type == 'ucb':
            res = minimize(lambda x: -acq_gp(x, gp, b_n),
                           x_try, bounds=bounds, method="L-BFGS-B")
        elif acq_type == 'lcb':
            res = minimize(lambda x: -acq_lcb(x, gp, b_n),
                           x_try, bounds=bounds, method="L-BFGS-B")
        else:
            raise AssertionError("Acquisition function not supported!")

        # See if success
        if not res.success:
            continue

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]

    # Just in case, due to floating points technique
    x_max = np.clip(x_max, bounds[:, 0], bounds[:, 1])

    return x_max


def acq_maximize_fixopt_local(gp, b_n, bounds, acq_type='ucb'):

    # Optimization with the rectangles as bounding box
    # 1st: Warm up with random points
    x_tries = [np.random.uniform(x[0], x[1], size=5000) for x in bounds]
    x_tries = np.asarray(x_tries).T
    if acq_type == 'ucb':
        ys = acq_gp(x_tries, gp, b_n)
    elif acq_type == 'lcb':
        ys = acq_lcb(x_tries, gp, b_n)
    else:
        raise AssertionError("Acquisition function not supported!")

    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()
    
    # Running L-BFGS-B from x_max
    if acq_type == 'ucb':
        res = minimize(lambda x: -acq_gp(x, gp, b_n),
                       x_max, bounds=bounds, method="L-BFGS-B")
    elif acq_type == 'lcb':
        res = minimize(lambda x: -acq_lcb(x, gp, b_n),
                       x_max, bounds=bounds, method="L-BFGS-B")
    else:
        raise AssertionError("Acquisition function not supported!")

    # Store it if better than previous minimum(maximum).
    if max_acq is None or -res.fun[0] >= max_acq:
        x_max = res.x
        max_acq = -res.fun[0]
    

    # 2nd: Running L-BFGS-B from (250) random starting points
    x_seeds = [np.random.uniform(x[0], x[1], size=50) for x in bounds]
    x_seeds = np.asarray(x_seeds).T

    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        if acq_type == 'ucb':
            res = minimize(lambda x: -acq_gp(x, gp, b_n),
                           x_try, bounds=bounds, method="L-BFGS-B")
        elif acq_type == 'lcb':
            res = minimize(lambda x: -acq_lcb(x, gp, b_n),
                           x_try, bounds=bounds, method="L-BFGS-B")
        else:
            raise AssertionError("Acquisition function not supported!")

        # See if success
        if not res.success:
            continue

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]

    # Just in case, due to floating points technique
    x_max = np.clip(x_max, bounds[:, 0], bounds[:, 1])

    return x_max
