import numpy as np
import pandas as pd
import itertools

from generate_data import privatize_suff_stats
from model_conjugate_update import run_non_private, run_naive
from mcmc import mcmc
from Gibbs import Gibbs
import sys
import time


def run_methods(data_prior_params, model_prior_params, X, y, sensitivity_x, sensitivity_y, epsilon, N, methods, runtimes_csv=None):

    # sufficient statistics
    S = {'XX': X.T.dot(X),
         'Xy': X.T.dot(y),
         'yy': y.T.dot(y)[0, 0]
         }

    posteriors = {}

    # Track the execution times
    runtimes = pd.DataFrame(np.nan, index=list(methods), columns=['runtime'])

    if 'non-private' in methods:
        print("===========================")
        print("Running non-private")
        print("===========================")

        start = time.time()
        posteriors['non-private'] = run_non_private(model_prior_params, S, N)
        end = time.time()
        runtimes.at['non-private', 'runtime'] = end - start

    if 'naive' in methods:
        print("===========================")
        print("Running naive")
        print("===========================")

        start = time.time()
        posteriors['naive'] = run_naive(model_prior_params, S, N, sensitivity_x, sensitivity_y, epsilon)
        end = time.time()
        runtimes.at['naive', 'runtime'] = end - start

    if 'mcmc' in methods:
        print("===========================")
        print("Running mcmc")
        print("===========================")

        start = time.time()
        posteriors['mcmc'] = run_mcmc(model_prior_params, data_prior_params, S, N, sensitivity_x, sensitivity_y, epsilon)
        end = time.time()
        runtimes.at['mcmc', 'runtime'] = end - start

    if 'gibbs-noisy' in methods:
        print("===========================")
        print("Running gibbs-noisy")
        print("===========================")

        start = time.time()
        posteriors['gibbs-noisy'] = run_gibbs_noisy(X,
                                                    S,
                                                    sensitivity_x,
                                                    sensitivity_y,
                                                    epsilon,
                                                    data_prior_params,
                                                    model_prior_params,
                                                    N,
                                                    )
        end = time.time()
        runtimes.at['gibbs-noisy', 'runtime'] = end - start

    if 'gibbs-update' in methods:
        print("===========================")
        print("Running gibbs-update")
        print("===========================")

        start = time.time()
        posteriors['gibbs-update'] = run_gibbs_update(S,
                                                      sensitivity_x,
                                                      sensitivity_y,
                                                      epsilon,
                                                      data_prior_params,
                                                      model_prior_params,
                                                      N,
                                                      )
        end = time.time()
        runtimes.at['gibbs-update', 'runtime'] = end - start


    #--------------------------------------------------------------------------
    # Save the runtimes
    #--------------------------------------------------------------------------
    print(runtimes)
    if runtimes_csv is not None:
        runtimes.to_csv(runtimes_csv, index=True, index_label='method')
        print("Runtimes saved in %s" % runtimes_csv)

    return posteriors


def run_mcmc(model_prior_params, data_prior_params, S, N, sensitivity_x, sensitivity_y, epsilon):

    Z, sensitivity = privatize_suff_stats(S, sensitivity_x, sensitivity_y, epsilon)

    theta, sigma_squared = mcmc(model_prior_params, data_prior_params, N, epsilon, Z, sensitivity, 2000)

    return theta, sigma_squared


def run_gibbs_noisy(X,
                    S,
                    sensitivity_x,
                    sensitivity_y,
                    epsilon,
                    data_prior_params,
                    model_prior_params,
                    N,
                    ):

    epsilon_moments = epsilon / 2
    epsilon_suff_stats = epsilon / 2

    Z, sensitivity_suff_stats = privatize_suff_stats(S, sensitivity_x, sensitivity_y, epsilon_suff_stats)

    d = X.shape[1]
    X_fourth_moment = np.empty((d, d, d, d))
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    X_fourth_moment[i, j, k, l] = np.sum(X[:, i] * X[:, j] * X[:, k] * X[:, l])

    component_sensitivity = np.empty((d, d, d, d))
    for i in range(d):
        si = sensitivity_x[i] if sensitivity_x[i] != 0 else 1
        for j in range(d):
            sj = sensitivity_x[j] if sensitivity_x[j] != 0 else 1
            for k in range(d):
                sk = sensitivity_x[k] if sensitivity_x[k] != 0 else 1
                for l in range(d):
                    sl = sensitivity_x[l] if sensitivity_x[l] != 0 else 1
                    component_sensitivity[i, j, k, l] = si * sj * sk * sl

    sensitivity_moments = sum(np.unique(component_sensitivity))  # don't need to take all componenets into account because we'll symmetrize

    noisy_X_fourth_moment = np.random.laplace(loc=X_fourth_moment, scale=sensitivity_moments / epsilon_moments)

    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                for l in range(k, d):
                    combos = list(set(itertools.permutations((i, j, k, l))))
                    val = noisy_X_fourth_moment[combos[0]]
                    for combo in combos:
                        noisy_X_fourth_moment[combo] = val

    noisy_X_fourth_moment[-1, -1, -1, -1] = X.shape[0] # dont' need to add noise to constant term

    # average the moments to get individual moments
    noisy_X_fourth_moment /= X.shape[0]

    theta, sigma_squared = Gibbs(data_prior_params,
                                 model_prior_params,
                                 N,
                                 epsilon_suff_stats,
                                 Z,
                                 sensitivity_suff_stats,
                                 2000,
                                 5000,
                                 'gibbs-noisy',
                                 X_fourth_moment=X_fourth_moment)

    return theta, sigma_squared


def run_gibbs_update(S,
                     sensitivity_x,
                     sensitivity_y,
                     epsilon,
                     data_prior_params,
                     model_prior_params,
                     N,
                     ):

    Z, sensitivity = privatize_suff_stats(S, sensitivity_x, sensitivity_y, epsilon)

    theta, sigma_squared = Gibbs(data_prior_params,
                                 model_prior_params,
                                 N,
                                 epsilon,
                                 Z,
                                 sensitivity,
                                 2000,
                                 5000,
                                 'gibbs-update')

    return theta, sigma_squared
