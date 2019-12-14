import numpy as np
import scipy
import scipy.stats

import matplotlib.pyplot as plt
import sys

import NIW
import NIG


if __name__ == '__main__':

    data_dim = 1
    N = 1000

    # NIG = [mu, lambda, alpha, beta]
    lambda_0 = 0.5/(20.0 - 1.0)
    model_prior_params = [np.array([0] * (data_dim + 1))[:, None],
                          np.diag([lambda_0] * (data_dim + 1)),
                          20.0,
                          0.5
                          ]

    # NIW = [mu', lambda', psi, nu]
    data_prior_params = [np.array([0] * data_dim)[:, None],
                         np.diag([1] * data_dim),
                         1,
                         50
                        ]

    print("------------------------------------------------")
    print("data_prior_params  = %s" % data_prior_params)
    print("model_prior_params = %s" % model_prior_params)
    print("------------------------------------------------\n")

    # (Theta, sigma_squared) ~ NIG([mu, lambda, alpha, beta])
    theta, sigma_squared = NIG.NIG_rvs_single_variance(*model_prior_params)

    # (mu_x, tau_squared) ~ NIW([mu', lambda', psi, nu])
    mu_x, tau_squared = NIW.NIW_rvs(*data_prior_params)

    print("------------------------------------------------")
    print("theta=%s\n sigma_squared=%s\n mu_x=%s\n tau_squared=%s\n" % (
        theta, sigma_squared, mu_x, tau_squared
    ))
    print("------------------------------------------------\n")


    # X ~ N(mu_x, tau_squared)
    X = np.random.multivariate_normal(mu_x.flatten(), tau_squared, size=N)

    # S = X, X.T.dot(X)
    S = {'X': sum(X)[:, None], 'XX': X.T.dot(X)}

    print("------------------------------------------------")
    print("X.shape",  X.shape)
    print("S = %s" % S)
    print("------------------------------------------------\n")

    # Adding noise to the data
    Z, sensitivity = privatize_suff_stats(S, sensitivity_x, sensitivity_y, epsilon)

    sys.exit(0)
    num_trials = 100

    Us = np.empty((num_trials, 2))
    for t in range(num_trials):
        mu_x, tau_squared = NIW_rvs(*data_prior_params)
        X = np.random.multivariate_normal(mu_x.flatten(), tau_squared, size=N)
        X = np.hstack((X, np.ones(X.shape)))
        S = {'X': sum(X)[:, None],
             'XX': X.T.dot(X)}

        posterior = NIW_conjugate_update(S, data_prior_params, N, size=2000)

        Us[t, 0] = np.sum(np.array(posterior[0]).squeeze() < mu_x.flatten()) / 2000
        Us[t, 1] = np.sum(np.array(posterior[1]).squeeze() < tau_squared) / 2000

    fig, axes = plt.subplots(ncols=2)
    for p in range(2):

        ax = axes[p]

        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.plot(np.array(range(num_trials)) / float(num_trials), sorted(Us[:, p]))

        ax.set_xlabel('rank sort position')
        if p == 0:
            ax.set_ylabel('cdf of true parameter')
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set(aspect='equal')  # , adjustable='box-forced'

    # Project4 Modified
    # plt.savefig('/Users/gbernstein/Desktop/QQ.png', bbox_inches='tight')
    plt.savefig('./figures/QQ.png', bbox_inches='tight')
    plt.close(fig)

