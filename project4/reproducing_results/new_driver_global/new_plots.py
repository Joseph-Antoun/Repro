import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from sklearn.linear_model import LinearRegression
from scipy import stats
import math

import torch
import NIW

sns.set()


def save_true_params(true_params, csv_name):

    param_list  = ["theta_0", "theta_bias", "sigma_squared", "mu_x", "tau"]
    params      = pd.DataFrame(index=param_list, columns=["true_value"])

    params.at["theta_0"]        = true_params['theta'][0][0]
    params.at["theta_bias"]     = true_params['theta'][1][0]
    params.at["sigma_squared"]  = true_params['sigma_squared']
    params.at["mu_x"]           = true_params['mu_x'][0][0]
    params.at["tau"]            = true_params['Tau'][0][0]

    params.to_csv(csv_name, index=True, index_label="param")
    print("File %s is ready." % csv_name)



def plot_data_distributions(true_params, data, N, posteriors, img_name):

    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(14,9)

    #------------------------------------------------------
    # First plot: true data
    # Plot frequentist OLS line + parameter estimates
    #------------------------------------------------------
    X = pd.DataFrame(data['X'])
    y = pd.DataFrame(data['y'])

    linear_regressor = LinearRegression()
    linear_regressor.fit(X, y)
    y_pred = linear_regressor.predict(X)

    ax = axs[0,0]
    ax.plot(data['X'], data['y'], 'o', label="True data")
    ax.plot(data['X'], y_pred, color='red', label="Frequentist Linear Regression")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("True Synthetic Data \n" + r"N=%s, Frequentist OLS params: $\theta_0$=%.2f, $\theta_{bias}$=%.2f" %  
        (
            N,
            linear_regressor.coef_[0][0],
            linear_regressor.intercept_[0]
        )
    )
    ax.legend()



    #------------------------------------------------------
    # Second plot
    # Posterior distributions for theta_0 plus true value
    #------------------------------------------------------
    ax = axs[0,1]

    for method in posteriors.keys():
        ax.hist(
            posteriors[method]['theta_0'], 
            bins=50, alpha=0.4, label=method,
            linewidth=1.5, histtype='step', stacked=True, fill=False
        )
    true_value = float(true_params[true_params['param']=='theta_0']['true_value'])
    ax.axvline(x=true_value, label="True Value")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(r"Posterior distributions for $\theta_0$ by method")
    ax.legend()

    #------------------------------------------------------
    # Third plot
    # Posterior distributions for theta_bias plus true value
    #------------------------------------------------------
    ax = axs[1,0]

    for method in posteriors.keys():
        ax.hist(
            posteriors[method]['theta_bias'], 
            bins=50, alpha=0.4, label=method,
            linewidth=1.5, histtype='step', stacked=True, fill=False
        )
    true_value = float(true_params[true_params['param']=='theta_bias']['true_value'])
    ax.axvline(x=true_value, label="True Value")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(r"Posterior distributions for $\theta_bias$ by method")
    ax.legend()

    #------------------------------------------------------
    # Fourth plot
    # Posterior distributions for sigma_squared
    #------------------------------------------------------
    ax = axs[1,1]

    for method in posteriors.keys():
        ax.hist(
            posteriors[method]['sigma_squared'], 
            bins=50, alpha=0.4, label=method,
            linewidth=1.5, histtype='step', stacked=True, fill=False
        )
    true_value = float(true_params[true_params['param']=='sigma_squared']['true_value'])
    ax.axvline(x=true_value, label="True Value")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(r"Posterior distributions for $\sigma_squared$ by method")
    ax.legend()

    plt.savefig(img_name)
    print("Plot %s is ready" % img_name)
    plt.close()

   


def save_data(X, y, data_dim, N, csv_name):

    if data_dim != 1:
        print("save_data() only works with data_dim = 1 for now. Skipping this step.")
        return 1

    data = pd.DataFrame({'X': X[:,0].flatten(), 'y': y.flatten()})
    data.to_csv(csv_name, index=False)
    print("File %s is ready" % csv_name)


def save_posteriors(posteriors, data_dim, N, epsilon, dir_name):

    if data_dim != 1:
        print("save_posteriors() only works with data_dim = 1 for now. Skipping this step.")
        return 1

    # Posterior parameter distributions to be saved
    params = ['theta_0', 'theta_bias', 'sigma_squared']

    # Methods that were compared in this experiment
    methods = list(posteriors.keys())

    for method in posteriors:
        
        csv_name = "%s/%s_posteriors_n%s_e%s.csv" % (dir_name, method, N, epsilon)
        
        # Posterior distribution for current method and parameter
        theta_0_dist        = posteriors[method][0][:,0]
        theta_bias_dist     = posteriors[method][0][:,0]
        sigma_squared_dist  = posteriors[method][1]

        df = pd.DataFrame({
            'theta_0'       : posteriors[method][0][:,0].flatten(), 
            'theta_bias'    : posteriors[method][0][:,0].flatten(),
            'sigma_squared' : posteriors[method][1].flatten()
        })

        df.to_csv(csv_name, index=False)
        print("Posterior distributions saved in %s" % csv_name)


def snk(x, y, sigma=1.0):
    """
    Standard normal kernel as per the paper
    """
    ratio1  = 1.0 / (2.0 * math.pi * math.pow(sigma,2))
    ratio2  = (math.pow(x,2) + math.pow(y,2)) / (2.0*math.pow(sigma,2))
    exp     = math.exp(-ratio2)
    return ratio1 * exp


def mmd_squared(P,Q):
    """
    Use the formula for MMD squared provided in the paper
    to compute the maximum mean discrepancy score
    """
    m       = P.shape[0]
    ratio   = 1.0 / (m*(m-1.0))
    sigma   = 0.0

    m = 300
    for i in range(m):
        for j in range(m):
            if i != j:
                sigma = sigma + snk(P[i],P[j]) + snk(Q[i],Q[j]) - snk(P[i],Q[j]) - snk(P[j],Q[i])

    return ratio * sigma


def plot_MMD(posteriors, var, figname):
    # Dictionnary for scientific display
    dic = {
        'n'             : r"$N$",
        'e'             : r"$\epsilon$",
        'theta_0'       : r"$\theta_{0}$",
        'theta_bias'    : r"$\theta_{bias}$",
        'sigma_squared' : r"$\sigma^{2}$",
    }

    params = ['theta_0', 'theta_bias', 'sigma_squared']

    # Three plots, one per parameter
    fig, axs = plt.subplots(1, len(params), sharey=True)
    fig.set_size_inches(14, 5)

    # We are interested in the Maximum-Mean Discreptancy between
    # the non-private posterior distribution and each of the other
    # private methods
    i = 0
    for param in params:
        df = pd.DataFrame(columns=['threshold', 'method', 'mmd'])
        j  = 0

        # Loop through each value (either N or epsilon)
        for threshold in posteriors.keys():
            post    = posteriors[threshold]
            methods = [m for m in post.keys() if m != 'non-private']
            # non-private posterior distribution
            non_private = post['non-private'][param]

            # other methods 
            for method in methods:
                post_method = post[method][param]
                # Maximum-Mean Discreptancy
                mmd2 = mmd_squared(non_private, post_method)
                # Add new mmd to the dataframe
                df.at[j, 'threshold']   = threshold
                df.at[j, 'method']      = method
                df.at[j, 'mmd']         = mmd2
                j = j + 1
        
        # Plot the statistics for the current parameter
        ax = axs[i]
        for method in methods:
            x = df[df['method'] == method]['threshold'].to_numpy()
            y = df[df['method'] == method]['mmd'].to_numpy()
            ax.plot(x, y, '-o', label=method)
            
        ax.set_title(dic[params[i]])
        ax.set_ylabel("MMD")
        ax.legend()
        i = i + 1

    plt.xticks([])
    plt.savefig(figname)
    print("Plot %s is ready" % figname)
    plt.close()



def plot_KS_statistic(posteriors, var, figname):

    # Dictionnary for scientific display
    dic = {
        'n'             : r"$N$",
        'e'             : r"$\epsilon$",
        'theta_0'       : r"$\theta_{0}$",
        'theta_bias'    : r"$\theta_{bias}$",
        'sigma_squared' : r"$\sigma^{2}$",
    }

    params = ['theta_0', 'theta_bias', 'sigma_squared']

    # Three plots, one per parameter
    fig, axs = plt.subplots(1, len(params), sharey=True)
    fig.set_size_inches(14, 5)

    # We are interested in the Kolmogorov-Smirnov statistic between
    # the non-private posterior distribution and each of the other
    # private methods
    i = 0
    for param in params:
        df = pd.DataFrame(columns=['threshold', 'method', 'ks_stat'])
        j  = 0

        # Loop through each value (either N or epsilon)
        for threshold in posteriors.keys():
            
            post    = posteriors[threshold]
            methods = [m for m in post.keys() if m != 'non-private']

            # non-private posterior distribution
            non_private = post['non-private'][param]

            # other methods 
            for method in methods:
                post_method = post[method][param]

                # Kolmogorov-Smirnov statistic
                ks_stat, p_value = stats.ks_2samp(non_private, post_method)

                # Add new KS-stat to the dataframe
                df.at[j, 'threshold']   = threshold
                df.at[j, 'method']      = method
                df.at[j, 'ks_stat']     = ks_stat
                j = j + 1
        
        # Plot the statistics for the current parameter
        ax = axs[i]
        for method in methods:
            x = df[df['method'] == method]['threshold'].to_numpy()
            y = df[df['method'] == method]['ks_stat'].to_numpy()
            ax.plot(x, y, '-o', label=method)
            
        ax.set_title(dic[params[i]])
        ax.set_ylabel("KS Stat.")
        ax.legend()
        i = i + 1

    plt.xticks([])
    plt.savefig(figname)
    print("Plot %s is ready" % figname)
    plt.close()



def plot_posterior_params(df, figname):

    # Dictionnary for scientific display
    dic = {
        'n'             : r"$N$",
        'e'             : r"$\epsilon$",
        'theta_0'       : r"$\theta_{0}$",
        'theta_bias'    : r"$\theta_{bias}$",
        'sigma_squared' : r"$\sigma^{2}$",
    }

    # Value that is looped over
    var = [x for x in df.columns if x not in ['param', 'method', 'value']]
    if len(var) != 1:
        print("Error in plot_posterior_params(): length of var must be 1")
        return 1

    var         = var[0]
    methods     = list(set(df['method']))
    params      = list(set(df['param']))
    n_params    = len(params)

    # Three plots, one per parameter
    fig, axs = plt.subplots(1, n_params, sharey=True)
    fig.set_size_inches(14, 5)

    for i in range(n_params):
        # Curent parameter (subplot)
        ax = axs[i]
        
        for m in methods:
            subset = df[(df['method'] == m) & (df['param'] == params[i])]
            ax.plot(subset[var], subset['value'], '--o', label=m)

        ax.set_title(dic[params[i]])
        ax.set_xscale('log')
        ax.set_xlabel(dic[var])
        ax.legend()

    plt.savefig(figname)
    print("Plot %s is ready" % figname)
    plt.close()



def plot_runtimes(df, figname):

    var     = 'n'
    methods = list(set(df['method']))

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4)

    for m in methods:
        subset = df[df['method'] == m]
        ax.plot(subset[var], subset['runtime'], '--o', label=m)

    ax.set_title("Runtimes per method")
    ax.set_xscale('log')
    ax.legend()

    plt.savefig(figname)
    print("Plot %s is ready" % figname)
    plt.close()



def plot_qq(posteriors, figname):

    print("plot_qq")

    data_prior_params = [np.array([[0]]), np.array([[1]]), 1, 50]
    N = 1000
    num_trials = 10

    Us = np.empty((num_trials, 2))

    for t in range(num_trials):
        mu_x, tau_squared = NIW.NIW_rvs(*data_prior_params)
        X = np.random.multivariate_normal(mu_x.flatten(), tau_squared, size=N)
        X = np.hstack((X, np.ones(X.shape)))
        S = {'X': sum(X)[:, None],
             'XX': X.T.dot(X)}

        # non-private posterior
        posterior_mu_x, posterior_tau = NIW.NIW_conjugate_update(S, data_prior_params, N, size=2000)

        print(posterior_tau.shape)
        print(posterior_mu_x[0])
        print(tau_squared)
        print(np.array(posterior_tau).squeeze())
        sys.exit(0)
        Us[t, 0] = np.sum(np.array(posterior_mu_x).squeeze() < mu_x.flatten()) / 2000
        Us[t, 1] = np.sum(np.array(posterior_tau).squeeze() < tau_squared) / 2000

    #fig, axes = plt.subplots(ncols=2)
    #for p in range(2):

    #    ax = axes[p]

    #    ax.plot([0, 1], [0, 1], '--', color='gray')
    #    ax.plot(np.array(range(num_trials)) / float(num_trials), sorted(Us[:, p]))

    #    ax.set_xlabel('rank sort position')
    #    if p == 0:
    #        ax.set_ylabel('cdf of true parameter')
    #    ax.set_xlim((0, 1))
    #    ax.set_ylim((0, 1))
    #    ax.set(aspect='equal')  # , adjustable='box-forced'

    #plt.savefig(figname, bbox_inches='tight')
    #print("QQ plot saved as %s" % figname)
    #plt.close(fig)



def main():

    plot_posterior_params_e()


if __name__ == "__main__":
    main()

