#------------------------------------------------------------------------------
# Main script for this directory
#------------------------------------------------------------------------------
from generate_data import setup_data
from run_methods import run_methods

from save_data import save_data, save_posteriors, save_true_params
from new_plots import plot_data_distributions, plot_runtimes, plot_MMD

import sys
import numpy as np
import pandas as pd


def run_experiments(methods):

    # dimension of covariate data
    data_dim = 1

    # Loop over the number of individuals N
    for N in [10, 100, 1000]:

        # privacy setting
        epsilon = .01
        # generate the prior parameters
        data_prior_params, model_prior_params, X, y, sensitivity_x, sensitivity_y, true_params = setup_data(data_dim, N)
        # save the true parameters in a csv file
        save_true_params(true_params, "./data/true_params_n%s_e%s.csv" % (N,epsilon))
        # save the synthetic data
        save_data(X, y, data_dim, N, "./data/synthetic_data_n%s_e%s.csv" % (N,epsilon))
        # Loop through each method and compute the posteriors
        posteriors = run_methods(
            data_prior_params, model_prior_params, 
            X, y, 
            sensitivity_x, sensitivity_y, 
            epsilon, N, 
            methods,
            "./data/runtimes_n%s_e%s.csv" % (N,epsilon)
        )
        # Save the posterior distributions as pandas dataframes, then in one csv file per method
        save_posteriors(posteriors, data_dim, N, epsilon, "./data")


def plot_experiments(methods):

    runtimes_lst = []
    params_means = pd.DataFrame(index=range(9*len(methods)), columns=['param', 'method', 'n', 'value'])
    posteriors   = {}
    epsilon = .01
    i = 0

    for N in [10, 100, 1000]:


        # read the synthetic data, the true parameter values and the runtimes file
        data        = pd.read_csv("./data/synthetic_data_n%s_e%s.csv" % (N,epsilon))
        true_params = pd.read_csv("./data/true_params_n%s_e%s.csv" % (N,epsilon))
        runtimes    = pd.read_csv("./data/runtimes_n%s_e%s.csv" % (N,epsilon))
        posterior   = {}
        
        # Reconstruct the posterior dictionnary, each key corresponds to a method
        for method in methods:
            posterior[method] = pd.read_csv("./data/%s_posteriors_n%s_e%s.csv" % (method, N, epsilon))
            
            for param in posterior[method].columns:
                params_means.at[i, 'param']     = param
                params_means.at[i, 'method']    = method
                params_means.at[i, 'n']         = N
                params_means.at[i, 'value']     = posterior[method][param].mean()
                i = i + 1

        # reconstruct the runtimes per method per iteration
        runtimes['n'] = N
        runtimes_lst.append(runtimes)
        # Add to the list of posterior distributions that will be used to compute the KN statistic
        posteriors['%s'%N] = posterior

    # Plot the custom MMD values
    plot_MMD(posteriors, 'n', "./figures/mmd.png")
    # Plot the runtimes for this particular experiment
    plot_runtimes(pd.concat(runtimes_lst, axis=0), "./figures/runtimes_n%s_e%s.png" % (N, epsilon))



def main():

    methods  = ['non-private', 'naive', 'mcmc', 'gibbs-noisy', 'gibbs-update']
    #run_experiments(methods)
    plot_experiments(methods)


if __name__ == '__main__':
    main()


