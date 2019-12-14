from generate_data import setup_data
from run_methods import run_methods

from new_plots import save_data, save_posteriors, save_true_params
from new_plots import plot_data_distributions, plot_posterior_params, plot_runtimes, plot_KS_statistic
from new_plots import plot_MMD

import sys
import numpy as np
import pandas as pd


def run_experiments(methods):

    # dimension of covariate data
    data_dim = 1

    print("#--------------------------------------------------------------------------")
    print("# EXPERIMENT A: calibration vs. n for epsilon = 0.1 ")
    print("#--------------------------------------------------------------------------")

    # Loop over the number of individuals N
    for N in [10, 100, 1000]:

        # privacy setting
        epsilon = .01

        data_prior_params, model_prior_params, X, y, sensitivity_x, sensitivity_y, true_params = setup_data(data_dim, N)
        save_true_params(true_params, "./out_csv/loop_n/true_params_n%s_e%s.csv" % (N,epsilon))

        save_data(X, y, data_dim, N, "./out_csv/loop_n/synthetic_data_n%s_e%s.csv" % (N,epsilon))

        posteriors = run_methods(
            data_prior_params, 
            model_prior_params, 
            X, 
            y, 
            sensitivity_x, 
            sensitivity_y, 
            epsilon, 
            N, 
            methods,
            "./out_csv/loop_n/runtimes_n%s_e%s.csv" % (N,epsilon)
        )

        save_posteriors(posteriors, data_dim, N, epsilon, "./out_csv/loop_n")

    print("#--------------------------------------------------------------------------")
    print("# EXPERIMENT B: calibration vs. epsilon for n = 10 ")
    print("#--------------------------------------------------------------------------")

    for epsilon in [.01, .1, 1.]:

        # number of individuals
        N = 10

        data_prior_params, model_prior_params, X, y, sensitivity_x, sensitivity_y, true_params = setup_data(data_dim, N)
        save_true_params(true_params, "./out_csv/loop_epsilon/true_params_n%s_e%s.csv" % (N,epsilon))

        save_data(X, y, data_dim, N, "./out_csv/loop_epsilon/synthetic_data_n%s_e%s.csv" % (N,epsilon))

        posteriors = run_methods(
            data_prior_params, 
            model_prior_params, 
            X, 
            y, 
            sensitivity_x, 
            sensitivity_y, 
            epsilon, 
            N, 
            methods, 
            "./out_csv/loop_epsilon/runtimes_n%s_e%s.csv" % (N,epsilon)
        )

        save_posteriors(posteriors, data_dim, N, epsilon, "./out_csv/loop_epsilon")

    print("#--------------------------------------------------------------------------")
    print("# EXPERIMENT D: Method runtimes for epsilon = 0.1 ")
    print("#--------------------------------------------------------------------------")
    for N in [10, 100, 1000]:

        # privacy setting
        epsilon = .1

        data_prior_params, model_prior_params, X, y, sensitivity_x, sensitivity_y, true_params = setup_data(data_dim, N)
        save_true_params(true_params, "./out_csv/loop_runtime/true_params_n%s_e%s.csv" % (N,epsilon))

        save_data(X, y, data_dim, N, "./out_csv/loop_runtime/synthetic_data_n%s_e%s.csv" % (N,epsilon))

        posteriors = run_methods(
            data_prior_params, 
            model_prior_params, 
            X, 
            y, 
            sensitivity_x, 
            sensitivity_y, 
            epsilon, 
            N, 
            methods,
            "./out_csv/loop_runtime/runtimes_n%s_e%s.csv" % (N,epsilon)
        )

        save_posteriors(posteriors, data_dim, N, epsilon, "./out_csv/loop_runtime")



def plot_experiments(methods):

    print("#--------------------------------------------------------------------------")
    print("# PLOT A: calibration vs. n for epsilon = 0.1 ")
    print("#--------------------------------------------------------------------------")

    runtimes_lst = []
    params_means = pd.DataFrame(index=range(9*len(methods)), columns=['param', 'method', 'n', 'value'])
    posteriors   = {}
    i = 0

    for N in [10, 100, 1000]:

        epsilon = .01

        data        = pd.read_csv("./out_csv/loop_n/synthetic_data_n%s_e%s.csv" % (N,epsilon))
        runtimes    = pd.read_csv("./out_csv/loop_n/runtimes_n%s_e%s.csv" % (N,epsilon))
        true_params = pd.read_csv("./out_csv/loop_n/true_params_n%s_e%s.csv" % (N,epsilon))
        posterior   = {}
        
        for method in methods:
            posterior[method] = pd.read_csv("./out_csv/loop_n/%s_posteriors_n%s_e%s.csv" % (method, N, epsilon))
            
            for param in posterior[method].columns:
                params_means.at[i, 'param']     = param
                params_means.at[i, 'method']    = method
                params_means.at[i, 'n']         = N
                params_means.at[i, 'value']     = posterior[method][param].mean()
                i = i + 1

        runtimes['n'] = N
        runtimes_lst.append(runtimes)

        # Add to the list of posterior distributions that will be used to compute the KN statistic
        posteriors['%s'%N] = posterior

        # Plot the parameter distributions for current run
        plot_data_distributions(true_params, data, N, posterior, "./figures/loop_n/data_n%s_e%s.png" % (N, epsilon))
    
    # Plot the runtimes for this particular experiment
    plot_runtimes(pd.concat(runtimes_lst, axis=0), "./figures/loop_n/runtimes_n%s_e%s.png" % (N, epsilon))

    # Figure a from the paper
    plot_posterior_params(params_means, "./figures/loop_n/posterior_params_n%s_e%s.png" % (N, epsilon))
    plot_KS_statistic(posteriors, 'n', "./figures/loop_n/KS_statistic_n%s_e%s.png" % (N, epsilon))
    #plot_MMD(posteriors, 'n', "./figures/loop_n/mmd_n%s_e%s.png" % (N, epsilon))

    print("#--------------------------------------------------------------------------")
    print("# PLOT B: calibration vs. epsilon for n = 10 ")
    print("#--------------------------------------------------------------------------")

    runtimes_lst = []
    params_means = pd.DataFrame(index=range(9*len(methods)), columns=['param', 'method', 'e', 'value'])
    posteriors   = {}
    i = 0

    for epsilon in [.01, .1, 1.]:

        N = 10

        data        = pd.read_csv("./out_csv/loop_epsilon/synthetic_data_n10_e%s.csv" % epsilon)
        runtimes    = pd.read_csv("./out_csv/loop_epsilon/runtimes_n10_e%s.csv" % epsilon)
        posterior   = {}
        
        for method in methods:
            posterior[method] = pd.read_csv("./out_csv/loop_epsilon/%s_posteriors_n%s_e%s.csv" % (method, N, epsilon))

            for param in posterior[method].columns:
                params_means.at[i, 'param']     = param
                params_means.at[i, 'method']    = method
                params_means.at[i, 'e']         = epsilon
                params_means.at[i, 'value']     = posterior[method][param].mean()
                i = i + 1

        runtimes['n'] = N
        runtimes_lst.append(runtimes)

        # Add to the list of posterior distributions that will be used to compute the KN statistic
        posteriors['%s'%epsilon] = posterior

        # Plot the parameter distributions for current run
        plot_data_distributions(true_params, data, N, posterior, "./figures/loop_epsilon/data_n%s_e%s.png" % (N, epsilon))
    
    # Plot the runtimes for this particular experiment
    plot_runtimes(pd.concat(runtimes_lst, axis=0), "./figures/loop_epsilon/runtimes_n%s_e%s.png" % (N, epsilon))

    # Figure a from the paper
    #plot_posterior_params(params_means, "./figures/loop_epsilon/posterior_params_n%s_e%s.png" % (N, epsilon))
    plot_KS_statistic(posteriors, 'e', "./figures/loop_epsilon/KS_statistic_n%s_e%s.png" % (N, epsilon))

    print("#--------------------------------------------------------------------------")
    print("# EXPERIMENT D: Method runtimes for epsilon = 0.1 ")
    print("#--------------------------------------------------------------------------")

    runtimes_lst = []
    params_means = pd.DataFrame(index=range(9*len(methods)), columns=['param', 'method', 'n', 'value'])
    i = 0

    for N in [10, 100, 1000]:

        epsilon = .1

        data        = pd.read_csv("./out_csv/loop_runtime/synthetic_data_n%s_e.1.csv" % N)
        runtimes    = pd.read_csv("./out_csv/loop_runtime/runtimes_n%s_e.1.csv" % N)
        posterior   = {}
        
        for method in methods:
            posterior[method] = pd.read_csv("./out_csv/loop_runtime/%s_posteriors_n%s_e%s.csv" % (method, N, epsilon))

            for param in posterior[method].columns:
                params_means.at[i, 'param']     = param
                params_means.at[i, 'method']    = method
                params_means.at[i, 'n']         = N
                params_means.at[i, 'value']     = posterior[method][param].mean()
                i = i + 1

        runtimes['n'] = N
        runtimes_lst.append(runtimes)

        # Plot the parameter distributions for current run
        plot_data_distributions(true_params, data, N, posterior, "./figures/loop_runtime/data_n%s_e%s.png" % (N, epsilon))
    
    # Plot the runtimes for this particular experiment
    plot_runtimes(pd.concat(runtimes_lst, axis=0), "./figures/loop_runtime/runtimes_n%s_e%s.png" % (N, epsilon))

    # Figure a from the paper
    plot_posterior_params(params_means, "./figures/loop_runtime/posterior_params_n%s_e%s.png" % (N, epsilon))



def main():

    #methods  = ['non-private', 'naive']
    methods  = ['non-private', 'naive', 'mcmc', 'gibbs-noisy', 'gibbs-update']

    #run_experiments(methods)
    plot_experiments(methods)


if __name__ == '__main__':
    main()


