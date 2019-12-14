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



