import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


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
        theta_bias_dist     = posteriors[method][0][:,1]
        sigma_squared_dist  = posteriors[method][1]

        df = pd.DataFrame({
            'theta_0'       : posteriors[method][0][:,0].flatten(), 
            'theta_bias'    : posteriors[method][0][:,1].flatten(),
            'sigma_squared' : posteriors[method][1].flatten()
        })

        df.to_csv(csv_name, index=False)
        print("Posterior distributions saved in %s" % csv_name)



