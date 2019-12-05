=======================
Error #1
=======================

when running >>> python driver.py

Traceback (most recent call last):
  File "driver.py", line 33, in <module>
    main()
  File "driver.py", line 27, in main
    posteriors = run_methods(data_prior_params, model_prior_params, X, y, sensitivity_x, sensitivity_y, epsilon, N, methods)
  File "/home/cann/work/my-notebooks/mcgill/private_bayesian_regression/run_methods.py", line 27, in run_methods
    posteriors['mcmc'] = run_mcmc(model_prior_params, data_prior_params, S, N, sensitivity_x, sensitivity_y, epsilon)
  File "/home/cann/work/my-notebooks/mcgill/private_bayesian_regression/run_methods.py", line 57, in run_mcmc
    theta, sigma_squared = mcmc(model_prior_params, data_prior_params, N, epsilon, Z, sensitivity, 2000)
  File "/home/cann/work/my-notebooks/mcgill/private_bayesian_regression/mcmc.py", line 32, in mcmc
    tau_squared = InverseGamma('ts', alpha=data_prior_params[2], beta=data_prior_params[3][0, 0])
TypeError: 'int' object is not subscriptable

	=======================
		Bug Fix
	=======================
	the variable data_prior_params is equal to [array([[0]]), 1, array([[1]]), 50]
	Replace
       	tau_squared = InverseGamma('ts', alpha=data_prior_params[2], beta=data_prior_params[3][0, 0])
	By 
        tau_squared = InverseGamma('ts', alpha=data_prior_params[1], beta=data_prior_params[3])


=======================
Error #2
=======================

Same type of error for line 
mu_x = Deterministic('mu', data_prior_params[0][0, 0] + mu_x_offset * pm.math.sqrt(tau_squared / data_prior_params[1][0, 0])

	=======================
		Bug Fix
	=======================
	the variable data_prior_params is equal to [array([[0]]), 1, array([[1]]), 50]
	Replace
       	mu_x = Deterministic('mu', data_prior_params[0][0, 0] + mu_x_offset * pm.math.sqrt(tau_squared / data_prior_params[1][0, 0])
	By 
        mu_x = Deterministic('mu', data_prior_params[0][0, 0] + mu_x_offset * pm.math.sqrt(tau_squared / data_prior_params[2][0, 0]))




