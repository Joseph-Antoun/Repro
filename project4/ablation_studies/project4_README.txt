This directory contains the code we used for section 4 of the write-up: Ablation Studies and Improvements
You will find one subdirectory per experiment

./1_computing_posteriors/

    Code associated to Section 4.1 "Isolate and improve the process of generating the synthetic data, and the posterior distributions"    
    
    The custom scripts that we wrote are
        experiment1_driver.py
        experiment1_save_data.py
        experiment1_plots.py

    To run the experiment
        >>> cd ./1_computing_posteriors/
        >>> python experiment1_driver.py

    You will find the outputs of our custom functions inside
        ./data
        ./figures

./2_testing_noise_naive

    Code associated to Section 4.2 "Does the Noise-naive method produce an asymptotically correct posterior"    
    
    The custom scripts that we wrote are
        experiment2_driver.py
        experiment2_save_data.py
        experiment2_plots.py

    To run the experiment
        >>> cd ./2_testing_noise_naive/
        >>> python experiment2_driver.py

    You will find the outputs of our custom functions inside
        ./data
        ./figures


