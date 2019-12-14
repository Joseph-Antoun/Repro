This directory contains the code we used for section 3 and 5 of the write-up: Ablation Studies and Improvements
The subdirectories are listed bellow

./mmd_metric/

    Custom functions to compute the MMD^2 metric, and plot the results.

    The custom scripts that we wrote are
        new_driver.py
        new_plots.py
        save_data.py

    To run the experiment - the computation of the MMD metruc will take some time
        >>> cd ./mmd_metric/
        >>> python new_driver.py

    You will find the outputs of our custom functions inside
        ./data
        ./figures

./KS_statistic

    Custom functions to compute the KS statistic, and plot the results.

    The custom scripts that we wrote are
        new_driver.py 
        new_plots.py
        save_data.py

    Please note that these are *not* the same scripts as in ./mmd_metric/

    To run the experiment
        >>> cd ./KS_statistic
        >>> python new_driver.py

    You will find the outputs of our custom functions inside
        ./data/
            /n
            /epsilon
        ./figures
            /n
            /epsilon


