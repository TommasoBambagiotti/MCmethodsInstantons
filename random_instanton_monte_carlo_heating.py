'''
Heating based on the instanton
random liqud model
'''

import numpy as np

import utility_custom
import utility_rilm as rilm
from utility_monte_carlo import two_loop_density
import time


def random_instanton_liquid_model_heating(n_lattice,  # size of the grid
                                          n_mc_sweeps,  # monte carlo sweeps
                                          n_points,  #
                                          n_meas,
                                          n_heating,
                                          n_ia=0,
                                          x_potential_minimum=1.4,
                                          dtau=0.05,
                                          delta_x=0.5):
    """Compute correlation functions of the anharmonic oscillator using
    a random instanton ensemble including non-Gaussian fluctuations around
    the semi-classical path.

    The non-Gaussian corrections to the Gaussian potential (second order
    in the path variation dx) are computed using the heating method.

    We use a system of unit of measurements where h_bar=1, m=1/2 and
    lambda=1.

    Parameters
    ----------
    n_lattice : int
        Number of lattice point in euclidean time.
    n_mc_sweeps : int
        Number of Monte Carlo sweeps.
    n_points : int
        Number of points on which correlation functions are computed.
    n_meas : int
        Number of measurement of correlation functions in a MC sweep.
    n_heating : int
        Number of heating sweeps.
    n_ia : int, default=0
        Number of instantons and anti-instantons. If 0, n_ia is computed
        at 2-loop order. It has to be even.
    x_potential_minimum : float, default=1.4
        Position of the minimum(a) of the anharmonic potential.
    dtau : float, default=0.05
        Lattice spacing.
    delta_x : float, default=0.5
        Width of Gaussian distribution for Metropolis update.

    Returns
    ----------
    None
    """
    # Control output filepath
    output_path = './output_data/output_rilm_heating'
    utility_custom.output_control(output_path)

    # Eucliadian time
    tau_array = np.linspace(0.0, n_lattice * dtau, n_lattice, False)

    # Correlation functions
    x_cor_sums = np.zeros((3, n_points))
    x2_cor_sums = np.zeros((3, n_points))

    start = time.time()

    if n_ia == 0:
        # n_ia evaluated from 2-loop semi-classical expansion
        n_ia = int(np.rint(two_loop_density(x_potential_minimum)
                           * n_lattice * dtau))

    for i_mc in range(n_mc_sweeps):
        if i_mc % 10000 == 0:
            print(f'#{i_mc} sweep in {n_mc_sweeps - 1}')

        # print last config
        if i_mc % int(n_mc_sweeps-1) == 0:
            x_ansatz , x_ansatz_heated = rilm.rilm_heated_monte_carlo_step(n_ia,
                                          n_heating,
                                          n_points,
                                          n_meas,
                                          tau_array,
                                          x_cor_sums,
                                          x2_cor_sums,
                                          x_potential_minimum,
                                          dtau,
                                          delta_x)
            with open(output_path + '/x1_config.txt',
                      'w') as f_w:
                np.savetxt(f_w, x_ansatz)
            with open(output_path + '/x2_config.txt',
                      'w') as f_w:
                np.savetxt(f_w, x_ansatz_heated)
        else:
            _, _ = rilm.rilm_heated_monte_carlo_step(n_ia,
                                          n_heating,
                                          n_points,
                                          n_meas,
                                          tau_array,
                                          x_cor_sums,
                                          x2_cor_sums,
                                          x_potential_minimum,
                                          dtau,
                                          delta_x)


    utility_custom. \
        output_correlation_functions_and_log(n_points,
                                             x_cor_sums,
                                             x2_cor_sums,
                                             n_mc_sweeps * n_meas,
                                             output_path,
                                             dtau)

    end = time.time()
    print(f'Elapsed: {end - start}')