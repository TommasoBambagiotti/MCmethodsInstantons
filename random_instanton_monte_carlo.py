'''
Random instanton-anti instanton
gas model
'''

import numpy as np
import utility_custom
import utility_rilm as rilm
from utility_monte_carlo import two_loop_density


def random_instanton_liquid_model(n_lattice,
                                  n_mc_sweeps,
                                  n_points,
                                  n_meas,
                                  n_ia=0,
                                  x_potential_minimum=1.4,
                                  dtau=0.05):
    """Compute correlation functions of the anharmonic oscillator
    using a random ensemble of instantons and the instanton/anti-instan-
    ton zero crossing distribution.

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
    n_ia : int, default=0
        Number of instantons and anti-instantons. If 0, n_ia is computed
        at 2-loop order. It has to be even.
    x_potential_minimum : float, default=1.4
        Position of the minimum(a) of the anharmonic potential.
    dtau : float, default=0.05
        Lattice spacing.

    Returns
    ---------
    None

    Notes
    ---------
    We use a system of unit of measurements where h_bar=1, m=1/2 and
    lambda=1.
    """
    # Control output filepath
    output_path = './output_data/output_rilm'
    utility_custom.output_control(output_path)

    # Euclidean time
    tau_array = np.linspace(0.0, n_lattice * dtau, n_lattice, False)

    # Correlation functions
    x_cor_sums = np.zeros((3, n_points))
    x2_cor_sums = np.zeros((3, n_points))

    if n_ia == 0:
        # n_ia evaluated from 2-loop semi-classical expansion
        n_ia = int(np.rint(two_loop_density(x_potential_minimum)
                           * n_lattice * dtau))
    
    hist_writer = open(output_path +'/zcr_hist.txt','w')

    # Monte Carlo simulation
    for i_mc in range(n_mc_sweeps):
        if i_mc % 100 == 0:
            print(f'#{i_mc} sweep in {n_mc_sweeps - 1}')
            
        tau_centers_ia = rilm.rilm_monte_carlo_step(n_ia,
                                   n_points,
                                   n_meas,
                                   tau_array,
                                   x_cor_sums,
                                   x2_cor_sums,
                                   x_potential_minimum,
                                   dtau)

        # construct the i/a zero crossing distribution
        for i in range(0, n_ia, 2):
            if i == 0:
                zero_m = tau_centers_ia[-1] - n_lattice * dtau
            else:
                zero_m = tau_centers_ia[i-1]
                
            z_ia = min((tau_centers_ia[i+1]-tau_centers_ia[i]),
                       (tau_centers_ia[i] - zero_m))

            hist_writer.write(str(z_ia) + '\n')

    hist_writer.close()

    # compute correlation functions
    utility_custom.\
        output_correlation_functions_and_log(n_points,
                                             x_cor_sums,
                                             x2_cor_sums,
                                             n_mc_sweeps * n_meas,
                                             output_path,
                                             dtau)
