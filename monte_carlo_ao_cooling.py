'''
Calculation of correlation functions
using the cooling method to estrapolte 
instantons - anti instantons configurations
'''

import numpy as np
import utility_monte_carlo as mc
import utility_custom
import random as rnd
import input_parameters as ip


def cooled_monte_carlo(
        n_lattice,
        n_equil,
        n_mc_sweeps,
        n_points,
        n_meas,
        i_cold,
        n_sweeps_btw_cooling,
        n_cooling_sweeps):
    
    # number of total cooling processes
    n_cooling = 0

    if n_mc_sweeps < n_equil:
        print("too few Monte Carlo sweeps/ N_equilib > N_Monte_Carlo")
        return 0

    # Control output filepath
    output_path = './output_data/output_cooled_monte_carlo'
    utility_custom.output_control(output_path)

    # Correlation functions
    x_cor_sums = np.zeros((3, n_points))
    x2_cor_sums = np.zeros((3, n_points))
    x_cold_cor_sums = np.zeros((3, n_points))
    x2_cold_cor_sums = np.zeros((3, n_points))

    # x position along the tau axis
    x_config = mc.initialize_lattice(n_lattice, i_cold)

    # Monte Carlo sweeps: Principal cycle

    # Equilibration cycle
    for i_equil in range(n_equil):
        mc.metropolis_question(x_config)

    # Rest of the MC sweeps
    for i_mc in range(n_mc_sweeps - n_equil):
        mc.metropolis_question(x_config)
        
        # COOLING

        if (i_mc % n_sweeps_btw_cooling) == 0:

            # expected number of cooled configuration = n_conf/n_sweeps_btw_cooling
            print(f'cooling #{n_cooling} of {(n_mc_sweeps - n_equil) / n_sweeps_btw_cooling}\n'
                      f'in configuration #{i_mc}')

            x_cold_config = np.copy(x_config)
            n_cooling += 1

            for i_cooling in range(n_cooling_sweeps):
                mc.configuration_cooling(x_cold_config,
                                         ip.x_potential_minimum)

            # Compute correlation functions for the cooled configuration
            for k_meas in range(n_meas):

                i_p0 = int((n_lattice - n_points) * rnd.uniform(0., 1.))
                x_0 = x_cold_config[i_p0]

                for i_point in range(n_points):
                    x_1 = x_cold_config[i_p0 + i_point]

                    x_cold_cor_sums[0, i_point] += x_0 * x_1
                    x_cold_cor_sums[1, i_point] += pow(x_0 * x_1, 2)
                    x_cold_cor_sums[2, i_point] += pow(x_0 * x_1, 3)

                    x2_cold_cor_sums[0, i_point] += pow(x_0 * x_1, 2)
                    x2_cold_cor_sums[1, i_point] += pow(x_0 * x_1, 4)
                    x2_cold_cor_sums[2, i_point] += pow(x_0 * x_1, 6)

        # End of Montecarlo simulation

    # Correlation functions

    utility_custom.\
        output_correlation_functions_and_log(n_points, 
                                             x_cor_sums,
                                             x2_cor_sums,
                                             n_cooling * n_meas,
                                             output_path)

    return 1
