'''
Monte carlo approach
to the anharmonic quantum
oscillator
'''

import random as rnd
import numpy as np

import utility_custom
import utility_monte_carlo as mc


def monte_carlo_ao(n_lattice,  # size of the grid
                   n_equil,  # equilibration sweeps
                   n_mc_sweeps,  # monte carlo sweeps
                   n_points,  #
                   n_meas,  #
                   i_cold):  # cold/hot start):
    '''Solve the anharmonic oscillator through
    Monte Carlo technique on an Euclidian Axis'''

    if n_mc_sweeps < n_equil:
        print("too few Monte Carlo sweeps/ N_equilib > N_Monte_Carlo")
        return 0

    # Control output filepath
    output_path = './output_data/output_monte_carlo'
    utility_custom.output_control(output_path)

    # Correlation functions
    x_cor_sums = np.zeros((3, n_points))
    x2_cor_sums = np.zeros((3, n_points))

    # x position along the tau axis

    x_config = mc.initialize_lattice(n_lattice, i_cold)

    # Monte Carlo sweeps: Principal cycle

    # Equilibration cycle
    for _ in range(n_equil):
        
        mc.metropolis_question(x_config)

        # Rest of the MC sweeps
    with open(output_path + '/ground_state_histogram.dat', 'wb') as hist_writer:
        for _ in range(n_mc_sweeps - n_equil):

            mc.metropolis_question(x_config)

            np.save(hist_writer, x_config[0:(n_lattice - 1)])
            for _ in range(n_meas):
                i_p0 = int((n_lattice - n_points) * rnd.uniform(0., 1.))
                x_0 = x_config[i_p0]
                for i_point in range(n_points):
                    x_1 = x_config[i_p0 + i_point]

                    x_cor_sums[0, i_point] += x_0 * x_1
                    x_cor_sums[1, i_point] += pow(x_0 * x_1, 2)
                    x_cor_sums[2, i_point] += pow(x_0 * x_1, 3)

                    x2_cor_sums[0, i_point] += pow(x_0 * x_1, 2)
                    x2_cor_sums[1, i_point] += pow(x_0 * x_1, 4)
                    x2_cor_sums[2, i_point] += pow(x_0 * x_1, 6)

    # Evaluate averages and other stuff, maybe we can create a function

    utility_custom.\
        output_correlation_functions_and_log(n_points,
                                             x_cor_sums,
                                             x2_cor_sums,
                                             (n_mc_sweeps-n_equil) * n_meas,
                                             output_path)

    return 1
