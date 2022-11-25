'''
Monte carlo approach
to the anharmonic quantum
oscillator
'''

import numpy as np

import utility_custom
import utility_monte_carlo as mc


def monte_carlo_ao(n_lattice,  # size of the grid
                   n_equil,  # equilibration sweeps
                   n_mc_sweeps,  # monte carlo sweeps
                   n_points,  #
                   n_meas,  #
                   i_cold,
                   x_potential_minimum=1.4,
                   dtau=0.05,
                   delta_x=0.5):  # cold/hot start):
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

    x_config = mc.initialize_lattice(n_lattice,
                                     x_potential_minimum,
                                     i_cold)

    # Monte Carlo sweeps: Principal cycle

    # Equilibration cycle
    for _ in range(n_equil):

        mc.metropolis_question(x_config,
                               x_potential_minimum,
                               dtau,
                               delta_x)

        # Rest of the MC sweeps
    # with open(output_path + '/ground_state_histogram.txt', 'wb') as hist_writer:
    for i_mc in range(n_mc_sweeps - n_equil):
        if i_mc % 100 == 0:
            print(f'{i_mc} in {n_mc_sweeps - n_equil}')

        mc.metropolis_question(x_config,
                               x_potential_minimum,
                               dtau,
                               delta_x)

        #np.savetxt(hist_writer, x_config[0:(n_lattice - 1)])
        utility_custom.correlation_measurments(n_lattice, n_meas, n_points,
                                               x_config, x_cor_sums, x2_cor_sums)

    # Evaluate averages and other stuff, maybe we can create a function

    utility_custom.\
        output_correlation_functions_and_log(n_points,
                                             x_cor_sums,
                                             x2_cor_sums,
                                             (n_mc_sweeps-n_equil) * n_meas,
                                             output_path)

    return 1
