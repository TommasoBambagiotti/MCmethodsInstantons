
import random as rnd
import numpy as np

import utility_custom
import utility_monte_carlo as mc

import input_parameters as ip


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
    for i_equil in range(n_equil):

        mc.metropolis_question(x_config)

        # Rest of the MC sweeps
    with open(output_path + '/ground_state_histogram.dat', 'wb') as hist_writer:
        for i_mc in range(n_mc_sweeps - n_equil):

            mc.metropolis_question(x_config)

            np.save(hist_writer, x_config[0:(n_lattice - 1)])
            for k_meas in range(n_meas):
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

    x_cor_av = np.zeros((3, n_points))
    x_cor_err = np.zeros((3, n_points))
    
    for i_stat in range(3):
        x_cor_av[i_stat] , x_cor_err[i_stat] = \
            mc.stat_av_var(x_cor_sums[i_stat],
                           x2_cor_sums[i_stat],
                           n_meas * (n_mc_sweeps - n_equil)
                           )
            
        
        with open(output_path + f'/average_x_cor_{i_stat + 1}.txt', 'w') as av_writer:
            np.savetxt(av_writer, x_cor_av[i_stat])
            
        with open(output_path + f'/error_x_cor_{i_stat + 1}.txt', 'w') as err_writer:
            np.savetxt(err_writer, x_cor_err[i_stat])


    derivative_log_corr_funct = np.zeros((3, n_points - 1))
    derivative_log_corr_funct_err = np.zeros((3, n_points - 1))

    for i_stat in range(3):

        if i_stat != 1:

            derivative_log_corr_funct[i_stat], derivative_log_corr_funct_err[i_stat] = \
                mc.log_central_der_alg(x_cor_av[i_stat], x_cor_err[i_stat], ip.dtau)

        else:
            # In the case of log <x^2x^2> the constant part <x^2>
            # is circa the average for the greatest tau

            # subtraction of the constant term (<x^2>)^2 in <x(0)^2x(t)^2>
            cor_funct_err = np.sqrt(np.square(x_cor_err[i_stat])
                                    + pow(x_cor_err[i_stat, n_points - 1], 2))

            derivative_log_corr_funct[i_stat], derivative_log_corr_funct_err[i_stat] = \
                mc.log_central_der_alg(
                    x_cor_av[i_stat] - x_cor_av[i_stat, n_points - 1],
                    cor_funct_err,
                    ip.dtau)
            
            # Save into files

        # w/o cooling
        with open(output_path + f'/average_der_log_{i_stat + 1}.txt', 'w') as av_writer:
            np.savetxt(av_writer, derivative_log_corr_funct[i_stat])

        with open(output_path + f'/error_der_log_{i_stat + 1}.txt', 'w') as err_writer:
            np.savetxt(err_writer, derivative_log_corr_funct_err[i_stat])


    # time array
    with open(output_path + '/tau_array.txt', 'w') as tau_writer:

        np.savetxt(tau_writer, np.linspace(0, n_points * ip.dtau, n_points, False))

    return 1



