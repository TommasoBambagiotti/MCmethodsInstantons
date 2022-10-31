'''
'''
import random as rnd
import numpy as np

import utility_custom
import utility_monte_carlo as mc
import utility_rilm as rilm
import input_parameters as ip

                        
def random_instanton_liquid_model_heating(n_lattice,  # size of the grid
                                          n_mc_sweeps,  # monte carlo sweeps
                                          n_points,  #
                                          n_meas,
                                          n_heating):

    # Control output filepath
    output_path = './output_data/output_rilm_heating'
    utility_custom.output_control(output_path)

    # Eucliadian time
    tau_array = np.linspace(0.0, n_lattice * ip.dtau, n_lattice, False)

    # Correlation functions
    x_cor_sums = np.zeros((3, n_points))
    x2_cor_sums = np.zeros((3, n_points))


    # n_ia evaluated from 2-loop semi-classical expansion
    s0 = 4 / 3 * pow(ip.x_potential_minimum, 3)
    loop_2 = 8 * pow(ip.x_potential_minimum, 5 / 2) \
        * pow(2 / np.pi, 1/2) * np.exp(-s0 - 71 / (72 * s0))
    n_ia = int(np.rint(loop_2 * n_lattice * ip.dtau))
    print(n_ia)
    for i_mc in range(n_mc_sweeps):
        print(f'#{i_mc} sweep in {n_mc_sweeps-1}')
        
        rilm.rilm_heated_monte_carlo_step(n_ia,
                                          n_heating,
                                          n_points,
                                          n_meas,
                                          tau_array,
                                          x_cor_sums,
                                          x2_cor_sums)
        

    x_cor_av = np.zeros((3, n_points))
    x_cor_err = np.zeros((3, n_points))

    for i_stat in range(3):
        x_cor_av[i_stat], x_cor_err[i_stat] = \
            mc.stat_av_var(x_cor_sums[i_stat],
                           x2_cor_sums[i_stat],
                           n_meas * (n_mc_sweeps))

        with open(output_path + f'/average_x_cor_{i_stat + 1}.txt', 'w') as av_writer:
            np.savetxt(av_writer, x_cor_av[i_stat])
        with open(output_path + f'/error_x_cor_{i_stat + 1}.txt', 'w') as err_writer:
            np.savetxt(err_writer, x_cor_err[i_stat])

    with open(output_path + '/tau_array_conf.txt', 'w') as tau_writer:
        np.savetxt(tau_writer,
                   np.linspace(0, n_lattice * ip.dtau, n_lattice, False))

    derivative_log_corr_funct = np.zeros((3, n_points - 1))
    derivative_log_corr_funct_err = np.zeros((3, n_points - 1))

    for i_stat in range(3):

        if i_stat != 1:

            derivative_log_corr_funct[i_stat], derivative_log_corr_funct_err[i_stat] = \
                mc.log_central_der_alg(
                    x_cor_av[i_stat], x_cor_err[i_stat], ip.dtau)

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
        with open(output_path + f'/average_der_log_{i_stat + 1}.txt', 'w',
                  encoding='utf8') as av_writer:
            np.savetxt(av_writer, derivative_log_corr_funct[i_stat])

        with open(output_path + f'/error_der_log_{i_stat + 1}.txt', 'w',
                  encoding='utf8') as err_writer:
            np.savetxt(err_writer, derivative_log_corr_funct_err[i_stat])
                    
    # time array
    with open(output_path + '/tau_array.txt', 'w',
              encoding='utf8') as tau_writer:

        np.savetxt(tau_writer, np.linspace(
            0, n_points * ip.dtau, n_points, False))
