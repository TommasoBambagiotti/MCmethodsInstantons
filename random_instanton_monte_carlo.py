'''

'''
import random as rnd
import numpy as np
import scipy.special as sp
from scipy.stats import rv_discrete as dis

import utility_custom
import utility_monte_carlo as mc

import input_parameters as ip


def centers_setup(n_ia, beta_max):
    tau_centers_ia = np.random.uniform(0.0, beta_max, (n_ia))
    return tau_centers_ia


def ansatz_instanton_conf(tau_centers_ia, tau_array):

    if tau_centers_ia.size == 0:
        x_ansatz = np.repeat(-ip.x_potential_minimum, tau_array.size + 1)
        return x_ansatz

    x_ansatz = np.zeros((tau_array.size), float)

    tau_centers_ia_sorted = np.sort(tau_centers_ia)
    top_charge = 1
    for tau_ia in np.nditer(tau_centers_ia_sorted):
        x_ansatz += top_charge * ip.x_potential_minimum \
            * np.tanh(2 * ip.x_potential_minimum
                      * (tau_array - tau_ia))

        top_charge *= -1

    x_ansatz -= ip.x_potential_minimum

    # Border periodic conditions
    x_ansatz[0] = x_ansatz[-1]
    x_ansatz = np.append(x_ansatz, x_ansatz[1])

    return x_ansatz


def rilm_monte_carlo_step(n_ia,  # number of instantons and anti inst.
                          n_points,  #
                          n_meas,
                          tau_array,
                          x_cor_sums,
                          x2_cor_sums):

    # Center of instantons and anti instantons
    tau_centers_ia = centers_setup(n_ia, tau_array[-1])
    # Ansatz sum of indipendent instantons
    x_ansatz = ansatz_instanton_conf(tau_centers_ia, tau_array)

    for _ in range(n_meas):
        i_p0 = int((tau_array.size - n_points) * rnd.uniform(0., 1.))
        x_0 = x_ansatz[i_p0]
        for i_point in range(n_points):
            x_1 = x_ansatz[i_p0 + i_point]

            x_cor_sums[0, i_point] += x_0 * x_1
            x_cor_sums[1, i_point] += pow(x_0 * x_1, 2)
            x_cor_sums[2, i_point] += pow(x_0 * x_1, 3)

            x2_cor_sums[0, i_point] += pow(x_0 * x_1, 2)
            x2_cor_sums[1, i_point] += pow(x_0 * x_1, 4)
            x2_cor_sums[2, i_point] += pow(x_0 * x_1, 6)

# Classic distribution of instantons
def rho_n_ia(n_ia, tau_max):
    norm = 10

    x = np.power(tau_max/norm, n_ia)
    f = sp.factorial(n_ia/2)
    f = np.multiply(f, f)

    exponential = np.exp(-4/3 * pow(ip.x_potential_minimum, 3)
                         * n_ia)
    x = np.divide(x, f)
    x = np.multiply(x, exponential)

    partition_function = np.sum(x, axis=0)

    return x/partition_function


def random_instanton_liquid_model(n_lattice,  # size of the grid
                                  n_mc_sweeps,  # monte carlo sweeps
                                  n_points,  #
                                  n_meas,
                                  loop_flag):

    # Control output filepath
    output_path = './output_data/output_rilm'
    utility_custom.output_control(output_path)

    # Eucliadian time
    tau_array = np.linspace(0.0, n_lattice * ip.dtau, n_lattice, False)

    # Correlation functions
    x_cor_sums = np.zeros((3, n_points))
    x2_cor_sums = np.zeros((3, n_points))

    # Classic distribution (WRONG)
    if loop_flag is False:

        # Discrete distribution
        n_ia_array = [0, 2, 4, 6, 8]
        prob = rho_n_ia(np.array(n_ia_array), n_lattice * ip.dtau)
        n_ia_distribution = \
            dis(name='number_inst_distribution', values=(n_ia_array, prob))

        for i_mc in range(n_mc_sweeps):
            n_ia = n_ia_distribution.rvs(size=1)

            rilm_monte_carlo_step(n_ia,
                                  n_points,
                                  n_meas,
                                  tau_array,
                                  x_cor_sums,
                                  x2_cor_sums)

    # n_ia evaluated from 2-loop semi-classical expansion
    else:
        s0 = 4 / 3 * pow(ip.x_potential_minimum, 3)
        loop_2 = 8 * pow(ip.x_potential_minimum, 5 / 2) \
            * pow(2 / np.pi, 1/2) * np.exp(-s0 - 71 / (72 * s0))
        n_ia = int(np.rint(loop_2 * n_lattice * ip.dtau))
        print(n_ia)
        for i_mc in range(n_mc_sweeps):
            rilm_monte_carlo_step(n_ia,
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

    with open(output_path + '/tau_array.txt', 'w') as tau_writer:
        np.savetxt(tau_writer,
                   np.linspace(0, n_points * ip.dtau, n_points, False))

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
