'''
'''
import random as rnd
import numpy as np
import cProfile as cP

import utility_custom
import utility_monte_carlo as mc
import input_parameters as ip



def centers_setup(tau_array, n_ia, n_lattice):
    tau_centers_ia_index = np.random.randint(0, n_lattice, size=n_ia)
    tau_centers_ia_index = np.sort(tau_centers_ia_index)

    tau_centers_ia = np.zeros((n_ia))
    for i_tau in range(n_ia):
        tau_centers_ia[i_tau] = \
            tau_array[tau_centers_ia_index[i_tau]]

    return tau_centers_ia, tau_centers_ia_index


def ansatz_instanton_conf(tau_centers_ia, tau_array):

    if tau_centers_ia.size == 0:
        x_ansatz = np.repeat(-ip.x_potential_minimum, tau_array.size + 1)
        return x_ansatz

    x_ansatz = np.zeros((tau_array.size), float)

    top_charge = 1
    for tau_ia in np.nditer(tau_centers_ia):
        x_ansatz += top_charge * ip.x_potential_minimum \
            * np.tanh(2 * ip.x_potential_minimum
                      * (tau_array - tau_ia))

        top_charge *= -1

    x_ansatz -= ip.x_potential_minimum

    # Border periodic conditions
    x_ansatz[0] = x_ansatz[-1]
    x_ansatz = np.append(x_ansatz, x_ansatz[1])

    return x_ansatz


def gaussian_potential(x_pos, tau, tau_ia_centers):
    potential = 0.0
    for tau_ia in np.nditer(tau_ia_centers):
        potential += -3.0 / (2.0 * pow(np.cosh(2 * ip.x_potential_minimum
                                           * (tau - tau_ia)), 2))
    potential += 1
    potential *= 4.0 * ip.x_potential_minimum \
                * ip.x_potential_minimum * x_pos * x_pos
    
    return potential


def configuration_heating(x_config, tau_array, tau_centers_ia,
                          tau_centers_ia_index):
    
    n_lattice = tau_array.size
    
    for i in range(1, n_lattice):
        
        if (i in tau_centers_ia_index) is False:
            action_loc_old = (
                    pow((x_config[i] - x_config[i - 1]) / (2 * ip.dtau), 2) 
                    + pow((x_config[i + 1] - x_config[i]) / (2 * ip.dtau), 2) 
                    + gaussian_potential(x_config[i], tau_array[i],
                                         tau_centers_ia)
                    
                ) * ip.dtau
        
            if (i+1) in tau_centers_ia_index or (i-1) in tau_centers_ia_index:
                der = (x_config[i+1] - x_config[i-1]) / (2.0 * ip.dtau)
                if der > -0.001 and der < 0.001:
                    der = 1.0
                action_loc_old += -np.log(np.abs(der))
    
            x_new = x_config[i] + rnd.gauss(0, ip.delta_x)
    
            action_loc_new = (
                   pow((x_new - x_config[i - 1]) / (2 * ip.dtau), 2)
                   + pow((x_config[i + 1] - x_new) / (2 * ip.dtau), 2) 
                   + gaussian_potential(x_new, tau_array[i],
                                        tau_centers_ia)
            ) * ip.dtau
    
            if (i-1) in tau_centers_ia_index:
                der = (x_config[i + 1] - x_new) / (2.0 * ip.dtau)
                if der > -0.001 and der < 0.001:
                    der = 1.0
                
                action_loc_new += - np.log(np.abs(der))
                
            elif (i+1) in tau_centers_ia_index:
                der = (x_new - x_config[i - 1]) / (2.0 * ip.dtau)
                if der > -0.001 and der < 0.001:
                    der = 1.0
            
                action_loc_new += - np.log(np.abs(der))
                
            delta_action = action_loc_new - action_loc_old
            
            if np.exp(-delta_action) > rnd.uniform(0., 1.):
                x_config[i] = x_new
            
    x_config[n_lattice - 1] = x_config[0]
    x_config[n_lattice] = x_config[1]
    
def rilm_monte_carlo_step(n_ia, # number of instantons and anti inst.
                          n_heating,
                          n_points,  #
                          n_meas,
                          tau_array,
                          x_cor_sums,
                          x2_cor_sums):

    # Center of instantons and anti instantons
    tau_centers_ia, tau_centers_ia_index = centers_setup(tau_array, n_ia, tau_array.size)
    # Ansatz sum of indipendent instantons
    x_ansatz = ansatz_instanton_conf(tau_centers_ia, tau_array)
    # Difference from the classical solution
    x_delta_config = np.zeros((tau_array.size + 1))
    # Heating sweeps
    for _ in range(n_heating):
        configuration_heating(x_delta_config, 
                              tau_array,
                              tau_centers_ia,
                              tau_centers_ia_index)
        
        x_ansatz_heated = x_ansatz + x_delta_config

    
    np.savetxt('./output_data/output_rilm_heating/configuration.txt', x_ansatz)
    np.savetxt('./output_data/output_rilm_heating/configuration_heated.txt', x_ansatz_heated)

    for _ in range(n_meas):
        i_p0 = int((tau_array.size - n_points) * rnd.uniform(0., 1.))
        x_0 = x_ansatz_heated[i_p0]
        for i_point in range(n_points):
            x_1 = x_ansatz_heated[i_p0 + i_point]

            x_cor_sums[0, i_point] += x_0 * x_1
            x_cor_sums[1, i_point] += pow(x_0 * x_1, 2)
            x_cor_sums[2, i_point] += pow(x_0 * x_1, 3)

            x2_cor_sums[0, i_point] += pow(x_0 * x_1, 2)
            x2_cor_sums[1, i_point] += pow(x_0 * x_1, 4)
            x2_cor_sums[2, i_point] += pow(x_0 * x_1, 6)
            
            
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
        
        rilm_monte_carlo_step(n_ia,
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
