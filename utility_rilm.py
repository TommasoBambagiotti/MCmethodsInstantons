import numpy as np
import random as rnd

import input_parameters as ip


def centers_setup(tau_array, n_ia, n_lattice):
    tau_centers_ia = np.random.uniform(0, n_lattice * ip.dtau, size=n_ia)
    tau_centers_ia = np.sort(tau_centers_ia)

    return tau_centers_ia


def centers_setup_gauss(tau_array, n_ia, n_lattice):
    tau_centers_ia_index = np.random.randint(0, n_lattice, size=n_ia)
    tau_centers_ia_index = np.sort(tau_centers_ia_index)

    tau_centers_ia = np.zeros((n_ia))
    for i_tau in range(n_ia):
        tau_centers_ia[i_tau] = \
            tau_array[tau_centers_ia_index[i_tau]]

    return tau_centers_ia, tau_centers_ia_index


def ansatz_instanton_conf(tau_centers_ia, tau_array):

    # if tau_centers_ia.size == 0:
    #     x_ansatz = np.repeat(-ip.x_potential_minimum, tau_array.size + 1)
    #     return x_ansatz

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


def hard_core_action(n_lattice,
                     tau_centers_ia,
                     tau_core,
                     action_core):
    
    action = 0.0
    
    for i_ia in range(0, tau_centers_ia.size, 2):
        if i_ia == 0:
            zero_crossing_m = tau_centers_ia[-1] - n_lattice *ip.dtau
        else:
            zero_crossing_m = tau_centers_ia[i_ia-1]
        
        action += action_core * np.exp(-(tau_centers_ia[i_ia]
                                         - zero_crossing_m)
                                       / tau_core)
    
    return action


def configuration_heating(x_delta_config, tau_array, tau_centers_ia,
                          tau_centers_ia_index):

    n_lattice = tau_array.size

    for i in range(1, n_lattice):

        if (i in tau_centers_ia_index) is False:
            action_loc_old = (
                pow((x_delta_config[i] -
                    x_delta_config[i - 1]) / (2 * ip.dtau), 2)
                + pow((x_delta_config[i + 1] -
                      x_delta_config[i]) / (2 * ip.dtau), 2)
                + gaussian_potential(x_delta_config[i], tau_array[i],
                                     tau_centers_ia)

            ) * ip.dtau

            if (i+1) in tau_centers_ia_index or (i-1) in tau_centers_ia_index:
                der = (x_delta_config[i+1] -
                       x_delta_config[i-1]) / (2.0 * ip.dtau)
                if der > -0.001 and der < 0.001:
                    der = 1.0
                action_loc_old += -np.log(np.abs(der))

            x_new = x_delta_config[i] + rnd.gauss(0, ip.delta_x)

            action_loc_new = (
                pow((x_new - x_delta_config[i - 1]) / (2 * ip.dtau), 2)
                + pow((x_delta_config[i + 1] - x_new) / (2 * ip.dtau), 2)
                + gaussian_potential(x_new, tau_array[i],
                                     tau_centers_ia)
            ) * ip.dtau

            if (i-1) in tau_centers_ia_index:
                der = (x_delta_config[i + 1] - x_new) / (2.0 * ip.dtau)
                if der > -0.001 and der < 0.001:
                    der = 1.0

                action_loc_new += - np.log(np.abs(der))

            elif (i+1) in tau_centers_ia_index:
                der = (x_new - x_delta_config[i - 1]) / (2.0 * ip.dtau)
                if der > -0.001 and der < 0.001:
                    der = 1.0
                action_loc_new += - np.log(np.abs(der))

            delta_action = action_loc_new - action_loc_old

            if np.exp(-delta_action) > rnd.uniform(0., 1.):
                x_delta_config[i] = x_new

    x_delta_config[n_lattice - 1] = x_delta_config[0]
    x_delta_config[n_lattice] = x_delta_config[1]


def rilm_monte_carlo_step(n_ia,  # number of instantons and anti inst.
                          n_points,  #
                          n_meas,
                          tau_array,
                          x_cor_sums,
                          x2_cor_sums):

    # Center of instantons and anti instantons
    tau_centers_ia= centers_setup(tau_array, n_ia, tau_array.size)

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

    # We return the x_conf for the tau_zcr distribution
    return tau_centers_ia


def rilm_heated_monte_carlo_step(n_ia,  # number of instantons and anti inst.
                                 n_heating,
                                 n_points,  #
                                 n_meas,
                                 tau_array,
                                 x_cor_sums,
                                 x2_cor_sums):

    # Center of instantons and anti instantons
    tau_centers_ia, tau_centers_ia_index = centers_setup_gauss(
        tau_array, n_ia, tau_array.size)
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
