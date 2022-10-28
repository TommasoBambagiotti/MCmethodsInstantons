"""

"""
import time
import random as rnd
import numpy as np
import cProfile as cP

import utility_monte_carlo as mc
import utility_custom
import input_parameters as ip


def free_energy_harmonic_osc(beta):
    '''
    Return the Free Helmoltz energy for an harmonic oscillator
    '''
    return -np.log(2.0 * np.sinh(beta * ip.w_omega0 / 2.0)) / beta


def monte_carlo_ao_switching(n_lattice,  # size of the grid
                             n_equil,  # equilibration sweeps
                             n_mc_sweeps,  # monte carlo sweeps
                             n_switching,  #
                             i_cold):  # cold/hot start
    '''
    Find the free energy of an anharmonic oscillator through
    Monte Carlo technique on an Euclidian Axis and the adiabatic
    switching
    '''
    if n_mc_sweeps < n_equil:
        print("too few Monte Carlo sweeps/ N_equilib > N_Monte_Carlo")

    # Variables for switching algorithm
    d_alpha = 1.0 / n_switching

    delta_s_alpha = np.zeros((2 * n_switching + 1))
    delta_s_alpha2 = np.zeros((2 * n_switching + 1))
    print(f'Adiabatic switching for beta = {n_lattice * ip.dtau}')
    # Now the principal cycle is over the coupling constant alpha
    for i_switching in range(2 * n_switching + 1):

        x_config = mc.initialize_lattice(n_lattice, i_cold)

        if i_switching <= n_switching:
            a_alpha = i_switching * d_alpha
        else:
            a_alpha = 2.0 - (i_switching) * d_alpha

        print(f'Switching #{i_switching}')

        for i_equil in range(n_equil):
            mc.metropolis_question(x_config, a_alpha)

        for i_mc in range(n_equil, n_mc_sweeps):

            delta_s_alpha_temp = 0.0
            mc.metropolis_question(x_config, a_alpha)
            for j in range(n_lattice):
                potential_0 = pow(ip.w_omega0 * x_config[j], 2) / 4.0
                potential_1 = pow(x_config[j] * x_config[j]
                                  - (ip.x_potential_minimum
                                     * ip.x_potential_minimum)
                                  , 2)
                delta_s_alpha_temp += (potential_1 - potential_0) * ip.dtau

            delta_s_alpha[i_switching] += delta_s_alpha_temp
            delta_s_alpha2[i_switching] += pow(delta_s_alpha_temp, 2)

        # Monte Carlo End
        # Control Acceptance ratio

    delta_s_alpha_av, delta_s_alpha_err = \
        mc.stat_av_var(delta_s_alpha, delta_s_alpha2,
                       n_mc_sweeps - n_equil)

    # Integration over alpha in <deltaS>
    integral_01 = np.trapz(delta_s_alpha_av[0:(n_switching + 1)], dx=d_alpha)
    integral_10 = np.trapz(
        delta_s_alpha_av[n_switching:(2 * n_switching + 1)], dx=d_alpha)

    integral_01_err = np.trapz(delta_s_alpha_err[0:(n_switching + 1)], dx=d_alpha)
    integral_10_err = np.trapz(
        delta_s_alpha_err[n_switching:(2 * n_switching + 1)], dx=d_alpha)

    trapezoidal_error = np.abs(
        (delta_s_alpha_av[n_switching] - delta_s_alpha_av[n_switching - 1]
         + delta_s_alpha_av[1] - delta_s_alpha_av[0]) /
        (d_alpha * 12 * n_switching * n_switching))

    propagation_error = np.sqrt(integral_01_err + integral_10_err)
    hysteresis_error = np.abs(integral_01 - integral_10)

    return -(integral_01 + integral_10) / 2.0, np.sqrt(pow(propagation_error, 2) +
                                                       pow(hysteresis_error, 2) +
                                                       pow(trapezoidal_error, 2))


def monte_carlo_virial_theorem(n_lattice,  # size of the grid
                               n_equil,  # equilibration sweeps
                               n_mc_sweeps,  # monte carlo sweeps
                               i_cold):
    print('Monte Carlo virial theorem approach:')
    # x position along the tau axis
    x_config = mc.initialize_lattice(n_lattice, i_cold)

    virial_hamiltonian = 0.0
    virial_hamiltonian2 = 0.0

    for i_equil in range(n_equil):
        x_config = mc.metropolis_question(x_config)

    # Rest of the MC sweeps

    for i_mc in range(n_mc_sweeps - n_equil):
        x_config = mc.metropolis_question(x_config)
        pot = 0.0
        kin = 0.0
        for i_pos in range(1, n_lattice):
            pot = pow(pow(x_config[i_pos], 2)
                      - pow(ip.x_potential_minimum, 2)
                      , 2)
            kin = 4.0 * pow(x_config[i_pos], 2) \
                  * (pow(x_config[i_pos], 2)
                     - pow(ip.x_potential_minimum, 2)
                     ) / 2.0
            virial_hamiltonian += (pot + kin)
            virial_hamiltonian2 += pow(pot + kin, 2)
            pot = 0.0
            kin = 0.0

    return virial_hamiltonian, virial_hamiltonian2


def free_energy_anharm(n_beta,
                       beta_max,
                       n_equil,
                       n_mc_sweeps,
                       n_switching,
                       i_cold):
    # Control output filepath
    output_path = './output_data/output_monte_carlo_switching'
    utility_custom.output_control(output_path)

    rnd.seed(time.time())

    # Free Helmoltz energy for the anharmonic oscillator
    beta_array = np.linspace(1.0, beta_max, n_beta, False)
    temperature_array = 1.0 / beta_array

    free_energy = np.empty((n_beta), float)
    free_energy_err = np.empty((n_beta), float)

    for i_beta in range(n_beta):
        n_lattice = int(beta_array[i_beta] / ip.dtau)

        free_energy[i_beta], free_energy_err[i_beta] = \
            monte_carlo_ao_switching(n_lattice,  # n_lattice
                                     n_equil,  # n_equil
                                     n_mc_sweeps,  # n_mc_sweeps-
                                     n_switching,
                                     i_cold)
        free_energy[i_beta] /= beta_array[i_beta]
        free_energy_err[i_beta] /= beta_array[i_beta]
        free_energy[i_beta] += free_energy_harmonic_osc(beta_array[i_beta])

    n_virial = 400
    v_ham, v_ham2 = monte_carlo_virial_theorem(n_virial,
                                               n_equil,
                                               n_mc_sweeps,
                                               i_cold)

    virial_hamiltonian_av, virial_hamiltonian_err = \
        mc.stat_av_var(v_ham, v_ham2, n_virial * (n_mc_sweeps - n_equil))

    with open(output_path + '/free_energy_mc.txt', 'w') as f_writer:
        np.savetxt(f_writer, free_energy)
        f_writer.write(str(-virial_hamiltonian_av))
    with open(output_path + '/free_energy_mc_err.txt', 'w') as f_writer:
        np.savetxt(f_writer, free_energy_err)
        f_writer.write(str(virial_hamiltonian_err))
    with open(output_path + '/temperature.txt', 'w') as f_writer:
        np.savetxt(f_writer, temperature_array)
        f_writer.write(str(1 / (n_virial * ip.dtau)))

# cP.run("free_energy_anharm(4, 20.0, 100, 50000, 20, False)")