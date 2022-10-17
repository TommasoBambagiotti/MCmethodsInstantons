"""

"""
import time
import random as rnd
import numpy as np

<<<<<<< Updated upstream
import monte_carlo_ao
=======
import utility_monte_carlo as mc
>>>>>>> Stashed changes
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

    delta_s_alpha = np.zeros((2*n_switching + 1))
    delta_s_alpha2 = np.zeros((2*n_switching + 1))


    # Now the principal cycle is over the coupling constant alpha
    for i_switching in range(2 * n_switching + 1):

<<<<<<< Updated upstream
        x_config = monte_carlo_ao.initialize_lattice(n_lattice,
                                                     i_cold)
=======
        x_config = mc.initialize_lattice(n_lattice,i_cold)
>>>>>>> Stashed changes

        if i_switching <= n_switching:
            a_alpha = i_switching * d_alpha
        else:
            a_alpha = 2.0 - (i_switching) * d_alpha


        for i_equil in range(n_equil):
<<<<<<< Updated upstream
            x_config = monte_carlo_ao.metropolis_question(x_config,
                                                          a_alpha)
        for i_mc in range(n_mc_sweeps - n_equil):

            delta_s_alpha_temp = 0.0
            x_config = monte_carlo_ao.metropolis_question(x_config,
                                                      a_alpha)
=======
            x_config = mc.metropolis_question(x_config, a_alpha)
        for i_mc in range(n_mc_sweeps - n_equil):

            delta_s_alpha_temp = 0.0
            x_config = mc.metropolis_question(x_config, a_alpha)
>>>>>>> Stashed changes
            for j in range(n_lattice):
                potential_0 = pow(ip.w_omega0 * x_config[j], 2) / 4.0
                potential_1 = pow((pow(x_config[j], 2) - pow(ip.x_potential_minimum, 2)), 2)
                delta_s_alpha_temp += (potential_1 - potential_0) * ip.dtau


            delta_s_alpha[i_switching] += delta_s_alpha_temp
            delta_s_alpha2[i_switching] += pow(delta_s_alpha_temp, 2)

        # Monte Carlo End
        # Control Acceptance ratio

    delta_s_alpha_av, delta_s_alpha_err = \
<<<<<<< Updated upstream
        monte_carlo_ao.stat_av_var(delta_s_alpha,
                               delta_s_alpha2,
                               n_mc_sweeps - n_equil)
=======
        mc.stat_av_var(delta_s_alpha,delta_s_alpha2,
                       n_mc_sweeps - n_equil)
>>>>>>> Stashed changes

    # Integration over alpha in <deltaS>
    integral_01 = np.trapz(delta_s_alpha_av[0:(n_switching + 1)], dx = d_alpha)
    integral_10 = np.trapz(delta_s_alpha_av[n_switching:(2*n_switching + 1)], dx = d_alpha)

    integral_01_err = np.trapz(delta_s_alpha_err[0:(n_switching + 1)], dx = d_alpha)
    integral_10_err = np.trapz(delta_s_alpha_err[n_switching:(2*n_switching + 1)], dx = d_alpha)

    trapezoidal_error = np.abs(
        (delta_s_alpha_av[n_switching] - delta_s_alpha_av[n_switching-1] +
          delta_s_alpha_av[1] - delta_s_alpha_av[0]) /
        (d_alpha * 12 * n_switching * n_switching ) )

    propagation_error = np.sqrt(integral_01_err + integral_10_err)
    hysteresis_error = np.abs(integral_01 - integral_10)

    return -(integral_01 + integral_10) / 2.0, np.sqrt(pow(propagation_error, 2) +
                                                      pow(hysteresis_error, 2) +
                                                      pow(trapezoidal_error, 2))

def free_energy_anharm(n_beta,
                       n_equil,
                       n_mc_sweeps,
                       n_switching):

    # Control output filepath
<<<<<<< Updated upstream
    output_path = r'.\output_data\output_monte_carlo_switching'
=======
    output_path = './output_data/output_monte_carlo_switching'
>>>>>>> Stashed changes
    utility_custom.output_control(output_path)

    rnd.seed(time.time())

    # Free Helmoltz energy for the anharmonic oscillator
<<<<<<< Updated upstream
    beta_array = np.linspace(1.0, 8.0, n_beta, False)
    temperature_array = 1.0 / beta_array

    dtau = 0.05
=======
    beta_array = np.linspace(1.0, 2.0, n_beta, False)
    temperature_array = 1.0 / beta_array
>>>>>>> Stashed changes

    free_energy = np.empty((n_beta), float)
    free_energy_err = np.empty((n_beta), float)

    for i_beta in range(n_beta):

<<<<<<< Updated upstream
        n_lattice = int(beta_array[i_beta]/dtau)
=======
        n_lattice = int(beta_array[i_beta]/ip.dtau)
>>>>>>> Stashed changes

        free_energy[i_beta], free_energy_err[i_beta] = \
        monte_carlo_ao_switching(n_lattice,  # n_lattice
                                 n_equil,  # n_equil
                                 n_mc_sweeps,  # n_mc_sweeps-
                                 n_switching,
                                 False)
        free_energy[i_beta] /= beta_array[i_beta]
        free_energy_err[i_beta] /= beta_array[i_beta]
        free_energy[i_beta] += free_energy_harmonic_osc(beta_array[i_beta])

<<<<<<< Updated upstream
    with open(output_path + r'\free_energy_mc.txt', 'w') as f_writer:
        np.savetxt(f_writer, free_energy)
    with open(output_path + r'\free_energy_mc_err.txt', 'w') as f_writer:
        np.savetxt(f_writer, free_energy_err)
    with open(output_path + r'\temperature.txt', 'w') as f_writer:
=======
    with open(output_path + '/free_energy_mc.txt', 'w') as f_writer:
        np.savetxt(f_writer, free_energy)
    with open(output_path + '/free_energy_mc_err.txt', 'w') as f_writer:
        np.savetxt(f_writer, free_energy_err)
    with open(output_path + '/temperature.txt', 'w') as f_writer:
>>>>>>> Stashed changes
        np.savetxt(f_writer, temperature_array)

    print(free_energy)

if __name__ == '__main__':

<<<<<<< Updated upstream
    free_energy_anharm(4,
                       100,
                       50000,
=======
    free_energy_anharm(2,
                       100,
                       500,
>>>>>>> Stashed changes
                       20)
