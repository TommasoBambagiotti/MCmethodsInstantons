import time
import numpy as np
import utility_monte_carlo as mc
import utility_custom


def instantons_density_switching(n_lattice,
                                 n_equil,
                                 n_mc_sweeps,
                                 n_switching,
                                 n_minima,
                                 first_minimum=1.4,
                                 dtau=0.05,
                                 delta_x=0.5):

    output_path = './output_data/output_monte_carlo_density_switching'
    utility_custom.output_control(output_path)

    # clean folder

    utility_custom.clear_folder(output_path)

    if n_mc_sweeps < n_equil:
        print("too few Monte Carlo sweeps/ N_equilib > N_Monte_Carlo")

    # Densities

    # gauss_density = np.zeros(3)
    # gauss_density_err = np.zeros(3)
    #
    # non_gauss_density = np.zeros(3)
    # non_gauss_density_error = np.zeros(3)

    potential_minima = np.zeros(n_minima, float)
    total_density = np.zeros(n_minima, float)
    total_density_error = np.zeros(n_minima, float)
    # Beta
    beta = n_lattice * dtau

    for i_minimum in range(n_minima):
        print(f'New monte carlo for eta = {first_minimum+ 0.1 * i_minimum}')

        x_potential_minimum = first_minimum + 0.1 * i_minimum

        potential_minima[i_minimum] = x_potential_minimum

        density = np.zeros(2)
        density_error = np.zeros(2)
        gauss_density = 8 * pow(x_potential_minimum, 5/2)\
            * np.sqrt(2/np.pi)*np.exp(-4*pow(x_potential_minimum, 3)/3)

        # Variables for switching algorithm
        d_alpha = 1.0 / n_switching

        delta_s_alpha = np.zeros((2, 2 * n_switching + 1))
        delta_s2_alpha = np.zeros((2, 2 * n_switching + 1))

        # Averages
        delta_s_alpha_av = np.zeros((2, 2 * n_switching + 1))
        delta_s_alpha_err = np.zeros((2, 2 * n_switching + 1))
        delta_s_av = np.zeros(2)

        # Errors

        trapezoidal_error = np.zeros(2)
        propagation_error = np.zeros(2)
        hysteresis_error = np.zeros(2)

        # Integration

        integral_01 = np.zeros(2)
        integral_10 = np.zeros(2)
        integral_01_err = np.zeros(2)
        integral_10_err = np.zeros(2)

        # Configuration initialization

        # classical configuration
        x_config_one_sector = mc.initialize_lattice(n_lattice, x_potential_minimum,
                                                    False, True)

        # we expand the action about this configuration (central point in a taylor polyn.)
        x_config_0_one = mc.initialize_lattice(n_lattice, x_potential_minimum,
                                               False, True)

        second_der_0_one = mc.second_derivative_action(x_config_0_one[1:-1],
                                                       x_potential_minimum)
        # vacuum configuration
        x_config_zero_sector = np.full(n_lattice+1, -x_potential_minimum)

        # we expand the action about this configuration (central point in a taylor polyn.)
        x_config_0_zero = np.full(n_lattice+1, -x_potential_minimum)

        second_der_0_zero = mc.second_derivative_action(x_config_0_zero[1:-1],
                                                        x_potential_minimum)

        x_config_0_sector = np.array([x_config_0_zero,
                                      x_config_0_one])

        x_config_sector = np.array([x_config_zero_sector,
                                    x_config_one_sector])

        second_der_0 = np.array([second_der_0_zero,
                                 second_der_0_one])

        print(f'Adiabatic switching for beta = {n_lattice * dtau}')

        for i_sector in [0, 1]:
            print(i_sector)
            for i_switching in range(2*n_switching+1):

                start = time.time()
                # set alpha variable for integration
                if i_switching <= n_switching:

                    a_alpha = i_switching * d_alpha

                else:

                    a_alpha = 2.0 - i_switching * d_alpha

                print(f'Switching #{i_switching} of #{2*n_switching}')

                # equilibrium sweeps
                for _ in range(n_equil):

                    # metropolis i_sector instanton sector
                    mc.metropolis_question_density_switching(
                        x_config_sector[i_sector],
                        x_config_0_sector[i_sector],
                        second_der_0[i_sector],
                        mc.gaussian_potential,
                        x_potential_minimum,
                        dtau,
                        delta_x,
                        i_sector,  # i_sector instanton sector
                        a_alpha)

                for i_mc in range(n_equil, n_mc_sweeps):

                    # temp variables
                    delta_s_alpha_temp = 0.0
                    # metropolis i_sector instanton sector
                    mc.metropolis_question_density_switching(
                        x_config_sector[i_sector],
                        x_config_0_sector[i_sector],
                        second_der_0[i_sector],
                        mc.gaussian_potential,
                        x_potential_minimum,
                        dtau,
                        delta_x,
                        i_sector,  # i_sector instanton sector
                        a_alpha)

                    potential_0 = 1.0 / 2.0 * second_der_0[i_sector]\
                        * np.square(x_config_sector[i_sector, 1:-1]
                                    - x_config_0_sector[i_sector, 1:-1])\
                        + mc.potential_anh_oscillator(x_config_0_sector[i_sector, 1:-1],
                                                      x_potential_minimum)

                    potential_1 = mc.potential_anh_oscillator(
                        x_config_sector[i_sector, 1:-1],
                        x_potential_minimum)

                    delta_s_alpha_temp += dtau * \
                        np.sum(potential_1-potential_0)

                    delta_s_alpha[i_sector,
                                  i_switching] += delta_s_alpha_temp#/beta
                    # error
                    delta_s2_alpha[i_sector,
                                   i_switching] += np.square(delta_s_alpha_temp)#/beta

                    if i_mc == n_mc_sweeps-1:
                        print(f'delta V alpha {i_sector} for {i_switching}:')
                        print(delta_s_alpha[i_sector,
                              i_switching]/(n_mc_sweeps-n_equil))

                end = time.time()
                print(f'Elapsed: {end - start}')
                # Monte-Carlo end
                # Compute averages and then integrate over alpha

                # 0: zero inst sector, 1: one inst. sector

            delta_s_alpha_av[i_sector], delta_s_alpha_err[i_sector] = \
                utility_custom.stat_av_var(delta_s_alpha[i_sector],
                                           delta_s2_alpha[i_sector],
                                           n_mc_sweeps - n_equil)

            integral_01[i_sector] = np.trapz(
                delta_s_alpha_av[i_sector, 0:(n_switching + 1)],
                dx=d_alpha)

            integral_10[i_sector] = np.trapz(
                delta_s_alpha_av[i_sector, n_switching:(2 * n_switching + 1)],
                dx=d_alpha)

            delta_s_av[i_sector] = (
                integral_01[i_sector] + integral_10[i_sector]) / 2

            integral_01_err[i_sector] = np.trapz(
                delta_s_alpha_err[i_sector, 0:(n_switching + 1)],
                dx=d_alpha)

            integral_10_err[i_sector] = np.trapz(
                delta_s_alpha_err[i_sector, n_switching:(2 * n_switching + 1)],
                dx=d_alpha)
            # Errors
            trapezoidal_error[i_sector] = np.abs(
                (delta_s_alpha_av[i_sector, n_switching] -
                 delta_s_alpha_av[i_sector, n_switching - 1] +
                 delta_s_alpha_av[i_sector, 1] - delta_s_alpha_av[i_sector, 0]) /
                (d_alpha * 12 * n_switching * n_switching))

            propagation_error[i_sector] = np.sqrt(
                integral_01_err[i_sector] + integral_10_err[i_sector])

            hysteresis_error[i_sector] = np.abs(
                integral_01[i_sector] - integral_10[i_sector])

            # Density computation

            density[i_sector] = np.exp(-delta_s_av[i_sector])

            density_error[i_sector] = density[i_sector] * np.sqrt(
                np.square(trapezoidal_error[i_sector]) +
                np.square(propagation_error[i_sector]) +
                np.square(hysteresis_error[i_sector])
            )

        # Total density and error
        total_density[i_minimum] = gauss_density *\
            np.exp(-delta_s_av[1]) * np.exp(delta_s_av[0])
        total_density_error[i_minimum] = total_density[i_minimum] *\
            np.sqrt(pow(density_error[0], 2) + pow(density_error[1], 2))

        print(f'density {total_density[i_minimum]} +/- {total_density_error[i_minimum]}')
        print('Switching end\n')

    with open(output_path + '/total_density.txt', 'w',
              encoding='utf8') as f_writer:
        np.savetxt(f_writer, total_density)
    with open(output_path + '/total_density_err.txt', 'w',
              encoding='utf8') as f_writer:
        np.savetxt(f_writer, total_density_error)

    with open(output_path + '/potential_minima.txt', 'w',
              encoding='utf8') as pot_writer:
        np.savetxt(pot_writer, potential_minima)
