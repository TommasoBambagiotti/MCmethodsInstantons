import random as rnd
import numpy as np
import utility_monte_carlo as mc
import utility_custom
import input_parameters as ip


def instantons_density_switching(n_lattice,
                                 n_equil,
                                 n_mc_sweeps,
                                 n_switching):

    output_path = './output_data/output_monte_carlo_density_switching'
    utility_custom.output_control(output_path)

    # clean folder

    utility_custom.clear_folder(output_path)

    if n_mc_sweeps < n_equil:
        print("too few Monte Carlo sweeps/ N_equilib > N_Monte_Carlo")

    # Densities

    gauss_density = np.zeros(3)
    gauss_density_err = np.zeros(3)
    non_gauss_density = np.zeros(3)
    non_gauss_density_error = np.zeros(3)

    density = np.zeros(2)
    density_error = np.zeros(2)
    #gauss_density = 8 * pow(ip.x_potential_minimum, 5/2)*np.sqrt(2/np.pi)*np.exp(-4*pow(ip.x_potential_minimum, 3)/3)

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
    x_config_one_sector = mc.initialize_lattice(n_lattice, False, True)

    # we expand the action about this configuration (central point in a taylor polyn.)
    x_config_0_one = mc.initialize_lattice(n_lattice, False, True)

    # vacuum configuration
    x_config_zero_sector = np.full(n_lattice+1, ip.x_potential_minimum)

    # we expand the action about this configuration (central point in a taylor polyn.)
    x_config_0_zero = np.full(n_lattice+1, ip.x_potential_minimum)

    # Plots

    tau = np.linspace(0, ip.dtau*n_lattice, n_lattice+1)
    with open(output_path + '/tau_array.txt', 'w') as f_writer:
        np.savetxt(f_writer, tau)

    n_rnd_config = 5
    rnd_configuration = np.zeros(n_rnd_config)

    # Random configuration: we extract #n_rnd_config random configurations, both
    # in the one and zero sector, to plot them

    for i_rnd in range(n_rnd_config):

        rnd_configuration[i_rnd] = rnd.randint(n_equil, n_mc_sweeps-n_equil)

    # random alpha
    rnd_i_switching = rnd.randint(0, n_switching)

    with open(output_path + '/configuration_number.txt', 'w') as f_writer:
        np.savetxt(f_writer, rnd_configuration, fmt='%3d')

    f_writer.close()

    print(f'Adiabatic switching for beta = {n_lattice * ip.dtau}')

    for i_switching in range(2*n_switching+1):

        # set alpha variable for integration
        if i_switching <= n_switching:

           a_alpha = i_switching * d_alpha

        else:

            a_alpha = 2.0 - i_switching * d_alpha

        print(f'Switching #{i_switching} of #{2*n_switching}')

        # equilibrium sweeps
        for i_equil in range(n_equil):

            # metropolis one instanton sector
            mc.metropolis_question_density_switching(
                                                    x_config_one_sector,
                                                    x_config_0_one,
                                                    mc.gaussian_potential,
                                                    1,  # one instanton sector
                                                    a_alpha)

            # metropolis zero instanton sector
            mc.metropolis_question_density_switching(
                                                    x_config_zero_sector,
                                                    x_config_0_zero,
                                                    mc.gaussian_potential,
                                                    0,  # zero instanton sector
                                                    a_alpha)

        for i_mc in range(n_equil, n_mc_sweeps):

            # temp variables
            delta_s_alpha_temp_one = 0.0
            delta_s_alpha_temp_zero = 0.0
            # Metropolis one instanton sector
            mc.metropolis_question_density_switching(
                                                    x_config_one_sector,
                                                    x_config_0_one,
                                                    mc.gaussian_potential,
                                                    1,  # one instanton sector
                                                    a_alpha)

            # Plots some random configuration for alpha=0
            if (i_mc in rnd_configuration) and (i_switching == rnd_i_switching):

                with open(output_path + f'/config_0alpha_1sec_{i_mc}.0.txt', 'w') as f_writer:
                    np.savetxt(f_writer, x_config_one_sector)

                f_writer.close()

            # compute delta_s (minus the kinetic energy) for the one instanton sector

            for i in range(1, n_lattice):

                second_der_0 = mc.second_derivative_action(x_config_0_one[i])

                potential_0 = 1.0 / 2.0 * second_der_0 * pow((x_config_one_sector[i] - x_config_0_one[i]), 2) \
                              + mc.anh_potential(x_config_0_one[i], 1)

                potential_1 = pow((x_config_one_sector[i] * x_config_one_sector[i] -
                                   ip.x_potential_minimum * ip.x_potential_minimum), 2)

                delta_s_alpha_temp_one = ip.dtau*(potential_1-potential_0)

            delta_s_alpha[1, i_switching] += delta_s_alpha_temp_one
            # error
            delta_s2_alpha[1, i_switching] += pow(delta_s_alpha_temp_one, 2)


            # metropolis zero instanton sector
            mc.metropolis_question_density_switching(
                                                    x_config_zero_sector,
                                                    x_config_0_zero,
                                                    mc.gaussian_potential,
                                                    0,  # zero instanton sector
                                                    a_alpha)

            # Plots some random configuration for random alpha
            if (i_mc in rnd_configuration) and (i_switching == rnd_i_switching):

                with open(output_path + f'/config_0alpha_0sec_{i_mc}.0.txt', 'w') as f_writer:
                    np.savetxt(f_writer, x_config_zero_sector)

                f_writer.close()

            # compute delta_s (minus the kinetic energy) for the one instanton sector

            for i in range(1, n_lattice):

                second_der_0 = mc.second_derivative_action(x_config_0_zero[i])

                potential_0 = 1.0 / 2.0 * second_der_0 * pow((x_config_zero_sector[i] - x_config_0_zero[i]), 2) \
                              + mc.anh_potential(x_config_0_zero[i], 1)

                potential_1 = pow((x_config_zero_sector[i] * x_config_zero_sector[i] -
                                   ip.x_potential_minimum * ip.x_potential_minimum), 2)

                delta_s_alpha_temp_zero = ip.dtau*(potential_1-potential_0)

            delta_s_alpha[0, i_switching] += delta_s_alpha_temp_zero
            # error
            delta_s2_alpha[0, i_switching] += pow(delta_s_alpha_temp_zero , 2)

        # Monte-Carlo end
        # Compute averages and then integrate over alpha

        # 0: zero inst sector, 1: one inst. sector
    for i_sector in [0, 1]:

        delta_s_alpha_av[i_sector, :], delta_s_alpha_err[i_sector, :] = mc.stat_av_var(
                                                                                        delta_s_alpha[i_sector, :],
                                                                                        delta_s2_alpha[i_sector, :],
                                                                                        n_mc_sweeps - n_equil)

        integral_01[i_sector] = np.trapz(
                                        delta_s_alpha_av[i_sector, 0:(n_switching + 1)],
                                        dx=d_alpha)

        integral_10[i_sector] = np.trapz(
                                        delta_s_alpha_av[i_sector, n_switching:(2 * n_switching + 1)],
                                        dx=d_alpha)

        delta_s_av[i_sector] = (integral_01[i_sector] + integral_10[i_sector]) / 2

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

        propagation_error[i_sector] = np.sqrt(integral_01_err[i_sector] + integral_10_err[i_sector])

        hysteresis_error[i_sector] = np.abs(integral_01[i_sector] - integral_10[i_sector])

        # Density computation

        density[i_sector] = np.exp(-delta_s_av[i_sector])

        density_error[i_sector] = density[i_sector] * np.sqrt(
                                                                pow(trapezoidal_error[i_sector], 2) +
                                                                pow(propagation_error[i_sector], 2) +
                                                                pow(hysteresis_error[i_sector], 2))
        
        
    # NG corrections to Test1 simulation
    for i_file in range(3):

        # read instanton density and error for #(cooling sweeps) = 10
        count = 0
        read_lines = 1
        with open(f'./output_data/output_cooled_monte_carlo/n_instantons_{i_file + 1}.txt', 'r') as f_inst, \
             open(f'./output_data/output_cooled_monte_carlo/n_instantons_{i_file + 1}_err.txt', 'r') as f_inst_err:

            while read_lines:

                count += 1
                line = f_inst.readline()
                line_err = f_inst_err.readline()

                if count == 10:
                    gauss_density[i_file] = line
                    gauss_density_err[i_file] = line_err
                    read_lines = 0  # exit

        f_inst.close()
        f_inst_err.close()


        # compute densities with non gaussian contributions
        non_gauss_density[i_file] = gauss_density[i_file] * np.exp(-delta_s_av[1]) * np.exp(delta_s_av[0])

        non_gauss_density_error[i_file] = non_gauss_density[i_file] * \
                                          np.sqrt(pow(gauss_density_err[i_file]/gauss_density[i_file], 2) +
                                                  pow(density_error[0], 2) +
                                                  pow(density_error[1], 2))


        # Print densities
        print(f'For f = {1.4+i_file*0.1}:\n'
              f'Gaussian density from cooling: n0 = {gauss_density[i_file]}\n'
              f'Non-Gaussian density: n = {non_gauss_density[i_file]}\n')

        with open(output_path + f'/non_gaussian_density_{i_file+1}.txt', 'w') as f_inst, \
             open(output_path + f'/non_gaussian_density_{i_file+1}_err.txt', 'w') as f_inst_err:

            f_inst.write(f'{non_gauss_density[i_file]}\n')
            f_inst_err.write(f'{non_gauss_density_error[i_file]}\n')

        f_inst.close()
        f_inst_err.close()

    print('Switching end\n')







