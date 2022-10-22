import numpy as np
import utility_monte_carlo as mc
import utility_custom
import random as rnd
import input_parameters as ip


def cooled_monte_carlo(
        n_lattice,
        n_equil,
        n_mc_sweeps,
        n_points,
        n_meas,
        i_cold,
        n_sweeps_btw_cooling,
        n_cooling_sweeps):
    # contatore probabilmente inutile
    n_cooling = 0

    if n_mc_sweeps < n_equil:
        print("too few Monte Carlo sweeps/ N_equilib > N_Monte_Carlo")
        return 0

    # Control output filepath
    output_path = r'.\output_data\output_cooled_monte_carlo'
    utility_custom.output_control(output_path)

    # Correlation functions
    x_cor_sums = np.zeros((3, n_points))
    x2_cor_sums = np.zeros((3, n_points))
    x_cold_cor_sums = np.zeros((3, n_points))
    x2_cold_cor_sums = np.zeros((3, n_points))

    # x position along the tau axis
    x_config = mc.initialize_lattice(n_lattice, i_cold)

    # number of instantons
    n_total_instantons_sum = np.zeros(1)
    n2_total_instantons_sum = np.zeros(1)

    # Monte Carlo sweeps: Principal cycle

    # Equilibration cycle
    for i_equil in range(n_equil):
        x_config = mc.metropolis_question(x_config)

    # Rest of the MC sweeps
    with open(output_path + r'\ground_state_histogram.dat', 'wb') as hist_writer:
        for i_mc in range(n_mc_sweeps - n_equil):
            x_config = mc.metropolis_question(x_config)

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

            # COOLING

            if (i_mc % n_sweeps_btw_cooling) == 0:

                # expected number of cooled configuration = n_conf/n_sweeps_btw_cooling
                print(f'cooling #{n_cooling} of {(n_mc_sweeps - n_equil) / n_sweeps_btw_cooling}\n'
                      f'in configuration #{i_mc}')

                x_cold_config_temp = np.copy(x_config)
                n_cooling += 1

                for i_cooling in range(n_cooling_sweeps):
                    x_cold_config = mc.configuration_cooling(x_cold_config_temp,
                                                             ip.dtau,
                                                             ip.delta_x * 0.1,  # prova, come nel codice
                                                             ip.x_potential_minimum)

                    x_cold_config_temp = np.copy(x_cold_config)

                # Find instantons and anti-instantons
                n_instantons, n_anti_instantons, pos_instantons, pos_anti_instantons = \
                    mc.find_instantons(x_cold_config, n_lattice, ip.dtau)

                n_total_instantons_sum += (n_instantons + n_anti_instantons)

                n2_total_instantons_sum += pow((n_instantons + n_anti_instantons), 2)

                # Compute correlation functions for the cooled configuration
                for k_meas in range(n_meas):

                    i_p0 = int((n_lattice - n_points) * rnd.uniform(0., 1.))
                    x_0 = x_cold_config[i_p0]

                    for i_point in range(n_points):
                        x_1 = x_cold_config[i_p0 + i_point]

                        x_cold_cor_sums[0, i_point] += x_0 * x_1
                        x_cold_cor_sums[1, i_point] += pow(x_0 * x_1, 2)
                        x_cold_cor_sums[2, i_point] += pow(x_0 * x_1, 3)

                        x2_cold_cor_sums[0, i_point] += pow(x_0 * x_1, 2)
                        x2_cold_cor_sums[1, i_point] += pow(x_0 * x_1, 4)
                        x2_cold_cor_sums[2, i_point] += pow(x_0 * x_1, 6)

        # End of Montecarlo simulation

        # Correlation functions - w/o & w cooling

        x_cor_av = np.zeros((3, n_points))
        x_cor_err = np.zeros((3, n_points))

        x_cold_cor_av = np.zeros((3, n_points))
        x_cold_cor_err = np.zeros((3, n_points))

        for i_stat in range(3):
            # w/o cooling
            x_cor_av[i_stat], x_cor_err[i_stat] = mc.stat_av_var(x_cor_sums[i_stat],
                                                                 x2_cor_sums[i_stat],
                                                                 n_meas * (n_mc_sweeps - n_equil))
            # w cooling
            x_cold_cor_av[i_stat], x_cold_cor_err[i_stat] = mc.stat_av_var(x_cold_cor_sums[i_stat],
                                                                           x2_cold_cor_sums[i_stat],
                                                                           n_meas * n_cooling)
            # Save into files
            # w/o cooling

            with open(output_path + f'/average_x_cor_{i_stat + 1}.txt', 'w') as av_writer:
                np.savetxt(av_writer, x_cor_av[i_stat])

            with open(output_path + f'/error_x_cor_{i_stat + 1}.txt', 'w') as err_writer:
                np.savetxt(err_writer, x_cor_err[i_stat])

            # w cooling

            with open(output_path + f'/average_x_cor_cold_{i_stat + 1}.txt', 'w') as av_writer:
                np.savetxt(av_writer, x_cold_cor_av[i_stat])

            with open(output_path + f'/error_x_cor_cold_{i_stat + 1}.txt', 'w') as err_writer:
                np.savetxt(err_writer, x_cold_cor_err[i_stat])

        # Total number of instantons

        n_total_instantons_av = mc.stat_av_var(n_total_instantons_sum, n2_total_instantons_sum, n_cooling)

        # Log derivative of correlation functions

        # In the case of log <x^2x^2> the constant part <x^2>
        # is circa the average for the greatest tau

        derivative_log_cor_funct = np.zeros((3, n_points - 1))
        derivative_log_cor_funct_err = np.zeros((3, n_points - 1))

        derivative_log_cold_cor_funct = np.zeros((3, n_points - 1))
        derivative_log_cold_cor_funct_err = np.zeros((3, n_points - 1))

        for i_stat in range(3):

            if i_stat != 1:

                # w/o cooling
                derivative_log_cor_funct[i_stat], derivative_log_cor_funct_err[i_stat] = \
                    mc.log_central_der_alg(x_cor_av[i_stat], x_cor_err[i_stat], ip.dtau)
                # w cooling
                derivative_log_cold_cor_funct[i_stat], derivative_log_cold_cor_funct_err[i_stat] = \
                    mc.log_central_der_alg(x_cold_cor_av[i_stat], x_cold_cor_err[i_stat], ip.dtau)


            else:

                # subtraction of the constant term (<x^2>)^2 in <x(0)^2x(t)^2>
                cor_funct_err = np.sqrt(x_cor_err[i_stat] * x_cor_err[i_stat]
                                        + pow(x_cor_err[i_stat, n_points - 1], 2))

                cold_cor_funct_err = np.sqrt(x_cold_cor_err[i_stat] * x_cold_cor_err[i_stat] \
                                             + pow(x_cold_cor_err[i_stat, n_points - 1], 2))
                # w/o cooling
                derivative_log_cor_funct[i_stat], derivative_log_cor_funct_err[i_stat] = \
                    mc.log_central_der_alg(
                        x_cor_av[i_stat] - x_cor_av[i_stat, n_points - 1],
                        cor_funct_err,
                        ip.dtau)
                # w cooling
                derivative_log_cold_cor_funct[i_stat], derivative_log_cold_cor_funct_err[i_stat] = \
                    mc.log_central_der_alg(
                        x_cold_cor_av[i_stat] - x_cold_cor_av[i_stat, n_points - 1],
                        cold_cor_funct_err,
                        ip.dtau)
                # Save into files

            # w/o cooling
            with open(output_path + '/average_der_log_1.txt', 'w') as av_writer:
                np.savetxt(av_writer, derivative_log_cor_funct[i_stat])

            with open(output_path + '/error_der_log_1.txt', 'w') as err_writer:
                np.savetxt(err_writer, derivative_log_cor_funct_err[i_stat])

            # w cooling

            with open(output_path + '/average_der_log_1.txt', 'w') as av_writer:
                np.savetxt(av_writer, derivative_log_cold_cor_funct[i_stat])

            with open(output_path + '/error_der_log_1.txt', 'w') as err_writer:
                np.savetxt(err_writer, derivative_log_cold_cor_funct_err[i_stat])

        # time array
        with open(output_path + '/tau_array.txt', 'w') as tau_writer:

            np.savetxt(tau_writer, np.linspace(0, n_points * ip.dtau, n_points, False))

    return 1
