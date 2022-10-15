'''

'''
import random as rnd
import numpy as np
import utility_custom


def stat_av_var(observable, observable_sqrd, n_data):
    '''Evaluate the average and the variance of the average of a set of data,
    expressed in an array, directly as the sum and the sum of squares.
    We use the formula Var[<O>] = (<O^2> - <O>^2)/N'''
    # Control
    if observable.size != observable_sqrd.size:
        return None, None

    observable_av = observable / n_data
    observable_var = observable_sqrd / (n_data * n_data)
    observable_var -= (np.square(observable_av) / n_data)

    return observable_av, np.sqrt(observable_var)


def log_central_der_alg(corr_funct, corr_funct_err, delta_step):
    '''Log derivative of the correlation functions.
    We can not use the analytic formula because
    we do not know the energy eignevalues.'''

    if corr_funct.size != corr_funct_err.size:
        return None, None

    n_array = corr_funct.size
    # Reference method
    derivative_log = np.empty((n_array-1), float)
    derivative_log_err = np.empty((n_array-1), float)

    for i_array in range(n_array-1):
        derivative_log[i_array] = - (corr_funct[i_array+1] -corr_funct[i_array]) \
                                    / (corr_funct[i_array]*delta_step)

        derivative_log_err[i_array] = pow(
            pow(corr_funct_err[i_array+1] / corr_funct[i_array], 2)
            + pow(corr_funct_err[i_array] * corr_funct[i_array+1]
            / pow(corr_funct[i_array], 2), 2), 1/2) / delta_step

    return derivative_log, derivative_log_err


def initialize_lattice(n_lattice,
                       i_cold,
                       x_potential_minimum):
    if i_cold is True:
        x_config = np.repeat(-x_potential_minimum, n_lattice + 1)
    else:
        x_config = np.random.uniform(-x_potential_minimum,
                                     x_potential_minimum,
                                     n_lattice + 1)
        x_config[n_lattice-1] = x_config[0]
        x_config[n_lattice] = x_config[1]

    return x_config


def metropolis_question(x_config,
                        dtau,
                        delta_x,
                        x_potential_minimum):
    '''
    '''
    n_lattice = x_config.size - 1

    for i in range(1, n_lattice):

        action_loc_old = (
            pow((x_config[i] - x_config[i-1])/(2*dtau), 2) +
            pow((x_config[i+1] - x_config[i])/(2*dtau), 2) +
            pow(x_config[i] * x_config[i] -
                x_potential_minimum*x_potential_minimum, 2)
        )*dtau

        x_new = x_config[i] + rnd.gauss(0, delta_x)

        action_loc_new = (
            pow((x_new - x_config[i-1])/(2*dtau), 2) +
            pow((x_config[i+1] - x_new)/(2*dtau), 2) +
            pow(x_new * x_new -
                x_potential_minimum*x_potential_minimum, 2))*dtau

        delta_action = action_loc_new - action_loc_old

        # we put a bound on the value of delta_S
        # because we need the exp.
        delta_action = max(delta_action, -70.0)
        delta_action = min(delta_action, 70.0)
        # Metropolis question:
        if np.exp(-delta_action) > rnd.uniform(0., 1.):
            x_config[i] = x_new

    x_config[0] = x_config[n_lattice-1]
    x_config[n_lattice] = x_config[1]

    return x_config


def monte_carlo_ao(x_potential_minimum,  # potential well position
                   n_lattice,  # size of the grid
                   dtau,  # grid spacing in time
                   n_equil,  # equilibration sweeps
                   n_mc_sweeps,  # monte carlo sweeps
                   delta_x,  # width of gauss. dist. for update x
                   n_points,  #
                   n_meas,  #
                   i_cold):  # cold/hot start):
    '''Solve the anharmonic oscillator through
    Monte Carlo technique on an Euclidian Axis'''

    if n_mc_sweeps < n_equil:
        print("too few Monte Carlo sweeps/ N_equilib > N_Monte_Carlo")
        return 0

    # Control output filepath
    output_path = r'.\output_data\output_monte_carlo'
    utility_custom.output_control(output_path)

    # Correlation functions
    x_cor_sums = np.zeros((3, n_points))
    x2_cor_sums = np.zeros((3, n_points))

    # x position along the tau axis
    x_config = initialize_lattice(n_lattice, i_cold, x_potential_minimum)

    # Monte Carlo sweeps: Principal cycle

    # Equilibration cycle
    for i_equil in range(n_equil):
        x_config = metropolis_question(x_config, dtau, delta_x, x_potential_minimum)

        # Rest of the MC sweeps
    with open(output_path + r'\ground_state_histogram.dat', 'wb') as hist_writer:
        for i_mc in range(n_mc_sweeps - n_equil):
            x_config = metropolis_question(x_config, dtau, delta_x, x_potential_minimum)
            np.save(hist_writer, x_config[0:(n_lattice-1)])
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
    x_cor_av, x_cor_err = stat_av_var(x_cor_sums[0],
                                      x2_cor_sums[0],
                                      n_meas*(n_mc_sweeps - n_equil))
    x_cor_2_av, x_cor_2_err = stat_av_var(x_cor_sums[1],
                                          x2_cor_sums[1],
                                          n_meas*(n_mc_sweeps - n_equil))
    x_cor_3_av, x_cor_3_err = stat_av_var(x_cor_sums[2],
                                          x2_cor_sums[2],
                                          n_meas*(n_mc_sweeps - n_equil))
    
    with open(output_path + r'\tau_array.txt', 'w') as tau_writer:
        np.savetxt(tau_writer,
                   np.linspace(0, n_points * dtau, n_points, False))
    with open(output_path + r'\average_x_cor.txt', 'w') as av_writer:
        np.savetxt(av_writer, x_cor_av)
    with open(output_path + r'\error_x_cor.txt', 'w') as err_writer:
        np.savetxt(err_writer, x_cor_err)
    with open(output_path + r'\average_x_cor_2.txt', 'w') as av_writer:
        np.savetxt(av_writer, x_cor_2_av)
    with open(output_path + r'\error_x_cor_2.txt', 'w') as err_writer:
        np.savetxt(err_writer, x_cor_2_err)
    with open(output_path + r'\average_x_cor_3.txt', 'w') as av_writer:
        np.savetxt(av_writer, x_cor_3_av)
    with open(output_path + r'\error_x_cor_3.txt', 'w') as err_writer:
        np.savetxt(err_writer, x_cor_3_err)

    # Correlation function Log
    derivative_log_corr_funct, derivative_log_corr_funct_err = \
        log_central_der_alg(x_cor_av, x_cor_err, dtau)

    # In the case of log <x^2x^2> the constant part <x^2>
    # is circa the average for the greatest tau

    derivative_log_corr_funct_2, derivative_log_corr_funct_2_err = \
        log_central_der_alg(x_cor_2_av - x_cor_2_av[n_points-1],
                            np.sqrt(x_cor_2_err * x_cor_2_err + pow(x_cor_2_err, 2)),
                            dtau)

    derivative_log_corr_funct_3, derivative_log_corr_funct_3_err = \
        log_central_der_alg(x_cor_3_av, x_cor_3_err, dtau)

    with open(output_path + r'\average_der_log.txt', 'w') as av_writer:
        np.savetxt(av_writer, derivative_log_corr_funct)
    with open(output_path + r'\error_der_log.txt', 'w') as err_writer:
        np.savetxt(err_writer, derivative_log_corr_funct_err)
    with open(output_path + r'\average_der_log_2.txt', 'w') as av_writer:
        np.savetxt(av_writer, derivative_log_corr_funct_2)
    with open(output_path + r'\error_der_log_2.txt', 'w') as err_writer:
        np.savetxt(err_writer, derivative_log_corr_funct_2_err)
    with open(output_path + r'\average_der_log_3.txt', 'w') as av_writer:
        np.savetxt(av_writer, derivative_log_corr_funct_3)
    with open(output_path + r'\error_der_log_3.txt', 'w') as err_writer:
        np.savetxt(err_writer, derivative_log_corr_funct_3_err)

    return 1

if __name__ == '__main__':

    monte_carlo_ao(1.4,  # x_potential_minimum
                   800,  # n_lattice
                   0.05,  # dtau
                   100,  # n_equil
                   100000,  # n_mc_sweeps
                   0.45,  # delta_x
                   20,  # n_point
                   5,  # n_meas
                   False)
