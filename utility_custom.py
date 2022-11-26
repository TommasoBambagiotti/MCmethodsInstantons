'''
Miscellaneous functions
used all along in the different
programs
'''

import os
import shutil
import pathlib as pt
import numpy as np
from prettytable import PrettyTable
#from numba import njit


# MEAN and ERRORS--------------------------------------------------------------


def stat_av_var(observable, observable_sqrd, n_data):
    '''Evaluate the average and the variance of the average of a set of data,
    expressed in an array, directly as the sum and the sum of squares.
    We use the formula Var[<O>] = (<O^2> - <O>^2)/N'''

    if isinstance(observable, np.ndarray) and isinstance(observable_sqrd, np.ndarray):
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

    derivative_log = np.empty((n_array - 1), float)
    derivative_log_err = np.empty((n_array - 1), float)

    for i_array in range(n_array - 1):
        derivative_log[i_array] = - (corr_funct[i_array + 1] - corr_funct[i_array]) \
            / (corr_funct[i_array] * delta_step)

        derivative_log_err[i_array] = pow(
            pow(corr_funct_err[i_array + 1] / corr_funct[i_array], 2)
            + pow(corr_funct_err[i_array] * corr_funct[i_array + 1]
                  / pow(corr_funct[i_array], 2), 2), 1 / 2) / delta_step

    return derivative_log, derivative_log_err


#@njit
def correlation_measurments(n_lattice, n_meas, n_points,
                            x_config, x_cor_sums, x2_cor_sums):

    for _ in range(n_meas):
        i_p0 = int((n_lattice - n_points) * np.random.uniform(0., 1.))
        x_0 = x_config[i_p0]
        for i_point in range(n_points):
            x_1 = x_config[i_p0 + i_point]

            x_cor_sums[0, i_point] += x_0 * x_1
            x_cor_sums[1, i_point] += np.power(x_0 * x_1, 2)
            x_cor_sums[2, i_point] += np.power(x_0 * x_1, 3)

            x2_cor_sums[0, i_point] += np.power(x_0 * x_1, 2)
            x2_cor_sums[1, i_point] += np.power(x_0 * x_1, 4)
            x2_cor_sums[2, i_point] += np.power(x_0 * x_1, 6)



# OUTPUT-----------------------------------------------------------------------

def output_control(path_dir):
    path_output = pt.Path(path_dir)

    for parent_path in path_output.parents:
        if parent_path.exists() is True:
            if parent_path.is_dir() is False:
                print('The partent path is not a directory: Error\n')
                return 0
        else:
            parent_path.mkdir()

    if path_output.exists() is True:
        if path_output.is_dir() is False:
            print('The path is not a directory: Error\n')
            return 0
        return 1

    path_output.mkdir()
    if (path_output.exists() is True) and (path_output.is_dir() is False):
        print('The path is not a directory: Error\n')
        return 0
    return 1


def clear_folder(output_path):

    for filename in os.listdir(output_path):

        file_path = os.path.join(output_path, filename)

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        except Exception as error_message:
            print('Failed to delete %s. Reason: %s' %
                  (file_path, error_message))

def graphical_ui(which_gui):

    gui = PrettyTable()

    if which_gui in ['main']:

        gui.field_names = ['Program', 'Execution #']
        gui.add_row(['Anh. oscill. diagonalization', '0'])
        gui.add_row(['Anh. oscill. Montecarlo simulation', '1'])
        gui.add_row(['Anh. oscill. free en.', '2'])
        gui.add_row(['Inst. cooling', '3'])
        gui.add_row(['Inst. density cooling', '4'])
        gui.add_row(['Inst. density switching', '5'])
        gui.add_row(['Inst. liquid model', '6'])
        gui.add_row(['Inst. liquid model heating', '7'])
        gui.add_row(['Streamline method', '8'])
        gui.add_row(['Inst. inter. liquid model', '9'])
        gui.add_row(['Inst. zero crossing dist.', '10'])
        gui.add_row(['Plots', '11'])
        gui.add_row(['Exit', 'exit'])

        print(gui)

        print('Which (Execution #)?\n')

        return input()

    elif which_gui in ['plots']:

        gui.field_names = ['Plot', 'Plot #']
        gui.add_row(['', 'a'])
        gui.add_row(['', 'b'])
        gui.add_row(['', 'c'])
        gui.add_row(['', 'd'])
        gui.add_row(['', 'e'])
        gui.add_row(['', 'g'])
        gui.add_row(['', 'h'])
        gui.add_row(['', 'i'])
        gui.add_row(['', 'j'])
        gui.add_row(['', 'k'])
        gui.add_row(['', 'l'])
        gui.add_row(['', 'm'])
        gui.add_row(['', 'n'])
        gui.add_row(['', 'o'])
        gui.add_row(['Exit', 'exit'])
        print(gui)
        print('Which (Plot #)?\n')

        return input()




# Monte carlo correlation functions


def output_correlation_functions_and_log(n_points,
                                         x_cor_sums,
                                         x2_cor_sums,
                                         n_config,
                                         output_path,
                                         dtau=0.05):

    if isinstance(x_cor_sums, np.ndarray) \
            and isinstance(x2_cor_sums, np.ndarray):

        x_cor_av = np.zeros(x_cor_sums.shape)
        x_cor_err = np.zeros(x_cor_sums.shape)

        for i_stat in range(3):
            x_cor_av[i_stat], x_cor_err[i_stat] = \
                stat_av_var(x_cor_sums[i_stat],
                            x2_cor_sums[i_stat],
                            n_config
                            )

            with open(output_path + f'/average_x_cor_{i_stat + 1}.txt', 'w',
                      encoding='utf8') as av_writer:
                np.savetxt(av_writer, x_cor_av[i_stat])

            with open(output_path + f'/error_x_cor_{i_stat + 1}.txt', 'w',
                      encoding='utf8') as err_writer:
                np.savetxt(err_writer, x_cor_err[i_stat])

        derivative_log_corr_funct = np.zeros((3, n_points - 1))
        derivative_log_corr_funct_err = np.zeros((3, n_points - 1))

        for i_stat in range(3):

            if i_stat != 1:

                derivative_log_corr_funct[i_stat], derivative_log_corr_funct_err[i_stat] = \
                    log_central_der_alg(
                        x_cor_av[i_stat], x_cor_err[i_stat], dtau)

            else:
                # In the case of log <x^2x^2> the constant part <x^2>
                # is circa the average for the greatest tau

                # subtraction of the constant term (<x^2>)^2 in <x(0)^2x(t)^2>
                cor_funct_err = np.sqrt(np.square(x_cor_err[i_stat])
                                        + pow(x_cor_err[i_stat, -1], 2))

                derivative_log_corr_funct[i_stat], derivative_log_corr_funct_err[i_stat] = \
                    log_central_der_alg(
                        x_cor_av[i_stat] - x_cor_av[i_stat, -1],
                        cor_funct_err,
                        dtau)

                # Save into files

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
                0, np.size(x_cor_sums, 1) * dtau, np.size(x_cor_sums, 1), False))

        return 1

    return 0
