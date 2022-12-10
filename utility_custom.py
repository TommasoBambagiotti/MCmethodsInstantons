'''
Miscellaneous functions
used all along in the different
programs
'''

import os
import shutil
import pathlib as pt
import numpy as np
from numba import njit
from prettytable import PrettyTable


# MEAN and ERRORS--------------------------------------------------------------


def stat_av_var(observable, observable_sqrd, n_data):
    """Evaluate the average and the variance of the average of a set of data,
    expressed in an array, directly as the sum and the sum of squares.

    Parameters
    ----------
    observable : ndarray
        Data.
    observable_sqrd : ndarray
        Data squared.
    n_data : int
        Number of measurements.

    Returns
    -------
    observable_av : ndarray
        Average.
    ndarray
        Standard deviation.
    Notes
    -------
    The variance is computed as Var[<O>] = (<O^2> - <O>^2)/N
    """
    if isinstance(observable, np.ndarray) and \
            isinstance(observable_sqrd, np.ndarray):

        if observable.size != observable_sqrd.size:
            return None, None

    observable_av = observable / n_data
    observable_var = observable_sqrd / (n_data * n_data)
    observable_var -= (np.square(observable_av) / n_data)

    return observable_av, np.sqrt(observable_var)


def log_central_der_alg(corr_funct, corr_funct_err, delta_step):
    """Log-derivative of the correlation functions.
    We can not use the analytic formula because
    we do not know the energy eigenvalues.

    Parameters
    ----------
    corr_funct : ndarray
        Correlation function.
    corr_funct_err : ndarray
        Correlation function error.
    delta_step : float
        Differentiation step.

    Returns
    -------
    derivative_log : ndarray
        Log-derivative of correlation function.
    derivative_log_err : ndarray
        Error of the log-derivative of correlation function.

    Notes
    -------
    We use the forward-difference formula.
    """
    if corr_funct.size != corr_funct_err.size:
        return None, None

    n_array = corr_funct.size

    derivative_log = np.empty((n_array - 1), float)
    derivative_log_err = np.empty((n_array - 1), float)

    for i_array in range(n_array - 1):
        derivative_log[i_array] = - (
                corr_funct[i_array + 1] - corr_funct[i_array]) \
                                  / (corr_funct[i_array] * delta_step)

        derivative_log_err[i_array] = pow(
            pow(corr_funct_err[i_array + 1] / corr_funct[i_array], 2)
            + pow(corr_funct_err[i_array] * corr_funct[i_array + 1]
                  / pow(corr_funct[i_array], 2), 2), 1 / 2) / delta_step

    return derivative_log, derivative_log_err


@njit
def correlation_measurments(n_lattice, n_meas, n_points,
                            x_config, x_cor_sums, x2_cor_sums):
    """Compute correlation functions of spatial coordinates as function
    of euclidean time.

    Parameters
    ----------
    n_lattice : int
        Number of lattice point in euclidean time.
    n_meas : int
        Number of measurement of correlation functions in a MC sweep.
    n_points : int
        Number of points on which correlation functions are computed.
    x_config : ndarray
        Spatial coordinates.
    x_cor_sums : ndarray
        Correlation function.
    x2_cor_sums : ndarray
        Correlation function of squared coordinates.

    Returns
    ----------
    None
    """
    for _ in range(n_meas):
        i_p0 = int((n_lattice - n_points) * np.random.uniform(0., 1.))
        x_0 = x_config[i_p0]
        for i_point in range(n_points):
            x_1 = x_config[i_p0 + i_point]
            x_01 = x_0 * x_1

            x_cor_sums[0, i_point] += x_01
            x_cor_sums[1, i_point] += np.power(x_01, 2)
            x_cor_sums[2, i_point] += np.power(x_01, 3)

            x2_cor_sums[0, i_point] += np.power(x_01, 2)
            x2_cor_sums[1, i_point] += np.power(x_01, 4)
            x2_cor_sums[2, i_point] += np.power(x_01, 6)


# OUTPUT-----------------------------------------------------------------------

def output_control(path_dir):
    """Check if the output directory path is a directory. If it doesn't
    exist, it is created.

    Parameters
    ----------
    path_dir : basestring

    Returns
    -------

    """
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
    """Clean folder.

    Parameters
    ----------
    output_path : basestring

    Returns
    ----------
    None
    """
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
    """Graphic user interface.

    Parameters
    ----------
    which_gui : basestring
        Two interfaces, 'main' and 'plots'.

    Returns
    -------
    None
    """
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
        gui.add_row(['Anh. oscill. Montecarlo corr. func.', 'a'])
        gui.add_row(['Free energy graph.', 'b'])
        gui.add_row(['Montecarlo cooling corr. func.', 'c'])
        gui.add_row(['', 'd'])
        gui.add_row(['Random instanton corr. func.', 'e'])
        gui.add_row(['Random instanton heating corr. func.', 'f'])
        gui.add_row(['', 'g'])
        gui.add_row(['', 'h'])
        gui.add_row(['', 'i'])
        gui.add_row(['', 'j'])
        gui.add_row(['', 'k'])
        gui.add_row(['Instanton interactive liquid model corr. func.', 'l'])
        gui.add_row(['', 'm'])
        gui.add_row(['Groundstate', 'n'])
        gui.add_row(['', 'o'])
        gui.add_row(['', 'p'])
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
    """Save correlation functions and log-derivative of correlation
    functions into files.

    Parameters
    ----------
    n_points : int
        Number of points on which correlation functions are computed.
    x_cor_sums : ndarray
        Correlation function.
    x2_cor_sums : ndarray
        Correlation function of squared coordinates.
    n_config : int
        Number of measurements (for averages).
    output_path : basestring
        Output directory.
    dtau : float, default=0.05
        Lattice spacing.

    Returns
    -------
    int
        If x_cor_sums and x2_cor_sums are ndarray returns 1, otherwise
        returns 0.
    """
    if isinstance(x_cor_sums, np.ndarray) \
            and isinstance(x2_cor_sums, np.ndarray):

        x_cor_av = np.empty(x_cor_sums.shape)
        x_cor_err = np.empty(x_cor_sums.shape)

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

        derivative_log_corr_funct = np.empty((3, n_points - 1))
        derivative_log_corr_funct_err = np.empty((3, n_points - 1))

        for i_stat in range(3):

            if i_stat != 1:

                derivative_log_corr_funct[i_stat], \
                derivative_log_corr_funct_err[i_stat] = \
                    log_central_der_alg(
                        x_cor_av[i_stat], x_cor_err[i_stat], dtau)

            else:
                # In the case of log <x^2x^2> the constant part <x^2>
                # is circa the average for the greatest tau

                # subtraction of the constant term (<x^2>)^2 in <x(0)^2x(t)^2>
                cor_funct_err = np.sqrt(np.square(x_cor_err[i_stat])
                                        + pow(x_cor_err[i_stat, -1], 2))

                derivative_log_corr_funct[i_stat], \
                derivative_log_corr_funct_err[i_stat] = \
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
            np.savetxt(tau_writer,
                       np.linspace(0, np.size(x_cor_sums, 1) * dtau,
                                   np.size(x_cor_sums, 1), False))

        return 1

    return 0


def cor_plot_setup(which_plot=None):
    """Customise correlation function plots.

    Returns
    -------
    dict
        Plot configuration.
    """
    if which_plot in ['a']:
        # montecarlo
        # 2: log-derivative corr. func.; 1: corr. func.
        return {'x_inf_1': -0.05,
                'x_sup_1': 1.03,
                'x_inf_2': -0.05,
                'x_sup_2': 1.1,
                'y_inf_2': -1.0,
                'y_sup_2': 5,
                'cor1_s': 7,
                'cor2_s': 17,
                'cor3_s': 7,
                'cor2_s_fig1': 7}
    elif which_plot in ['c']:
        # montecarlo cooling
        return {'x_inf_1': -0.05,
                'x_sup_1': 1.03,
                'x_inf_2': -0.05,
                'x_sup_2': 1,
                'y_inf_2': -1.0,
                'y_sup_2': 5,
                'cor1_s': 0,
                'cor2_s': 8,
                'cor3_s': 0,
                'cor2_s_fig1': 0}
    elif which_plot in ['e']:
        #rilm
        return {'x_inf_1': -0.05,
                'x_sup_1': 1,
                'x_inf_2': -0.05,
                'x_sup_2': 1,
                'y_inf_2': -1.0,
                'y_sup_2': 5,
                'cor1_s': 0,
                'cor2_s': 7,
                'cor3_s': 0,
                'cor2_s_fig1': 0}
    elif which_plot in ['f']:
        # rilm heating
        return {'x_inf_1': -0.05,
                'x_sup_1': 1,
                'x_inf_2': -0.05,
                'x_sup_2': 1,
                'y_inf_2': -1.0,
                'y_sup_2': 5,
                'cor1_s': 0,
                'cor2_s': 8,
                'cor3_s': 0,
                'cor2_s_fig1': 0}
    elif which_plot in ['l']:
        # iilm
        return {'x_inf_1': -0.05,
                'x_sup_1': 1,
                'x_inf_2': -0.05,
                'x_sup_2': 1,
                'y_inf_2': -1.0,
                'y_sup_2': 5,
                'cor1_s': 0,
                'cor2_s': 15,
                'cor3_s': 0,
                'cor2_s_fig1': 0}
    else:
        return {'x_inf_1': -0.05,
                'x_sup_1': 1.5,
                'x_inf_2': -0.05,
                'x_sup_2': 1.5,
                'y_inf_2': -1,
                'y_sup_2': 5,
                'cor1_s': 0,
                'cor2_s': 0,
                'cor3_s': 0,
                'cor2_s_fig1': 0}
