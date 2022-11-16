'''
Miscellaneous functions
used all along in the different
programs
'''

import os
import shutil
import pathlib as pt
import numpy as np

import input_parameters as ip

# MEAN and ERRORS--------------------------------------------------------------

def stat_av_var(observable, observable_sqrd, n_data):
    '''Evaluate the average and the variance of the average of a set of data,
    expressed in an array, directly as the sum and the sum of squares.
    We use the formula Var[<O>] = (<O^2> - <O>^2)/N'''

    if (type(observable) is np.ndarray) and (type(observable_sqrd) is np.ndarray):
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

# INPUT -----------------------------------------------------------------------










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
    else:
        path_output.mkdir()
        if (path_output.exists() is True) and (path_output.is_dir() is False):
            print('The path is not a directory: Error\n')
            return 0
            
            
def clear_folder(output_path):

    for filename in os.listdir(output_path):

        file_path = os.path.join(output_path, filename)

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# Monte carlo correlation functions

def output_correlation_functions_and_log(n_points,
                                         x_cor_sums,
                                         x2_cor_sums,
                                         n_config,
                                         output_path):

    if type(x_cor_sums) is np.ndarray \
        and type(x2_cor_sums) is np.ndarray:
        
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
                        x_cor_av[i_stat], x_cor_err[i_stat], ip.dtau)
    
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
                0, np.size(x_cor_sums, 1) * ip.dtau, np.size(x_cor_sums, 1), False))
    
    else:
        return 0