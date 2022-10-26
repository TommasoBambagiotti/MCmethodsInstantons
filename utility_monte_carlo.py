import numpy as np
import random as rnd
import input_parameters as ip


def potential_alpha(x_position,
                    a_alpha):
    if (a_alpha > -0.01):
        potential_1 = pow(x_position * x_position -
                          ip.x_potential_minimum * ip.x_potential_minimum, 2)
        potential_0 = pow(ip.w_omega0 * x_position, 2) / 4.0
        return a_alpha * (potential_1 - potential_0) + potential_0
    else:
        return pow(x_position * x_position -
                   ip.x_potential_minimum * ip.x_potential_minimum, 2)


def metropolis_question(x_config,
                        a_alpha=-1.0):

    n_lattice = x_config.size - 1

    for i in range(1, n_lattice):

        action_loc_old = (
                pow((x_config[i] - x_config[i - 1]) / (2 * ip.dtau), 2)
                + pow((x_config[i + 1] - x_config[i]) / (2 * ip.dtau), 2)
                + potential_alpha(x_config[i], a_alpha)
                ) * ip.dtau

        x_new = x_config[i] + rnd.gauss(0, ip.delta_x)

        action_loc_new = (
                pow((x_new - x_config[i - 1]) / (2 * ip.dtau), 2)
                + pow((x_config[i + 1] - x_new) / (2 * ip.dtau), 2)
                + potential_alpha(x_new, a_alpha)
                ) * ip.dtau
        
        delta_action = action_loc_new - action_loc_old

        # we put a bound on the value of delta_S
        # because we need the exp.
        delta_action = max(delta_action, -70.0)
        delta_action = min(delta_action, 70.0)
        # Metropolis question:
        if np.exp(-delta_action) > rnd.uniform(0., 1.):
            x_config[i] = x_new

    x_config[0] = x_config[n_lattice - 1]
    x_config[n_lattice] = x_config[1]


def return_action(x_config):
    
    n_lattice = x_config.size - 1
    action = 0.0
    
    for i_pos in range(1, n_lattice):
        action += (pow((x_config[i_pos] - x_config[i_pos - 1]) / (2 * ip.dtau), 2) 
            + pow(x_config[i_pos] * x_config[i_pos] -
                ip.x_potential_minimum * ip.x_potential_minimum, 2)
            ) * ip.dtau
        
    return action


def initialize_lattice(n_lattice,
                       i_cold):
    if i_cold is True:
        x_config = np.repeat(-ip.x_potential_minimum, n_lattice + 1)
    else:
        x_config = np.random.uniform(-ip.x_potential_minimum,
                                     ip.x_potential_minimum,
                                     n_lattice + 1)
        x_config[n_lattice - 1] = x_config[0]
        x_config[n_lattice] = x_config[1]

    return x_config


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
    
    # Reference method
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

