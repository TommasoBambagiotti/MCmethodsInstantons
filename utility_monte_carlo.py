import numpy as np
import random as rnd


def metropolis_question(x_config,
                        dtau,
                        delta_x,
                        x_potential_minimum):
    '''
    '''
    n_lattice = x_config.size - 1

    for i in range(1, n_lattice):

        action_loc_old = (
                                 pow((x_config[i] - x_config[i - 1]) / (2 * dtau), 2) +
                                 pow((x_config[i + 1] - x_config[i]) / (2 * dtau), 2) +
                                 pow(x_config[i] * x_config[i] -
                                     x_potential_minimum * x_potential_minimum, 2)
                         ) * dtau

        x_new = x_config[i] + rnd.gauss(0, delta_x)

        action_loc_new = (
                                 pow((x_new - x_config[i - 1]) / (2 * dtau), 2) +
                                 pow((x_config[i + 1] - x_new) / (2 * dtau), 2) +
                                 pow(x_new * x_new -
                                     x_potential_minimum * x_potential_minimum, 2)) * dtau

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

    return x_config

def initialize_lattice(n_lattice,
                       i_cold,
                       x_potential_minimum):
    if i_cold is True:
        x_config = np.repeat(-x_potential_minimum, n_lattice + 1)
    else:
        x_config = np.random.uniform(-x_potential_minimum,
                                     x_potential_minimum,
                                     n_lattice + 1)
        x_config[n_lattice - 1] = x_config[0]
        x_config[n_lattice] = x_config[1]

    return x_config

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

def find_instantons(x,n_lattice,dt):

    pos_roots = 0
    neg_roots = 0
    pos_roots_position = np.zeros(1)
    neg_roots_position = np.zeros(1)
    i_zero = 0

    while(i_zero < n_lattice) and (x[i_zero] == 0):
        i_zero +=1

    x_pos = x[i_zero]
    #print(f'x_pos ={x[i_zero]} with i_zero = {i_zero}')

    for i in range(i_zero+1,n_lattice):

        if x_pos*x[i] < 0 :

            if x[i]-x_pos > 0:
                pos_roots += 1
                pos_roots_position = np.append(pos_roots_position, i*dt)
                x_pos = x[i]

            elif x[i]-x_pos < 0:

                neg_roots += 1
                neg_roots_position = np.append(neg_roots_position,i*dt)
                x_pos = x[i]

    return pos_roots,neg_roots,pos_roots_position,neg_roots_position