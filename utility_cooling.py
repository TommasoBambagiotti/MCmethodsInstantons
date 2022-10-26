
import numpy as np
import random as rnd
import initial_parameters as ip


def configuration_cooling(x_cold_config,
                          x_potential_minimum):
    n_lattice = x_cold_config.size - 1
    n_trials = 10

    for i in range(1, n_lattice):

        action_loc_old = (
                                 pow((x_cold_config[i] - x_cold_config[i - 1]) / (2 * ip.dtau), 2) +
                                 pow((x_cold_config[i + 1] - x_cold_config[i]) / (2 * ip.dtau), 2) +
                                 pow(x_cold_config[i] * x_cold_config[i] -
                                     x_potential_minimum * x_potential_minimum, 2)
                         ) * ip.dtau
        for j in range(1, n_trials):  # perchè??
    
            x_new = x_cold_config[i] + rnd.gauss(0, ip.delta_x * 0.1)
    
            action_loc_new = (
                                    pow((x_new - x_cold_config[i - 1]) / (2 * ip.dtau), 2) +
                                    pow((x_cold_config[i + 1] - x_new) / (2 * ip.dtau), 2) +
                                    pow(x_new * x_new -
                                        x_potential_minimum * x_potential_minimum, 2)
                             ) * ip.dtau
    
            if ((action_loc_new - action_loc_old) < 0):
                    x_cold_config[i] = x_new


    x_cold_config[0] = x_cold_config[n_lattice - 1]
    x_cold_config[n_lattice] = x_cold_config[1]


def find_instantons(x,n_lattice,dt):

    pos_roots = 0
    neg_roots = 0
    #pos_roots_position = np.zeros(1)
    #neg_roots_position = np.zeros(1)

    #print(f'x_pos ={x[i_zero]} with i_zero = {i_zero}')

    if x[0] == 0:

        if x[1] - x[0] > 0:
            pos_roots += 1
            #pos_roots_position = np.append(pos_roots_position, 0)
            i_zero = 1
            x_pos = x[i_zero]


        elif x[1] - x[0] < 0:

            neg_roots += 1
            #neg_roots_position = np.append(neg_roots_position, 0)
            i_zero = 1
            x_pos = x[i_zero]
    else:

        i_zero = 0
        x_pos = x[i_zero]

    for i in range(i_zero+1,n_lattice):


        if x_pos * x[i] < 0 :

            if x[i]-x[i-1] > 0:
                pos_roots += 1
                #pos_roots_position = np.append(pos_roots_position, (i*dt+(i-1)*dt)/2)
                x_pos = x[i]

            elif x[i]-x[i-1] < 0:

                neg_roots += 1
                #neg_roots_position = np.append(neg_roots_position,(i*dt+(i-1)*dt)/2)
                x_pos = x[i]
    #a = np.delete(pos_roots_position, 0)
    #b = np.delete(neg_roots_position, 0)

    return pos_roots,neg_roots#,a ,b

def initialize_instanton_lattice(n_lattice):
    
    tau_array = np.linspace(0, (n_lattice-1) * ip.dtau, n_lattice-1, False)
    x_config = np.tanh( (2.0 * ip.x_potential_minimum) * (tau_array - tau_array[:-1]/2)) \
        * ip.x_potential_minimum
    
    potential_initial = np.square(np.square(x_config) - pow(ip.x_potential_minimum, 2))
    
    anharmonic_frequency = 12 * np.square(x_config) - 4 * ip.x_potential_minimum
    
    x_config[n_lattice - 1] = - x_config[0]
    
    x_config.append(-x_config[1])
    
    x_config_initial = np.copy(x_config)
    
    return x_config, \
        x_config_initial, \
        potential_initial, \
        anharmonic_frequency
            

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