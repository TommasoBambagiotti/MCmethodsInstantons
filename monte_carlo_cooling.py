"""
Monte carlo cooling of anharmonic configuration path
"""

import random as rnd
import numpy as np

import utility_monte_carlo as mc
import utility_custom
import input_parameters as ip


def cooling_algorithm(x_config,
                      n_cooling):
    n_trials = 10
    n_lattice = x_config.size - 1

    for i_cool in range(n_cooling):
        
        for i in range(1, n_lattice):

            action_loc_old = (
                pow((x_config[i] - x_config[i - 1]) / (2 * ip.dtau), 2) +
                pow((x_config[i + 1] - x_config[i]) / (2 * ip.dtau), 2) +
                pow(pow(x_config[i], 2) - pow(ip.x_potential_minimum, 2), 2)
                ) * ip.dtau

            for i_trial in range(n_trials):
                x_new = x_config[i] + rnd.gauss(0, ip.delta_x * 0.1)

                action_loc_new = (
                pow((x_new - x_config[i - 1]) / (2 * ip.dtau), 2) +
                pow((x_config[i + 1] - x_new) / (2 * ip.dtau), 2) +
                pow(pow(x_config[i], 2) - pow(ip.x_potential_minimum, 2), 2)
                ) * ip.dtau

            # Metropolis question cooled version
                if action_loc_new < (action_loc_old + 0.0001):
                    x_config[i] = x_new

                x_config[0] = x_config[n_lattice - 1]
                x_config[n_lattice] = x_config[1]
        
    

def monte_carlo_cooling(n_lattice,  # size of the grid
                        n_equil,  # equilibration sweeps
                        n_mc_sweeps,  # monte carlo sweeps
                        #n_points,  #
                        #n_meas,
                        n_cooling,
                        n_sweeps_btw_cooling,
                        i_cold):  # 
    '''
    
    '''
    # Output control
    output_path = './output_data/monte_carlo_cooling'
    utility_custom.output_control(output_path)
    
    # Correlators functions
    #x_cor_sums = np.zeros((3, n_points))
    #x2_cor_sums = np.zeros((3, n_points))
    
    #x_cor_sums_cool = np.zeros((3, n_points))
    #x2_cor_sums_cool = np.zeros((3, n_points))
    
    n_istantons = np.zeros((n_mc_sweeps / n_sweeps_btw_cooling), float)
    n_anti_istantons = np.zeros((n_mc_sweeps / n_sweeps_btw_cooling), float)
    
    x_config = mc.initialize_lattice(n_lattice, i_cold)
    
    i_cooled_configuration = 0
    
    # Equilibrization cycle
    for i_equil in range(n_equil):
        x_config = mc.metropolis_question(x_config)
        
    # Rest of sweeps 
    for i_mc in range(n_mc_sweeps - n_equil):
        x_config = mc.metropolis_question(x_config)
        
        if (i_mc % n_sweeps_btw_cooling) == 0:
            x_config_cooled = cooling_algorithm(x_config,
                                                n_cooling)
            
            n_istantons[i_cooled_configuration], \
                n_anti_istantons[i_cooled_configuration] = \
                    mc.find_instantons(x_config_cooled, n_lattice, ip.dtau)
            
            i_cooled_configuration += 1
        
    tau_array = np.linspace(0.0, 5.0, n_lattice, False)
    
    with open(output_path + '/tau_array.txt', 'w') as tau_writer:
        np.savetxt(tau_writer, tau_array)
    with open(output_path + '/configuration_cooled.txt', 'w') as cool_writer: 
        np.savetxt(cool_writer, x_config_cooled)
    with open(output_path + '/configuration.txt', 'w') as x_writer:
        np.savetxt(x_writer, x_config)
            

monte_carlo_cooling(800, 
                    100, 
                    5000, 
                    50,
                    20, 
                    False)                    
                    