
import numpy as np

import utility_monte_carlo as mc
import utility_custom
import input_parameters as ip
import utility_cooling as cool


def non_gaussian_inst_density(n_lattice,  # size of the grid
                             n_equil,  # equilibration sweeps
                             n_mc_sweeps,  # monte carlo sweeps
                             n_switching,  #
                             i_cold):  # cold/hot st): 

    # initialize lattice and other arrays
    
    x_config_inst, x_config_inst_0, potential_initial, anharmonic_frequency = \
        cool.initialize_instanto_lattice(n_lattice)
        
    d_alpha = 1.0 / n_switching

    delta_s_alpha = np.zeros((2 * n_switching + 1))
    delta_s_alpha2 = np.zeros((2 * n_switching + 1))
    print(f'Adiabatic switching for beta = {n_lattice * 0.05}')
    # Now the principal cycle is over the coupling constant alpha
    for i_switching in range(2 * n_switching + 1):

        if i_switching <= n_switching:
            a_alpha = i_switching * d_alpha
        else:
            a_alpha = 2.0 - (i_switching) * d_alpha
            
        
        print(f'Switching #{i_switching}')

        for i_equil in range(n_equil):
           cool.metropolis_question(x_config_inst, a_alpha)

        for i_mc in range(n_equil, n_mc_sweeps):

            delta_s_alpha_temp = 0.0
            cool.metropolis_question(x_config_inst, a_alpha)
            for j in range(n_lattice):
                potential_0 = pow(ip.w_omega0 * x_config_inst[j], 2) / 4.0
                potential_1 = pow(x_config_inst[j] * x_config_inst[j]
                                  - (ip.x_potential_minimum
                                  * ip.x_potential_minimum)
                                  , 2)
                delta_s_alpha_temp += (potential_1 - potential_0) * ip.dtau

            delta_s_alpha[i_switching] += delta_s_alpha_temp
            delta_s_alpha2[i_switching] += pow(delta_s_alpha_temp, 2)