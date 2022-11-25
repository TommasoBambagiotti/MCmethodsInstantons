'''
Random instanton-anti instanton
gas model
'''

import numpy as np

import utility_custom
import utility_rilm as rilm


def random_instanton_liquid_model(n_lattice,  # size of the grid
                                  n_mc_sweeps,  # monte carlo sweeps
                                  n_points,  #
                                  n_meas,
                                  n_ia = 0,
                                  x_potential_minimum = 1.4,
                                  dtau = 0.05):

    # Control output filepath
    output_path = './output_data/output_rilm'
    utility_custom.output_control(output_path)

    # Eucliadian time
    tau_array = np.linspace(0.0, n_lattice * dtau, n_lattice, False)

    # Correlation functions
    x_cor_sums = np.zeros((3, n_points))
    x2_cor_sums = np.zeros((3, n_points))

    if n_ia == 0:
        # n_ia evaluated from 2-loop semi-classical expansion
        action_0 = np.power(x_potential_minimum,3) *4/6
        loop_2 = 8 * pow(x_potential_minimum, 5 / 2) \
            * pow(2 / np.pi, 1/2) * np.exp(-action_0 - 71 / (72 * action_0))
            
        n_ia = int(np.rint(loop_2 * n_lattice * dtau))
    
    hist_writer = open(output_path +'/zcr_hist.txt','w')

    for i_mc in range(n_mc_sweeps):
        if i_mc % 100 == 0:
            print(f'#{i_mc} sweep in {n_mc_sweeps - 1}')
            
        tau_centers_ia= rilm.rilm_monte_carlo_step(n_ia,
                                   n_points,
                                   n_meas,
                                   tau_array,
                                   x_cor_sums,
                                   x2_cor_sums,
                                   x_potential_minimum,
                                   dtau)

        for i in range(0, n_ia, 2):
            if i == 0:
                zero_m = tau_centers_ia[-1] - n_lattice * dtau
            else:
                zero_m = tau_centers_ia[i-1]
                
            z_ia = min((tau_centers_ia[i+1]-tau_centers_ia[i]),
                       (tau_centers_ia[i] - zero_m))
            
            hist_writer.write(str(z_ia) + '\n')

    hist_writer.close()

    utility_custom.\
        output_correlation_functions_and_log(n_points,
                                             x_cor_sums,
                                             x2_cor_sums,
                                             n_mc_sweeps * n_meas,
                                             output_path,
                                             dtau)
