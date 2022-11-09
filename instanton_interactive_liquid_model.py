'''
Instanton interactive
Liquid model 
'''
import numpy as np

import utility_custom
import utility_monte_carlo as mc
import utility_rilm as rilm
import input_parameters as ip


def inst_int_liquid_model(n_lattice,  # size of the grid
                          n_mc_sweeps,  # monte carlo sweeps
                          n_points,  #
                          n_meas,
                          tau_core,
                          action_core,
                          dx_update):
    
    # Control output filepath
    output_path = './output_data/output_iilm/iilm'
    utility_custom.output_control(output_path)

    # Eucliadian time
    tau_array = np.linspace(0.0, n_lattice * ip.dtau, n_lattice, False)

    # Loop 2 expectation density
    loop_2 = 8 * pow(ip.x_potential_minimum, 5 / 2) \
        * pow(2 / np.pi, 1/2) * np.exp(-ip.action_0 - 71 / (72 * ip.action_0))
        
    n_ia = int(np.rint(loop_2 * n_lattice * ip.dtau))

    # Center of instantons and anti instantons
    tau_centers_ia= rilm.centers_setup(tau_array, n_ia, tau_array.size)
    tau_centers_ia_temp = np.zeros(n_ia, float)

    # Ansatz sum of indipendent instantons
    x_config = rilm.ansatz_instanton_conf(tau_centers_ia, tau_array)
    x_config_temp = np.zeros(n_lattice + 1, float)

    # Correlation functions
    x_cor_sums = np.zeros((3, n_points))
    x2_cor_sums = np.zeros((3, n_points))

    # Histogram zero crossing density
    hist_writer = open(output_path +'/zcr_int.txt','w')
    
    for i_mc in range(n_mc_sweeps):
        action_old = mc.return_action(x_config)
        action_old += rilm.hard_core_action(n_lattice,
                                            tau_centers_ia,
                                            tau_core,
                                            action_core)
        
        
        if i_mc % 100 == 0:
            print(f'#{i_mc} sweep in {n_mc_sweeps - 1}')
        
        # for tau_ia, tau_ia_new in zip(np.nditer(tau_centers_ia),\
        #     np.nditer(tau_centers_ia_temp)):
                
        for i in range(tau_centers_ia.size):    
            tau_centers_ia_temp[i] =\
                tau_centers_ia[i]\
                + (np.random.uniform(0.0,1.0) - 0.5)* dx_update
            
            if tau_centers_ia_temp[i] > n_lattice*ip.dtau:
                tau_centers_ia_temp[i] -= n_lattice*ip.dtau
            elif tau_centers_ia_temp[i]< 0.0:
                tau_centers_ia_temp[i] += n_lattice*ip.dtau
            
        x_config_temp = rilm.ansatz_instanton_conf(tau_centers_ia_temp,
                                                   tau_array)
        
        action_new = mc.return_action(x_config_temp)
        action_new += rilm.hard_core_action(n_lattice,
                                            tau_centers_ia_temp,
                                            tau_core,
                                            action_core)
        
        delta_action = action_new - action_old
        if np.exp(-delta_action) > np.random.uniform(0., 1.):
            tau_centers_ia = np.copy(tau_centers_ia_temp)
            x_config = np.copy(x_config_temp)
        
        for i in range(0, tau_centers_ia.size, 2):
            if i == 0:
                zero_m = tau_centers_ia[-1] - n_lattice * ip.dtau
            else:
                zero_m = tau_centers_ia[i-1]
                
            z_ia = min((tau_centers_ia[i+1]-tau_centers_ia[i]),
                       (tau_centers_ia[i] - zero_m))
            
            hist_writer.write(str(z_ia) + '\n')
        
        
            