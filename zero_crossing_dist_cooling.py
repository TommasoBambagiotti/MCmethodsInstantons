"""
Hystogram of the zero-crossing of
instantons and anti-instantons configurations
after N-cooling sweeps
"""

import numpy as np

import utility_monte_carlo as mc
import utility_custom
import input_parameters as ip


def zero_crossing_cooling_density(n_lattice,
                                  n_equil,
                                  n_mc_sweeps,
                                  i_cold,
                                  n_sweeps_btw_cooling,
                                  n_cooling_sweeps,
                                  ):

    if n_mc_sweeps < n_equil:
        print("too few Monte Carlo sweeps/ N_equilib > N_Monte_Carlo")
        return 1

    # Control output filepath
    output_path = './output_data/output_cooled_monte_carlo/zero_crossing'
    utility_custom.output_control(output_path)

    # number of cooling procedures
    n_cooling = 0

    # x position along the tau axis
    x_config = mc.initialize_lattice(n_lattice, i_cold)

    # zero crossing density
    hist_writer = open(output_path + '/zcr_cooling.txt', 'a')

    # Equilibration cycle

    for _ in range(n_equil):
        mc.metropolis_question(x_config)

    # Rest of the MC sweeps

    for i_mc in range(n_mc_sweeps - n_equil):
        mc.metropolis_question(x_config)

        # COOLING

        if (i_mc % n_sweeps_btw_cooling) == 0:

            print(f'cooling #{n_cooling} of {(n_mc_sweeps - n_equil) / n_sweeps_btw_cooling}\n'
                  f'in configuration #{i_mc}')

            x_cold_config = np.copy(x_config)
            n_cooling += 1
            n_instantons, n_anti_instantons = 0, 0

            for _ in range(n_cooling_sweeps):
                mc.configuration_cooling(x_cold_config,
                                         ip.x_potential_minimum)

                # Find instantons and antiinstantons after the cooling procedure
            n_instantons, n_anti_instantons, pos_roots, neg_roots =\
                mc.find_instantons(x_cold_config,
                                   n_lattice,
                                   ip.dtau)

            # total zero crossings
            if n_instantons != 0:
                if pos_roots[0] < neg_roots[0]:
                    
                    for i in range(n_instantons):
                        
                        if i == 0:
                            zero_m = neg_roots[-1] - n_lattice * ip.dtau
                        else:
                            zero_m = neg_roots[i-1]
                        
                        z_ia = min(np.abs(neg_roots[i] - pos_roots[i]),
                                       np.abs(pos_roots[i] - zero_m))
        
                        hist_writer.write(str(z_ia) + '\n')
                        
                elif pos_roots[0] > neg_roots[0]:
                    for i in range(n_instantons):
                        if i == 0:
                            zero_p = pos_roots[-1] - n_lattice * ip.dtau
                        else:
                            zero_p = pos_roots[i-1]
                        
                        z_ia = min(np.abs(pos_roots[i] - neg_roots[i]),
                                       np.abs(neg_roots[i] - zero_p))
                        
                        hist_writer.write(str(z_ia) + '\n')
                        
                else:
                    continue
                
    hist_writer.close()

    return 0
