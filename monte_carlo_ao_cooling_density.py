
import numpy as np

import utility_monte_carlo as mc
import utility_custom
import input_parameters as ip


def cooled_monte_carlo(
        n_lattice,
        n_equil,
        n_mc_sweeps,
        i_cold,
        n_sweeps_btw_cooling,
        n_cooling_sweeps,
        n_minima):


    if n_mc_sweeps < n_equil:
        print("too few Monte Carlo sweeps/ N_equilib > N_Monte_Carlo")
        #return 0

    # Control output filepath
    output_path = './output_data/output_cooled_monte_carlo'
    utility_custom.output_control(output_path)

    potential_minima = np.zeros((n_minima), float)

    for i_minimum in range(n_minima):
        print(f'New monte carlo for eta = {1.4 + 0.1 * i_minimum}')

        #contatore probabilmente inutile
        n_cooling = 0

        ip.x_potential_minimum = 1.6 + 0.1 * i_minimum

        ip.print_minimum()

        potential_minima[i_minimum] = ip.x_potential_minimum

        # x position along the tau axis
        x_config = mc.initialize_lattice(n_lattice, i_cold)

        #number of instantons

        n_total_instantons_sum = np.zeros((n_cooling_sweeps), int)
        n2_total_instantons_sum = np.zeros((n_cooling_sweeps), int)

        action_cooling = np.zeros((n_cooling_sweeps), float)

        # Monte Carlo sweeps: Principal cycle

        # Equilibration cycle

        for i_equil in range(n_equil):
            mc.metropolis_question(x_config)

        np.savetxt(output_path + f'/config_equil_{i_minimum + 1}.txt', x_config )

        # Rest of the MC sweeps

        for i_mc in range(n_mc_sweeps - n_equil):
            mc.metropolis_question(x_config)

            # COOLING

            if (i_mc % n_sweeps_btw_cooling) == 0:

                print(f'cooling #{n_cooling} of {(n_mc_sweeps - n_equil) / n_sweeps_btw_cooling}\n'
                          f'in configuration #{i_mc}')

                x_cold_config = np.copy(x_config)
                n_cooling += 1

                for i_cooling in range(n_cooling_sweeps):
                    mc.configuration_cooling(x_cold_config,
                                             ip.x_potential_minimum)

                    # Find instantons and antiinstantons
                    n_instantons, n_anti_instantons = mc.find_instantons(x_cold_config,
                                                                             n_lattice,
                                                                             ip.dtau)
                    #pos_instantons, pos_anti_instantons 

                    print(n_instantons)
                    n_total_instantons_sum[i_cooling] += (n_instantons + n_anti_instantons)
                    n2_total_instantons_sum[i_cooling] += pow((n_instantons + n_anti_instantons), 2)

                    # action for each cooling configuration
                    action_cooling[i_cooling] += mc.return_action(x_cold_config)

        # Evaluate averages and errors

        action_cooling /= n_cooling

        n_total, n_total_err = mc.stat_av_var(n_total_instantons_sum,
                                              n2_total_instantons_sum,
                                              n_cooling)

        np.savetxt(output_path + '/tau_array_conf.txt', np.linspace(0.0, n_lattice * ip.dtau, n_lattice))
        np.savetxt(output_path + '/configuration.txt', x_config[0:-1])
        np.savetxt(output_path + '/configuration_cooled.txt', x_cold_config[0:-1])

        np.savetxt(output_path + f'/n_total_{i_minimum + 1}.txt', n_total)

        with open (output_path + f'/action_{i_minimum + 1}_original.txt', 'w',
                   encoding='utf-8') as act_writer:
            np.savetxt(act_writer, action_cooling)

        action_cooling = np.divide(action_cooling, n_total)


        n_total /= (n_lattice * ip.dtau * 2.0)
        n_total_err /= (n_lattice * ip.dtau * 2.0)

        

        with open (output_path + f'/n_instantons_{i_minimum + 1}.txt', 'w',
                  encoding='utf-8') as n_inst_writer:
            np.savetxt(n_inst_writer, n_total)

        with open (output_path + f'/n_instantons_{i_minimum + 1}_err.txt', 'w',
                  encoding='utf-8') as n_inst_writer:
            np.savetxt(n_inst_writer, n_total_err)

        with open (output_path + f'/action_{i_minimum + 1}.txt', 'w',
                   encoding='utf-8') as act_writer:
            np.savetxt(act_writer, action_cooling)

    with open (output_path + '/n_cooling.txt', 'w',
               encoding='utf-8') as n_inst_writer:
        np.savetxt(n_inst_writer, np.linspace(1, n_cooling_sweeps + 1, n_cooling_sweeps, False))

    # with open (output_path + '/potential_minima.txt', 'w',
    #         encoding='utf-8') as n_inst_writer:
    #     np.savetxt(n_inst_writer, potential_minima)
