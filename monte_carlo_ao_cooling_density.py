'''
Density of instantons
and anti-instantons in
cooling process
'''
import time
import numpy as np

import utility_monte_carlo as mc
import utility_custom


def cooled_monte_carlo_density(n_lattice,
                               n_equil,
                               n_mc_sweeps,
                               i_cold,
                               n_sweeps_btw_cooling,
                               n_cooling_sweeps,
                               n_minima,
                               first_minimum=1.4,
                               dtau=0.05,
                               delta_x=0.5):
    """Compute total number of instantons and anti-instantons and the action
    density as function of the number of cooling sweeps for each cooling.
    The computation is done for different value of the potential minima.

    We use a system of unit of measurements where h_bar=1, m=1/2 and
    lambda=1.

    Parameters
    ----------
    n_lattice : int
        Number of lattice point in euclidean time.
    n_equil : int
        Number of equilibration Monte Carlo sweeps.
    n_mc_sweeps : int
        Number of Monte Carlo sweeps.
    i_cold : bool
        True for cold start, False for hot start.
    n_sweeps_btw_cooling : int
        Number of MC sweeps between two consecutive coolings.
    n_cooling_sweeps : int
        Number of MC sweeps for each cooling.
    n_minima : int
        Number of potential minima.
    first_minimum : float
        First potential minimum.
    dtau : float, default=0.05
        Lattice spacing.
    delta_x : float, default=0.5
        Width of Gaussian distribution for Metropolis update.

    Returns
    ----------
    int
        Return 0 if n_mc_sweeps < n_equil, else return 1
    """

    start = time.time()
    end = np.zeros(n_minima + 1)
    end[0] = start
    if n_mc_sweeps < n_equil:
        print("too few Monte Carlo sweeps/ N_equilib > N_Monte_Carlo")
        return 0

    # Set and control output filepath
    output_path = './output_data/output_cooled_monte_carlo'
    utility_custom.output_control(output_path)

    potential_minima = []

    # Main cycle over potential minima
    for i_minimum in range(n_minima):
        print(f'New monte carlo for minimum ='
              f' {first_minimum + 0.1 * i_minimum}')

        # number of total cooling processes
        n_cooling = 0

        x_potential_minimum = first_minimum + 0.1 * i_minimum

        potential_minima.append('%.2f' % x_potential_minimum)

        # x position along the tau axis
        x_config = mc.initialize_lattice(n_lattice,
                                         x_potential_minimum,
                                         i_cold)

        # number of instantons
        n_total_instantons_sum = np.zeros(n_cooling_sweeps, int)
        n2_total_instantons_sum = np.zeros(n_cooling_sweeps, int)

        # instanton action density
        action_cooling = np.zeros(n_cooling_sweeps, float)
        action2_cooling = np.zeros(n_cooling_sweeps, float)

        # Equilibration sweeps
        # tau_array = np.linspace(0.0, dtau*n_lattice, n_lattice)
        for _ in range(n_equil):
            mc.metropolis_question(x_config,
                                   x_potential_minimum,
                                   mc.potential_anh_oscillator,
                                   dtau,
                                   delta_x)
        # Rest of the MC sweeps
        for i_mc in range(n_mc_sweeps - n_equil):
            mc.metropolis_question(x_config,
                                   x_potential_minimum,
                                   mc.potential_anh_oscillator,
                                   dtau,
                                   delta_x)

            if i_mc % int((n_mc_sweeps - n_equil) / 10) == 0:
                print(f'conf: {i_mc}\n'
                      + f'Action: {mc.return_action(x_config, x_potential_minimum, dtau)}')

            # COOLING
            if (i_mc % n_sweeps_btw_cooling) == 0:

                n_cooling += 1
                x_cold_config = np.copy(x_config)
                n_instantons, n_anti_instantons = 0, 0

                for i_cooling in range(n_cooling_sweeps):
                    mc.configuration_cooling(x_cold_config,
                                             x_potential_minimum,
                                             mc.potential_anh_oscillator,
                                             dtau,
                                             delta_x)

                    n_instantons, n_anti_instantons, _, _ = \
                        mc.find_instantons(x_cold_config,
                                           dtau)

                    n_total_instantons_sum[i_cooling] += (
                        n_instantons + n_anti_instantons)
                    n2_total_instantons_sum[i_cooling] += np.square(
                        n_instantons + n_anti_instantons)

                    action_temp = mc.return_action(x_cold_config,
                                                   x_potential_minimum, dtau)

                    action_cooling[i_cooling] += action_temp
                    action2_cooling[i_cooling] += np.square(action_temp)

        # Evaluate averages and errors
        action_av, action_err = utility_custom.stat_av_var(action_cooling,
                                                           action2_cooling,
                                                           n_cooling)

        n_total, n_total_err = \
            utility_custom.stat_av_var(n_total_instantons_sum,
                                       n2_total_instantons_sum,
                                       n_cooling)

        np.savetxt(output_path + f'/n_total_{i_minimum + 1}.txt', n_total)

        # Density and action density
        action_av = np.divide(action_av, n_total)
        action_err = np.divide(action_err, n_total)

        n_total /= (n_lattice * dtau)
        n_total_err /= (n_lattice * dtau)

        with open(output_path + f'/n_instantons_{i_minimum + 1}.txt', 'w',
                  encoding='utf-8') as n_inst_writer:
            np.savetxt(n_inst_writer, n_total)

        with open(output_path + f'/n_instantons_{i_minimum + 1}_err.txt', 'w',
                  encoding='utf-8') as n_inst_writer:
            np.savetxt(n_inst_writer, n_total_err)

        with open(output_path + f'/action_{i_minimum + 1}.txt', 'w',
                  encoding='utf-8') as act_writer:
            np.savetxt(act_writer, action_av)

        with open(output_path + f'/action_err_{i_minimum + 1}.txt', 'w',
                  encoding='utf-8') as act_writer:
            np.savetxt(act_writer, action_err)

        end[i_minimum + 1] = time.time()
        print(f'Time elapsed\nf = {potential_minima[i_minimum]}'
              + f' elapsed {end[i_minimum + 1] - end[i_minimum]}')

    with open(output_path + '/n_cooling.txt', 'w',
              encoding='utf-8') as n_inst_writer:
        np.savetxt(n_inst_writer, np.linspace(
            1, n_cooling_sweeps + 1, n_cooling_sweeps, False))

    with open(output_path + '/potential_minima.txt', 'w',
              encoding='utf-8') as pot_writer:
        for potential in potential_minima:
            pot_writer.write(potential + '\n')

    return 1