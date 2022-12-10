'''
Calculation of correlation functions
using the cooling method to estrapolte
instantons - anti instantons configurations
'''

import numpy as np
import utility_monte_carlo as mc
import utility_custom


def cooled_monte_carlo(
        n_lattice,
        n_equil,
        n_mc_sweeps,
        n_points,
        n_meas,
        i_cold,
        n_sweeps_btw_cooling,
        n_cooling_sweeps,
        x_potential_minimum=1.4,
        dtau=0.05,
        delta_x=0.5):
    """Compute spatial correlation functions for the anharmonic oscil-
    lator using Monte Carlo simulations for cooled configurations.

    Correlation functions are computed for cooled configurations. Cooling
    is a method to extract tunneling events removing short distance fluc-
    tuations from configurations generated using the Metropolis algorithm.
    In this method we accept only Metropolis update that lower the action.
    Cooling is performed every n_sweeps_btw_cooling, and it is iterated
    n_cooling_sweeps times. Finally, results are saved into files.
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
    n_points : int
        Number of points on which correlation functions are computed.
    n_meas : int
        Number of measurement of correlation functions in a MC sweep.
    i_cold : bool
        True for cold start, False for hot start.
    n_sweeps_btw_cooling : int
        Number of Monte Carlo sweeps between two successive cooling.
    n_cooling_sweeps : int
        Total number of cooling sweeps to perform.
    x_potential_minimum : float, default=1.4
        Position of the minimum(a) of the anharmonic potential.
    dtau : float, default=0.05
        Lattice spacing.
    delta_x : float, default=0.5
        Width of Gaussian distribution for Metropolis update.

    Returns
    -------
    int
        Return 0 if n_mc_sweeps < n_equil, else return 1.
    """
    # number of total cooling processes
    n_cooling = 0

    if n_mc_sweeps < n_equil:
        print("too few Monte Carlo sweeps/ N_equilib > N_Monte_Carlo")
        return 0

    # Control output filepath
    output_path = './output_data/output_cooled_monte_carlo'
    utility_custom.output_control(output_path)

    # Correlation functions
    x_cold_cor_sums = np.zeros((3, n_points))
    x2_cold_cor_sums = np.zeros((3, n_points))

    # x position along the tau axis
    x_config = mc.initialize_lattice(n_lattice, i_cold)

    # Monte Carlo sweeps: Principal cycle

    # Equilibration cycle
    for i_equil in range(n_equil):
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
        # save middle config
        if i_mc % int((n_mc_sweeps-n_equil)/2) == 0:
            with open(output_path+'/x2_config.txt','w') as f_writer:
                np.savetxt(f_writer, x_config)

        # Print action
        if i_mc % int((n_mc_sweeps - n_equil)/10) == 0:
            print(f'conf: {i_mc}\n'
                  +f'Action: {mc.return_action(x_config, x_potential_minimum, dtau)}')

        # COOLING

        if (i_mc % n_sweeps_btw_cooling) == 0:


            # expected number of cooled configuration = n_conf/n_sweeps_btw_cooling
            # print    f'in configuration #{i_mc}')

            x_cold_config = np.copy(x_config)
            n_cooling += 1

            for i_cooling in range(n_cooling_sweeps):
                mc.configuration_cooling(x_cold_config,
                                         x_potential_minimum,
                                         mc.potential_anh_oscillator,
                                         dtau,
                                         delta_x)
            # save mid config cooled
            if i_mc % int((n_mc_sweeps - n_equil) / 2) == 0:
                with open(output_path + '/x1_config.txt', 'w') as f_writer:
                    np.savetxt(f_writer, x_cold_config)
            
            # Compute correlation functions for the cooled configuration
            utility_custom.correlation_measurments(n_lattice, n_meas, n_points,
                                                   x_cold_config, 
                                                   x_cold_cor_sums,
                                                   x2_cold_cor_sums)

        # End of Montecarlo simulation

    # Correlation functions

    utility_custom.\
        output_correlation_functions_and_log(n_points,
                                             x_cold_cor_sums,
                                             x2_cold_cor_sums,
                                             n_cooling * n_meas,
                                             output_path)

    return 1
