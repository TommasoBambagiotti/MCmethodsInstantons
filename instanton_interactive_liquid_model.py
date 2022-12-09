'''
Instanton interactive Liquid model
'''
import numpy as np
import time
import utility_custom
import utility_monte_carlo as mc
import utility_rilm as rilm


def inst_int_liquid_model(n_lattice,
                          n_mc_sweeps,
                          n_points,
                          n_meas,
                          tau_core,
                          action_core,
                          dx_update,
                          x_potential_minimum=1.4,
                          dtau=0.05):
    """Compute correlation functions for the anharmonic oscillator
    using an interactin ensemble of instantons.

    Correlation functions are computed using MC for the discretized action
    of the anharmonic oscillator with a neighbor repulsive core interaction.
    Final configuration of instanton/anti-instanton centers, correlation
    functions and system path are saved into files.

    Parameters
    ----------
    n_lattice : int
        Number of lattice point in euclidean time.
    n_equil : int
        Number of equilibration Monte Carlo sweeps.
    n_mc_sweeps : int
        Number of Monte Carlo sweeps.
    n_meas : int
        Number of measurement of correlation functions in a MC sweep.
    tau_core : float
        Range of hard core interaction.
    action_core : float
        Strength of hard core interaction.
    dx_update : float
        Average position update.
    x_potential_minimum : float, default=1.4
        Position of the minimum(a) of the anharmonic potential.
    dtau : float, default=0.05
        Lattice spacing.

    Returns
    ----------
    None
    """
    # Control output filepath
    output_path = './output_data/output_iilm/iilm'
    utility_custom.output_control(output_path)

    # Eucliadian time
    tau_array = np.linspace(0.0, n_lattice * dtau, n_lattice, False)

    # Loop 2 expectation density
    action_0 = 4 / 3 * np.power(x_potential_minimum, 3)

    # loop_2 = 8 * np.power(x_potential_minimum, 5 / 2) \
    #          * np.power(2 / np.pi, 1 / 2) * np.exp(
    #     -action_0 - 71 / (72 * action_0))
    loop_2 = mc.two_loop_density(x_potential_minimum)

    n_ia = int(np.rint(loop_2 * n_lattice * dtau))

    # Center of instantons and anti instantons
    tau_centers_ia = rilm.centers_setup(n_ia, tau_array.size)

    # Correlation functions
    x_cor_sums = np.zeros((3, n_points))
    x2_cor_sums = np.zeros((3, n_points))

    # Print evolution in conf of tau centers
    n_conf = 3000
    tau_centers_evolution = np.zeros((n_conf, n_ia), float)

    # Initial centers
    tau_centers_evolution[0] = tau_centers_ia

    hist_writer = open(output_path + '/zcr_hist.txt', 'w')

    start = time.time()

    for i_mc in range(n_mc_sweeps):
        tau_centers_ia_store = np.copy(tau_centers_ia)

        x_config = rilm.ansatz_instanton_conf(tau_centers_ia,
                                              tau_array,
                                              x_potential_minimum)

        action_old = mc.return_action(x_config,
                                      x_potential_minimum,
                                      dtau)

        action_old += rilm.hard_core_action(n_lattice,
                                            tau_centers_ia,
                                            tau_core,
                                            action_core,
                                            action_0,
                                            dtau)

        # if i_mc % 100 == 0:
        #     print(f'#{i_mc} sweep in {n_mc_sweeps - 1}')

        for i in range(tau_centers_ia.size):
            tau_centers_ia[i] += \
                (np.random.uniform(0.0, 1.0) - 0.5) * dx_update

            if tau_centers_ia[i] > n_lattice * dtau:
                tau_centers_ia[i] -= n_lattice * dtau
            elif tau_centers_ia[i] < 0.0:
                tau_centers_ia[i] += n_lattice * dtau

            x_config = rilm.ansatz_instanton_conf(tau_centers_ia,
                                                  tau_array,
                                                  x_potential_minimum)

            action_new = mc.return_action(x_config,
                                          x_potential_minimum,
                                          dtau)

            action_new += rilm.hard_core_action(n_lattice,
                                                tau_centers_ia,
                                                tau_core,
                                                action_core,
                                                action_0,
                                                dtau)

            delta_action = action_new - action_old
            if np.exp(-delta_action) > np.random.uniform(0., 1.):
                action_old = action_new
            else:
                tau_centers_ia[i] = tau_centers_ia_store[i]

        if (i_mc + 1) < n_conf:
            tau_centers_evolution[i_mc + 1] = tau_centers_ia

        for i in range(0, n_ia, 2):
            if i == 0:
                zero_m = tau_centers_ia[-1] - n_lattice * dtau
            else:
                zero_m = tau_centers_ia[i - 1]

            z_ia = min((tau_centers_ia[i + 1] - tau_centers_ia[i]),
                       (tau_centers_ia[i] - zero_m))

            hist_writer.write(str(z_ia) + '\n')

        x_config = rilm.ansatz_instanton_conf(tau_centers_ia,
                                              tau_array,
                                              x_potential_minimum)

        utility_custom.correlation_measurments(n_lattice,
                                               n_meas,
                                               n_points,
                                               x_config,
                                               x_cor_sums,
                                               x2_cor_sums)

    utility_custom.output_correlation_functions_and_log(n_points,
                                                        x_cor_sums,
                                                        x2_cor_sums,
                                                        n_meas * n_mc_sweeps,
                                                        output_path)

    np.savetxt(output_path + '/n_conf.txt',
               np.linspace(0, n_conf, n_conf, False))

    for n in range(n_ia):
        np.savetxt(output_path + f'/center_{n + 1}.txt',
                   tau_centers_evolution[:, n])

    end = time.time()

    print(f'Elapsed time {end - start}')
