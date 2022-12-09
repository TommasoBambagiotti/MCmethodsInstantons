import time
import numpy as np
import utility_monte_carlo as mc
import utility_custom


def zero_crossing_cooling_density(n_lattice,
                                  n_equil,
                                  n_mc_sweeps,
                                  i_cold,
                                  n_sweeps_btw_cooling,
                                  n_cooling_sweeps,
                                  x_potential_minimum=1.4,
                                  dtau=0.05,
                                  delta_x=0.5):
    """Determine the zero crossing histogram for cooled configurations
    of instantons and anti-instantons.

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

    Notes
    -------
    We use a system of unit of measurements where h_bar=1, m=1/2 and
    lambda=1.
    """
    start = time.time()
    if n_mc_sweeps < n_equil:
        print("too few Monte Carlo sweeps/ N_equilib > N_Monte_Carlo")
        return 0

    # Control output filepath
    output_path = './output_data/output_cooled_monte_carlo/zero_crossing'
    utility_custom.output_control(output_path)

    # number of cooling procedures
    n_cooling = 0

    # x position along the tau axis
    x_config = mc.initialize_lattice(n_lattice, i_cold)
    count = 0
    # zero crossing density fistribution
    hist_writer = open(output_path + '/zcr_cooling.txt', 'w')

    n_data_point = 1000

    array_ia = np.zeros(n_data_point, float)
    array_int = np.zeros(n_data_point, float)

    # Equilibration cycle
    for _ in range(n_equil):
        mc.metropolis_question(x_config,
                               x_potential_minimum,
                               dtau,
                               delta_x)

    # control = open('control.txt', 'w', encoding='utf8')
    # Rest of the MC sweeps
    for i_mc in range(n_mc_sweeps - n_equil):

        if count > 600000:
            break

        mc.metropolis_question(x_config,
                               x_potential_minimum,
                               dtau,
                               delta_x)

        # COOLING

        if (i_mc % n_sweeps_btw_cooling) == 0:

            # print(f'cooling #{n_cooling} of {(n_mc_sweeps - n_equil) / n_sweeps_btw_cooling}\n'
            #       f'in configuration #{i_mc}')

            x_cold_config = np.copy(x_config)
            n_cooling += 1
            n_instantons, n_anti_instantons = 0, 0

            for _ in range(n_cooling_sweeps):
                mc.configuration_cooling(x_cold_config,
                                         x_potential_minimum,
                                         dtau,
                                         delta_x)

                # Find instantons and antiinstantons after the cooling procedure
            n_instantons, n_anti_instantons, pos_roots, neg_roots = \
                mc.find_instantons(x_cold_config,
                                   dtau)

            # inst-anti inst zero crossing
            z_ia = 0

            # total zero crossings
            if n_instantons == n_anti_instantons \
                    and n_instantons > 0 \
                    and n_instantons == len(pos_roots) \
                    and n_anti_instantons == len(neg_roots):

                if pos_roots[0] < neg_roots[0]:
                    for i in range(n_instantons):
                        if i == 0:
                            zero_m = neg_roots[-1] - n_lattice * dtau
                        else:
                            zero_m = neg_roots[i - 1]

                        z_ia = np.minimum(np.abs(neg_roots[i] - pos_roots[i]),
                                          np.abs(pos_roots[i] - zero_m))
                        count += 1
                        hist_writer.write(str(z_ia) + '\n')

                elif pos_roots[0] > neg_roots[0]:
                    for i in range(n_instantons):
                        if i == 0:
                            zero_p = pos_roots[-1] - n_lattice * dtau
                        else:
                            zero_p = pos_roots[i - 1]

                        z_ia = np.minimum(np.abs(pos_roots[i] - neg_roots[i]),
                                          np.abs(neg_roots[i] - zero_p))

                        count += 1
                        hist_writer.write(str(z_ia) + '\n')

                else:
                    continue

    array_int /= 4 / 3 * np.power(x_potential_minimum, 3)
    array_int -= 2.

    np.savetxt(output_path + '/array_ia.txt', array_ia)
    np.savetxt(output_path + '/array_int.txt', array_int)

    hist_writer.close()
    end = time.time()
    print(f'Elapsed time: {end - start}')

    return 1
