import numpy as np
import utility_custom
import utility_monte_carlo as mc


def monte_carlo_ao(n_lattice,
                   n_equil,
                   n_mc_sweeps,
                   n_points,
                   n_meas,
                   i_cold,
                   x_potential_minimum=1.4,
                   dtau=0.05,
                   delta_x=0.5):

    """Compute spatial correlation functions for the anharmonic oscillator
    with Monte Carlo simulations.

    This function compute euclidean correlation functions for the anharmon-
    ic oscillator using path integral Monte Carlo simulations. At every
    Monte Carlo sweep the system path is determined using the Metropolis-
    Hastings algorithm; finally results are saved into files.
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
    x_potential_minimum : float, default=1.4
        Position of the minimum(a) of the anharmonic potential.
    dtau : float, default=0.05
        Lattice spacing.
    delta_x : float, default=0.5
        Width of Gaussian distribution for Metropolis update.

    Returns
    ----------
    int
        Return 0 if n_mc_sweeps < n_equil, else return 1
    """

    if n_mc_sweeps < n_equil:
        print("too few Monte Carlo sweeps/ N_equilib > N_Monte_Carlo")
        return 0

    # Control output filepath
    output_path = './output_data/output_monte_carlo'
    utility_custom.output_control(output_path)

    # Correlation functions
    x_cor_sums = np.zeros((3, n_points))
    x2_cor_sums = np.zeros((3, n_points))

    # x position along the tau axis
    x_config = mc.initialize_lattice(n_lattice,
                                     x_potential_minimum,
                                     i_cold)

    # Equilibration sweeps
    for _ in range(n_equil):
        mc.metropolis_question(x_config,
                               x_potential_minimum,
                               mc.potential_anh_oscillator,
                               dtau,
                               delta_x)

    # Main MC sweeps
    for i_mc in range(n_mc_sweeps - n_equil):
        if i_mc % 100 == 0:
            print(f'{i_mc} in {n_mc_sweeps - n_equil}')

        mc.metropolis_question(x_config,
                               x_potential_minimum,
                               mc.potential_anh_oscillator,
                               dtau,
                               delta_x)

        utility_custom.correlation_measurments(n_lattice, n_meas, n_points,
                                               x_config, x_cor_sums,
                                               x2_cor_sums)

    # Evaluate correlation functions
    utility_custom. \
        output_correlation_functions_and_log(n_points,
                                             x_cor_sums,
                                             x2_cor_sums,
                                             (n_mc_sweeps - n_equil) * n_meas,
                                             output_path)

    return 1
