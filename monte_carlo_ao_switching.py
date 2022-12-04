import time
import numpy as np
import utility_monte_carlo as mc
import utility_custom


def free_energy_harmonic_osc(beta, x_potential_minimum):
    """Compute the free energy of the harmonic oscillator.

    Parameters
    ----------
    beta : ndarray
        Time axis.
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.

    Returns
    -------
    ndarray
        Free energy.
    """
    return -np.divide(
        np.log(2.0 * np.sinh(beta * 4.0 * x_potential_minimum / 2.0)), beta)


def monte_carlo_ao_switching(n_lattice,
                             n_equil,
                             n_mc_sweeps,
                             n_switching,
                             i_cold,
                             x_potential_minimum,
                             dtau,
                             delta_x):
    """Monte carlo simulation of the anharmonic oscillator path integral,
    where the action is adiabatically switched on from the anharmonic
    oscillator action.

    This function compute the integral of the difference between the
    full action of the anharmonic oscillator and the harmonic action using
    adiabatic switching.
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
    n_switching : int
        Number of integration steps in the adiabatic switching.
    i_cold : bool
        True for cold start, False for hot start.
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.
    dtau : float
        Lattice spacing.
    delta_x : float
        Width of Gaussian distribution for Metropolis update.

    Returns
    -------
    float, float
        Integral, total error.

    """
    # Variables for switching algorithm
    d_alpha = 1.0 / n_switching

    delta_s_alpha = np.zeros(2 * n_switching + 1)
    delta_s_alpha2 = np.zeros(2 * n_switching + 1)

    # Now the principal cycle is over the coupling constant alpha
    for i_switching in range(2 * n_switching + 1):

        x_config = mc.initialize_lattice(n_lattice,
                                         x_potential_minimum,
                                         i_cold)

        if i_switching <= n_switching:
            a_alpha = i_switching * d_alpha
        else:
            a_alpha = 2.0 - i_switching * d_alpha

        print(f'Switching #{i_switching}')

        for _ in range(n_equil):
            mc.metropolis_question_switching(x_config,
                                             x_potential_minimum,
                                             mc.potential_alpha,
                                             dtau,
                                             delta_x,
                                             a_alpha)

        for _ in range(n_mc_sweeps - n_equil):
            delta_s_alpha_temp = 0.0

            mc.metropolis_question_switching(x_config,
                                             x_potential_minimum,
                                             mc.potential_alpha,
                                             dtau,
                                             delta_x,
                                             a_alpha)

            potential_diff = mc.potential_anh_oscillator(x_config[0:-1],
                                                         x_potential_minimum)

            potential_diff -= mc.potential_0_switching(x_config[0:-1],
                                                       x_potential_minimum)

            delta_s_alpha_temp = np.sum(potential_diff) * dtau

            delta_s_alpha[i_switching] += delta_s_alpha_temp
            delta_s_alpha2[i_switching] += np.power(delta_s_alpha_temp, 2)

        # Monte Carlo End
        # Control Acceptance ratio

    delta_s_alpha_av, delta_s_alpha_err = \
        utility_custom.stat_av_var(delta_s_alpha, delta_s_alpha2,
                                   n_mc_sweeps - n_equil)

    # Integration over alpha in <deltaS>
    # Switching integral
    integral_01 = np.trapz(delta_s_alpha_av[0:(n_switching + 1)], dx=d_alpha)
    integral_10 = np.trapz(
        delta_s_alpha_av[n_switching:(2 * n_switching + 1)], dx=d_alpha)

    # Error evaluation

    # Statistical error
    integral_01_err = np.trapz(
        delta_s_alpha_err[0:(n_switching + 1)], dx=d_alpha)
    integral_10_err = np.trapz(
        delta_s_alpha_err[n_switching:(2 * n_switching + 1)], dx=d_alpha)

    propagation_error = np.sqrt(integral_01_err + integral_10_err) / 2.

    # Trapezoidal error
    trapezoidal_error = np.abs(
        (delta_s_alpha_av[n_switching] - delta_s_alpha_av[n_switching - 1]
         + delta_s_alpha_av[1] - delta_s_alpha_av[0]) /
        (d_alpha * 12 * n_switching * n_switching)
    )

    # Hysteresis error
    hysteresis_error = np.abs(integral_01 - integral_10)

    return -(integral_01 + integral_10) / 2.0, np.sqrt(
        np.power(propagation_error, 2) +
        np.power(hysteresis_error, 2) +
        np.power(trapezoidal_error, 2))


def monte_carlo_virial_theorem(n_lattice,
                               n_equil,
                               n_mc_sweeps,
                               i_cold,
                               x_potential_minimum,
                               dtau,
                               delta_x):
    """Compute the energy expectation value using the virial theorem.

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
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.
    dtau : float
        Lattice spacing.
    delta_x : float
        Width of Gaussian distribution for Metropolis update.

    Returns
    -------
    ndarray, ndarray
        Sum of the energy per each spatial point, sum of the energy squared
        per each spatial point.

    Warnings
    -------
    Energy have to be averaged over the total number of samplings.

    Notes
    -------
     We use a system of unit of measurements where h_bar=1, m=1/2 and
    lambda=1.
    """
    # x position along the tau axis
    x_config = mc.initialize_lattice(n_lattice,
                                     x_potential_minimum,
                                     i_cold)

    virial_hamiltonian = 0.0
    virial_hamiltonian2 = 0.0

    for _ in range(n_equil):
        mc.metropolis_question(x_config,
                               x_potential_minimum,
                               mc.potential_anh_oscillator,
                               dtau,
                               delta_x)

    # Rest of the MC sweeps

    for _ in range(n_mc_sweeps - n_equil):
        mc.metropolis_question(x_config,
                               x_potential_minimum,
                               mc.potential_anh_oscillator,
                               dtau,
                               delta_x)

        x_config_squared = x_config[1:-1] * x_config[1:-1]

        vir_ham = np.square(x_config_squared
                            - x_potential_minimum * x_potential_minimum)

        vir_ham += 2 * x_config_squared \
                   * (x_config_squared - x_potential_minimum
                       * x_potential_minimum)

        vir_ham_2 = np.square(vir_ham)

        virial_hamiltonian += np.sum(vir_ham)

        virial_hamiltonian2 += np.sum(vir_ham_2)

    return virial_hamiltonian, virial_hamiltonian2


def free_energy_anharm(n_beta,
                       beta_max,
                       n_equil,
                       n_mc_sweeps,
                       n_switching,
                       i_cold,
                       x_potential_minimum=1.4,
                       dtau=0.05,
                       delta_x=0.5):
    """Compute the free energy of the anharmonic oscillator using both
    adiabatic switching and the virial theorem.

    All results are saved into files.

    Parameters
    ----------
    n_beta : int
        Number of point at which compute the free energy.
    beta_max : int
        Lattice maximum length.
    n_equil : int
        Number of equilibration Monte Carlo sweeps.
    n_mc_sweeps : int
        Number of Monte Carlo sweeps.
    n_switching : int
        Number of integration steps in the adiabatic switching.
    i_cold : bool
        True for cold start, False for hot start.
    x_potential_minimum : float, default=1.4
        Position of the minimum(a) of the anharmonic potential.
    dtau : float, default=0.05
        Lattice spacing.
    delta_x : float, default=0.5
        Width of Gaussian distribution for Metropolis update.

    Returns
    -------
    int
        Return 0 if n_mc_sweeps < n_equil, else return 1
    """
    # Control output filepath
    output_path = './output_data/output_monte_carlo_switching'
    utility_custom.output_control(output_path)

    # Free Helmoltz energy for the anharmonic oscillator
    beta_array = np.linspace(1.0, beta_max, n_beta, False)
    temperature_array = 1.0 / beta_array

    free_energy = np.zeros(n_beta, float)
    free_energy_err = np.zeros(n_beta, float)

    virial_hamiltonian_av = np.zeros(n_beta, float)
    virial_hamiltonian_err = np.zeros(n_beta, float)

    for i_beta in range(n_beta):
        start = time.time()

        n_lattice = int(beta_array[i_beta] / dtau)

        if n_mc_sweeps < n_equil:
            print("too few Monte Carlo sweeps/ N_equilib > N_Monte_Carlo")
            return 0
        print(
            f'Adiabatic switching for beta = {n_lattice * dtau},'
            f' n_lattice = {n_lattice}')

        free_energy[i_beta], free_energy_err[i_beta] = \
            monte_carlo_ao_switching(n_lattice,  # n_lattice
                                     n_equil,  # n_equil
                                     n_mc_sweeps,  # n_mc_sweeps-
                                     n_switching,
                                     i_cold,
                                     x_potential_minimum,
                                     dtau,
                                     delta_x)

        free_energy[i_beta] /= beta_array[i_beta]
        free_energy_err[i_beta] /= beta_array[i_beta]
        free_energy[i_beta] += free_energy_harmonic_osc(beta_array[i_beta],
                                                        x_potential_minimum)

        end = time.time()
        print(f'ELapsed = {end - start}')

        start = time.time()

        print(
            f'Monte Carlo virial theorem approach for n_lattice = {n_lattice}:')

        virial_hamiltonian, virial_hamiltonian2 = \
            monte_carlo_virial_theorem(n_lattice,
                                       n_equil,
                                       n_mc_sweeps,
                                       i_cold,
                                       x_potential_minimum,
                                       dtau,
                                       delta_x)

        virial_hamiltonian_av[i_beta], virial_hamiltonian_err[i_beta] = \
            utility_custom.stat_av_var(virial_hamiltonian,
                                       virial_hamiltonian2,
                                       n_lattice * (n_mc_sweeps - n_equil))

        end = time.time()
        print(f'ELapsed = {end - start}')

    with open(output_path + '/free_energy_mc.txt', 'w',
              encoding='utf8') as f_writer:
        np.savetxt(f_writer, free_energy)
    with open(output_path + '/free_energy_vir.txt', 'w',
              encoding='utf8') as f_writer:
        np.savetxt(f_writer, -virial_hamiltonian_av)
    with open(output_path + '/free_energy_mc_err.txt', 'w',
              encoding='utf8') as f_writer:
        np.savetxt(f_writer, free_energy_err)
    with open(output_path + '/free_energy_vir_err.txt', 'w',
              encoding='utf8') as f_writer:
        np.savetxt(f_writer, virial_hamiltonian_err)
    with open(output_path + '/temperature.txt', 'w',
              encoding='utf8') as f_writer:
        np.savetxt(f_writer, temperature_array)

    with open(output_path + '/free_energy_harm.txt', 'w',
              encoding='utf8') as f_writer:
        np.savetxt(f_writer, free_energy_harmonic_osc(beta_array,
                                                      x_potential_minimum))

    return 1
