import numpy as np
from numba import jit
import utility_custom


#@jit(nopython=True)
def centers_setup(n_ia, n_lattice, dtau=0.05):
    """Generate a random ensemble of instantons/anti-instantons.

    Parameters
    ----------
    tau_array :
    n_ia : int
        Number of instanton/anti-instanton
    n_lattice : int
        Number of lattice points.
    dtau :
        Lattice spacing, default=0.05.

    Returns
    -------
    ndarray
        Instanton/anti-instanton centers.
    """
    tau_centers_ia = np.random.uniform(0, n_lattice * dtau, size=n_ia)
    tau_centers_ia = np.sort(tau_centers_ia)

    return tau_centers_ia


#@jit(nopython=True)
def centers_setup_gauss(tau_array, n_ia, n_lattice):
    """Generate a random ensemble of instantons/anti-instantons.

    Parameters
    ----------
    tau_array : ndarray
        Euclidean time axis.
    n_ia : int
        Number of instantons/anti-instantons
    n_lattice : int
        Number of lattice point in euclidean time.

    Returns
    -------
    tau_centers_ia : ndarray
        Instantons/anti-instantons centers.

    tau_centers_ia_index : ndarray
        Instantons/anti-instantons time axis indexes.

    """
    tau_centers_ia_index = np.random.randint(0, n_lattice, size=n_ia)
    tau_centers_ia_index = np.sort(tau_centers_ia_index)

    tau_centers_ia = np.zeros(n_ia)
    for i_tau in range(n_ia):
        tau_centers_ia[i_tau] = \
            tau_array[tau_centers_ia_index[i_tau]]

    return tau_centers_ia, tau_centers_ia_index


#@jit(nopython=True)
def ansatz_instanton_conf(tau_centers_ia,
                          tau_array,
                          x_potential_minimum):
    """Generate a path according to the sum ansatz.

    Parameters
    ----------
    tau_centers_ia : ndarray
        Instantons/anti-instantons ensemble.
    tau_array : ndarray
        Euclidean time axis.
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.
    dtau : float
        Lattice spacing.

    Returns
    -------
    x_ansatz : ndarray
        Configuration path.
    """
    # if tau_centers_ia.size == 0:
    #     x_ansatz = np.repeat(-ip.x_potential_minimum, tau_array.size + 1)
    #     return x_ansatz

    x_ansatz = np.zeros(tau_array.size, float)

    top_charge = 1
    for tau_ia in np.nditer(tau_centers_ia):
        x_ansatz += top_charge * x_potential_minimum \
                    * np.tanh(2 * x_potential_minimum
                              * (tau_array - tau_ia))

        top_charge *= -1

    x_ansatz -= x_potential_minimum

    # Boundary periodic conditions
    x_ansatz[0] = x_ansatz[-1]
    x_ansatz = np.append(x_ansatz, x_ansatz[1])

    return x_ansatz


#@jit(nopython=True)
def gaussian_potential(x_pos, tau, tau_ia_centers, x_potential_minimum):
    """Compute the gaussian potential for an ensemble of instantons/anti-
    instantons.

    Parameters
    ----------
    x_pos : ndarray
        Spatial configuration.
    tau : ndarray
        Euclidean time axis.
    tau_ia_centers : ndarray
        instantons/anti-instantons positions.
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.

    Returns
    -------
    potential : ndarray
        Gaussian potential.
    """
    potential = 0.0
    for tau_ia in np.nditer(tau_ia_centers):
        potential += -3.0 / (2.0 * np.power(np.cosh(2 * x_potential_minimum
                                                    * (tau - tau_ia)), 2))
    potential += 1
    potential *= 4.0 * x_potential_minimum \
                 * x_potential_minimum * x_pos * x_pos

    return potential


#@jit(nopython=True)
def hard_core_action(n_lattice,
                     tau_centers_ia,
                     tau_core,
                     action_core,
                     action_0,
                     dtau):
    """Compute the action for a short range repulsive hard core interaction.

    Parameters
    ----------
    n_lattice : int
        Number of lattice point in euclidean time.
    tau_centers_ia : ndarray
        Instanton/anti-instanton positions.
    tau_core : float
        Range of hard core interaction.
    action_core :
        Strength of hard core interaction.
    action_0 :
    dtau : float
        Lattice spacing.

    Returns
    -------
    action : float
        Hard core interaction action.

    """
    action = 0.0

    for i_ia in range(0, tau_centers_ia.size):
        if i_ia == 0:
            zero_crossing_m = tau_centers_ia[-1] - n_lattice * dtau
        else:
            zero_crossing_m = tau_centers_ia[i_ia - 1]

        action += action_0 * action_core * np.exp(-(tau_centers_ia[i_ia]
                                                    - zero_crossing_m)
                                                  / tau_core)

    return action


#@jit(nopython=True)
def configuration_heating(x_delta_config,
                          tau_array,
                          tau_centers_ia,
                          tau_centers_ia_index,
                          x_potential_minimum,
                          dtau,
                          delta_x):
    """Metropolis algorithm for Markov chain Monte Carlo simulations of an
    ensemble of instantons whose total action is computed using heating.

    Parameters
    ----------
    x_delta_config : ndarray
        Fluctuation path.
    tau_array : ndarray
        Euclidean time axis.
    tau_centers_ia : ndarray
        Instanton/anti-instanton positions.
    tau_centers_ia_index : ndarray
        Instantons/anti-instantons time axis indexes.
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.
    dtau : float
        Lattice spacing.
    delta_x : float
        Width of Gaussian distribution for Metropolis update.

    Returns
    ----------
    None
    """
    n_lattice = tau_array.size

    for i in range(1, n_lattice):

        if i in tau_centers_ia_index:
            continue
        else:
            action_loc_old = (
                                     np.square(
                                         x_delta_config[i] - x_delta_config[
                                             i - 1])
                                     + np.square(
                                 x_delta_config[i + 1] - x_delta_config[i])
                             ) / (4 * dtau) \
                             + dtau * gaussian_potential(x_delta_config[i],
                                                         tau_array[i],
                                                         tau_centers_ia,
                                                         x_potential_minimum)

            if (i + 1) in tau_centers_ia_index or (
                    i - 1) in tau_centers_ia_index:
                der = (x_delta_config[i + 1] -
                       x_delta_config[i - 1]) / (2.0 * dtau)
                if -0.001 < der < 0.001:
                    der = 1.0
                action_loc_old += -np.log(np.abs(der))

            x_new = x_delta_config[i] + np.random.normal(0, delta_x)

            action_loc_new = (
                                     np.square(x_new - x_delta_config[i - 1])
                                     + np.square(x_delta_config[i + 1] - x_new)
                             ) / (4 * dtau) \
                             + dtau * gaussian_potential(x_new, tau_array[i],
                                                         tau_centers_ia,
                                                         x_potential_minimum)

            if (i - 1) in tau_centers_ia_index:
                der = (x_delta_config[i + 1] - x_new) / (2.0 * dtau)
                if -0.001 < der < 0.001:
                    der = 1.0

                action_loc_new += - np.log(np.abs(der))

            elif (i + 1) in tau_centers_ia_index:
                der = (x_new - x_delta_config[i - 1]) / (2.0 * dtau)
                if -0.001 < der < 0.001:
                    der = 1.0
                action_loc_new += - np.log(np.abs(der))

            delta_action = action_loc_new - action_loc_old

            if np.exp(-delta_action) > np.random.uniform(0., 1.):
                x_delta_config[i] = x_new

    x_delta_config[n_lattice - 1] = x_delta_config[0]
    x_delta_config[n_lattice] = x_delta_config[1]


#@jit(nopython=True)
def rilm_monte_carlo_step(n_ia,  # number of instantons and anti inst.
                          n_points,  #
                          n_meas,
                          tau_array,
                          x_cor_sums,
                          x2_cor_sums,
                          x_potential_minimum,
                          dtau):
    """Compute correlation functions using the sum ansatz path and generate
    a random distribution of instantons/anti-instantons.

    Parameters
    ----------
    n_ia : int
        Number of instantons/anti-instantons.
    n_points : int
        Number of points on which correlation functions are computed.
    n_meas : int
        Number of measurement.
    tau_array : ndarray
        Euclidean time axis.
    x_cor_sums : ndarray
        Path spatial positions to be averaged.
    x2_cor_sums : ndarray
        Position squared to be averaged.
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.
    dtau : float
        Lattice spacing.

    Returns
    -------
    tau_centers_ia : ndarray
        Instantons/anti-instantons centers.
    """
    # Center of instantons and anti instantons
    tau_centers_ia = centers_setup(n_ia, tau_array.size)

    # Ansatz sum of indipendent instantons
    x_ansatz = ansatz_instanton_conf(tau_centers_ia,
                                     tau_array,
                                     x_potential_minimum)

    utility_custom.correlation_measurments(tau_array.size,
                                           n_meas,
                                           n_points,
                                           x_ansatz,
                                           x_cor_sums,
                                           x2_cor_sums)

    return tau_centers_ia


#@jit(nopython=True)
def rilm_heated_monte_carlo_step(n_ia,
                                 n_heating,
                                 n_points,
                                 n_meas,
                                 tau_array,
                                 x_cor_sums,
                                 x2_cor_sums,
                                 x_potential_minimum,
                                 dtau,
                                 delta_x):
    """Compute correlation functions for a random instanton ensemble using
    heating method.

    This function compute quantum correction to the semi-classical instant-
    on path using the heating method, where fluctuations are summed to the
    semi-classical solution. Quantum fluctuations are computed using the
    Metropolis Algorithm with a Gaussian potential.

    Parameters
    ----------
    n_ia : int
        Number of instantons/anti-instantons.
    n_heating : int
        Number of heating sweeps.
    n_points : int
        Number of points on which correlation functions are computed.
    n_meas : int
        Number of measurement of correlation functions in a MC sweep.
    tau_array : ndarray
        Euclidean time axis.
    x_cor_sums : ndarray
        Spatial configuration.
    x2_cor_sums : ndarray
        Spatial configuration squared.
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.
    dtau : float
        Lattice spacing.
    delta_x : float
        Width of Gaussian distribution for Metropolis update.

    Returns
    ----------
    None
    """
    # Center of instantons and anti instantons
    tau_centers_ia, tau_centers_ia_index = centers_setup_gauss(
        tau_array, n_ia, tau_array.size)

    # Ansatz sum of indipendent instantons
    x_ansatz = ansatz_instanton_conf(tau_centers_ia, tau_array,
                                     x_potential_minimum)
    # Difference from the classical solution
    x_delta_config = np.zeros((tau_array.size + 1))
    # Heating sweeps
    for _ in range(n_heating):
        configuration_heating(x_delta_config,
                              tau_array,
                              tau_centers_ia,
                              tau_centers_ia_index,
                              x_potential_minimum,
                              dtau,
                              delta_x)

        x_ansatz_heated = x_ansatz + x_delta_config

    utility_custom.correlation_measurments(tau_array.size,
                                           n_meas,
                                           n_points,
                                           x_ansatz_heated,
                                           x_cor_sums,
                                           x2_cor_sums)
