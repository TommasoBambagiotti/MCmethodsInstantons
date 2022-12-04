import numpy as np
from numba import njit


@njit
def potential_anh_oscillator(x_position,
                             x_potential_minimum):
    """Compute the anharmonic potential.

    Parameters
    ----------
    x_position : ndarray
        Space coordinates.
    x_potential_minimum : float
        Position of the minimum(a) of the potential.

    Returns
    -------
    ndarray
        Anharmonic potential.

    Notes
    -------
    Unity of measurements lambda=1.
    """
    return np.square(x_position * x_position
                     - x_potential_minimum * x_potential_minimum)


@njit
def potential_0_switching(x_position,
                          x_potential_minimum):
    """Compute the potential for the harmonic oscillator.

    Parameters
    ----------
    x_position : ndarray
        Space coordinates.
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.

    Returns
    -------
    ndarray
        Harmonic potential.

    Notes
    -------
    Unity of measurements lambda=1.
    """
    return np.square(x_position * (4 * x_potential_minimum)) / 4.


@njit
def potential_alpha(x_position,
                    x_potential_minimum,
                    alpha):
    """Compute the anharmonic potential using density switching.

    Parameters
    ----------
    x_position : ndarray
        Space coordinates.
    x_potential_minimum : float
        Position of the minimum(a) of the potential.
    alpha : float
        Switching algorithm integration step.

    Returns
    -------
    ndarray
        Potential.

    Notes
    -------
    Unity of measurements lambda=1.
    """
    potential_0 = potential_0_switching(x_position,
                                        x_potential_minimum)

    potential_1 = potential_anh_oscillator(x_position,
                                           x_potential_minimum)

    return alpha * (potential_1 - potential_0) + potential_0


@njit
def gaussian_potential(x_position,
                       x_potential_minimum,
                       a_alpha):
    """Compute the gaussian potential.

    The gaussian potential is computed expanding the action at second order
    in the system configuration around the classical one.

    Parameters
    ----------
    x_position : (1,3) ndarray
        First element is the classical configuration around which the pot-
        ential is expanded. Second element is the current spatial configur-
        ation.
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.
    a_alpha : float
        Switching algorithm integration step.

    Returns
    ----------
    float
        Gaussian potential for a given alpha.

    Notes
    -------
    Unity of measurements lambda=1.
    """
    potential_0 = 1.0 / 2.0 * x_position[2] \
                  * np.square(x_position[1] - x_position[0]) \
                  + potential_anh_oscillator(x_position[0],
                                             x_potential_minimum)

    potential_1 = potential_anh_oscillator(x_position[1],
                                           x_potential_minimum)

    return a_alpha * (potential_1 - potential_0) + potential_0


@njit
def metropolis_question_density_switching(x_config,
                                          x_0_config,
                                          second_der_0,
                                          f_potential,
                                          x_potential_minimum,
                                          dtau,
                                          delta_x,
                                          sector,
                                          a_alpha=1):
    """Metropolis algorithm for Markov chain Monte Carlo simulations of an
    ensemble of instantons whose total action is computed using adiabatic
    switching.

    This function generate a new configuration using the Metropolis-Has-
    tings algorithm, where the action is expanded around a reference con-
    figuration.

    Parameters
    ----------
    x_config : ndarray
        System (spatial) configuration.
    x_0_config : ndarray
        Classical configuration.
    second_der_0 : float
        Second derivative of the action.
    f_potential : function
        Potential (in coordinates space).
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.
    dtau : float
        Lattice spacing.
    delta_x : float
        Width of Gaussian distribution for Metropolis update.
    sector : {0,1}
        0 for the vacuum sector, 1 for the one instanton sector.
    a_alpha : float
        Switching algorithm integration step.

    Returns
    ----------
    None

    Notes
    ----------
    The Monte Carlo update is computed for a vacuum configuration, i.e.
    sector 0, and for the one instanton configuration, i.e. sector 1. In
    the second case the instanton position is fixed to the half of the
    total euclidean time. For the sector 0 periodic boundary condition
    are imposed, while anti-periodic for the other sector.
    We use a system of unit of measurements where h_bar=1, m=1/2 and
    lambda=1.
    """
    tau_fixed = int((x_config.size - 1) / 2)

    for i in range(1, x_config.size - 1):
        # for i = n_lattice/2 x is fixed
        if (i == tau_fixed) and (sector == 1):
            continue

        # expanding about the classical config

        x_position = np.array(
            [x_0_config[i], x_config[i], second_der_0[i - 1]])

        action_loc_old = (np.square(x_config[i] - x_config[i - 1])
                          + np.square(x_config[i + 1] - x_config[i])) / (
                                 4 * dtau) \
                         + dtau * f_potential(x_position,
                                              x_potential_minimum,
                                              a_alpha)

        # Jacobian (constrain)

        if (sector == 1) and ((i == tau_fixed + 1) or (i == tau_fixed - 1)):

            jacobian = (x_config[tau_fixed + 1] -
                        x_config[tau_fixed - 1]) / (2 * dtau)
            if np.abs(jacobian) < 0.001:
                jacobian = 0.001
            action_jac = np.log(np.abs(jacobian))
            action_loc_old = action_loc_old - action_jac

        x_new = x_config[i] + np.random.normal(0, delta_x)
        x_position[1] = x_new

        action_loc_new = (np.square(x_new - x_config[i - 1])
                          + np.square(x_config[i + 1] - x_new)) / (4 * dtau) \
                         + dtau * f_potential(x_position,
                                              x_potential_minimum,
                                              a_alpha)

        if (i == tau_fixed - 1) and (sector == 1):

            jacobian = (x_config[tau_fixed + 1] - x_new) / (2 * dtau)
            if np.abs(jacobian) < 0.001:
                jacobian = 0.001
            action_jac = np.log(np.abs(jacobian))

            action_loc_new = action_loc_new - action_jac

        if (i == tau_fixed + 1) and (sector == 1):

            jacobian = (x_new - x_config[tau_fixed - 1]) / (2 * dtau)
            if np.abs(jacobian) < 0.001:
                jacobian = 0.001
            action_jac = np.log(np.abs(jacobian))

            action_loc_new = action_loc_new - action_jac

        delta_action = action_loc_new - action_loc_old

        # Metropolis question:
        if np.exp(-delta_action) > np.random.uniform(0., 1.):
            x_config[i] = x_new

    # NEW BOUNDARY CONDITIONS
    if sector == 0:

        # PBC
        periodic_boundary_conditions(x_config)

    elif sector == 1:

        # APBC
        anti_periodic_boundary_conditions(x_config)


@njit
def metropolis_question(x_config,
                        x_potential_minimum,
                        f_potential,
                        dtau,
                        delta_x):
    """Metropolis algorithm for Markov chain Monte Carlo simulations of a
       physical described by the functional path integral in euclidean ti-
       me.

    This function generate a new configuration using the Metropolis-Has-
    tings algorithm. At the end periodic boundary conditions are imposed.

    Parameters
    ----------
    x_config : ndarray
        System (spatial) configuration.
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.
    f_potential : function
        Potential (in coordinates space).
    dtau : float
        Lattice spacing.
    delta_x : float
        Width of Gaussian distribution for Metropolis update.

    Returns
    ----------
    None

    Notes
    ----------
    We use a system of unit of measurements where h_bar=1, m=1/2 and
    lambda=1.
    """
    for i in range(1, x_config.size - 1):
        action_loc_old = (np.square(x_config[i] - x_config[i - 1])
                          + np.square(x_config[i + 1] - x_config[i])) / (
                                 4. * dtau) \
                         + dtau * f_potential(x_config[i],
                                              x_potential_minimum)

        x_new = x_config[i] + delta_x * \
                (2 * np.random.uniform(0.0, 1.0) - 1.)

        action_loc_new = (np.square(x_new - x_config[i - 1])
                          + np.square(x_config[i + 1] - x_new)) / (4. * dtau) \
                         + dtau * f_potential(x_new,
                                              x_potential_minimum)

        delta_action_exp = np.exp(action_loc_old - action_loc_new)

        if delta_action_exp > np.random.uniform(0., 1.):
            x_config[i] = x_new

    periodic_boundary_conditions(x_config)


@njit
def metropolis_question_switching(x_config,
                                  x_potential_minimum,
                                  f_potential,
                                  dtau,
                                  delta_x,
                                  alpha):
    """Metropolis algorithm for Markov chain Monte Carlo simulations of an
    ensemble of instantons whose total action is computed using adiabatic
    switching.

    This function generate a new configuration using the Metropolis-Has-
    tings algorithm, where the action is expanded around a reference con-
    figuration.

    Parameters
    ----------
    x_config : ndarray
        System (spatial) configuration.
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.
    f_potential : function
        Potential (in configurations space).
    dtau : float
        Lattice spacing.
    delta_x : float
        Width of Gaussian distribution for Metropolis update.
    alpha : float
        Switching algorithm integration step.

    Returns
    ----------
    None

    Notes
    ----------
    We use a system of unit of measurements where h_bar=1, m=1/2 and
    lambda=1.
    """
    for i in range(1, x_config.size - 1):
        action_loc_old = (np.square(x_config[i] - x_config[i - 1])
                          + np.square(x_config[i + 1] - x_config[i])) / (
                                 4. * dtau) \
                         + dtau * f_potential(x_config[i],
                                              x_potential_minimum,
                                              alpha)

        x_new = x_config[i] + delta_x * \
                (2 * np.random.uniform(0.0, 1.0) - 1.)

        action_loc_new = (np.square(x_new - x_config[i - 1])
                          + np.square(x_config[i + 1] - x_new)) / (4. * dtau) \
                         + dtau * f_potential(x_new,
                                              x_potential_minimum,
                                              alpha)

        delta_action_exp = np.exp(action_loc_old - action_loc_new)

        if delta_action_exp > np.random.uniform(0., 1.):
            x_config[i] = x_new

    periodic_boundary_conditions(x_config)


@njit
def configuration_cooling(x_cold_config,
                          x_potential_minimum,
                          f_potential,
                          dtau,
                          delta_x):
    """Metropolis algorithm for Markov chain Monte Carlo simulations of an
    ensemble of cooled instantons.

    Parameters
    ----------
    x_cold_config : ndarray
        System (spatial) configuration.
    x_potential_minimum : flaot
        Position of the minimum(a) of the anharmonic potential.
    f_potential : function
        Potential (in configurations space).
    dtau : float
        Lattice spacing.
    delta_x :
        Width of Gaussian distribution for Metropolis update.

    Returns
    ----------
    None

    Notes
    ---------
    The initial configuration is forced to move towards the classical so-
    lution by updating only those configurations whose actions decrease.
    At the end periodic boundary conditions are imposed.
    We use a system of unit of measurements where h_bar=1, m=1/2 and
    lambda=1.
    """
    n_trials = 10

    for i in range(1, x_cold_config.size - 1):
        action_loc_old = (np.square(x_cold_config[i] - x_cold_config[i - 1])
                          + np.square(
                    x_cold_config[i + 1] - x_cold_config[i])) / (4. * dtau) \
                         + dtau * f_potential(x_cold_config[i],
                                              x_potential_minimum)

        for _ in range(n_trials):
            x_new = x_cold_config[i] + delta_x * \
                    (2 * np.random.uniform(0.0, 1.0) - 1.)

            action_loc_new = (np.square(x_new - x_cold_config[i - 1])
                              + np.square(x_cold_config[i + 1] - x_new)) / (
                                     4. * dtau) \
                             + dtau * f_potential(x_new,
                                                  x_potential_minimum)

            if action_loc_new < action_loc_old:
                x_cold_config[i] = x_new

    periodic_boundary_conditions(x_cold_config)


@njit
def return_action(x_config,
                  x_potential_minimum,
                  dtau):
    """Compute the action for the anharmonic potential.

    Parameters
    ----------
    x_config : ndarray
        Spatial configuration.
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.
    dtau : float
        Lattice spacing.
    Returns
    -------
    float
        Action.

    Notes
    -------
    We use a system of unit of measurements where h_bar=1.
    """
    action = np.square(x_config[1:-1] - x_config[0:-2]) / (4. * dtau) \
             + dtau * potential_anh_oscillator(x_config[1:-1],
                                               x_potential_minimum)

    return np.sum(action)


@njit
def initialize_lattice(n_lattice,
                       x_potential_minimum,
                       i_cold=False,
                       classical_config=False,
                       dtau=0.05):
    """Initialize first configuration for Monte Carlo simulations.

    Parameters
    ----------
    n_lattice : int
        Number of lattice point in euclidean time.
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.
    i_cold : bool, default=False
        True for cold start, False for hot start.
    classical_config : bool, default=False
        True for classical instanton configuration, False otherwise.
    dtau : float, default=0.05
        Lattice spacing.

    Returns
    ----------
    ndarray
        Initialized configuration.
    """
    if (i_cold is True) and (not classical_config):
        x_config = np.zeros((n_lattice + 1))
        for i in range(n_lattice + 1):
            x_config[i] = -x_potential_minimum

        return x_config

    elif (i_cold is False) and (not classical_config):

        x_config = np.random.uniform(-x_potential_minimum,
                                     x_potential_minimum,
                                     size=n_lattice + 1)

        # PBC
        periodic_boundary_conditions(x_config)

        return x_config

    elif classical_config is True:
        tau_inst = (dtau * n_lattice) / 2
        tau_array = np.linspace(0, n_lattice * dtau, n_lattice + 1)
        x_config = instanton_classical_configuration(tau_array,
                                                     tau_inst,
                                                     x_potential_minimum)

        # APBC
        anti_periodic_boundary_conditions(x_config)

        return x_config


@njit
def find_instantons(x, dt):
    """Find the number of instantons and anti-instantons and save their
    positions.

    Parameters
    ----------
    x : ndarray
        Spatial configuration.
    dt : ndarray
        Euclidean time axis.

    Returns
    -------
    pos_roots : int
        Number of instantons.
    neg_roots : int
        Number of anti-instantons.
    a : array
        Instanton positions.
    b : array
        Anti-instanton positions.
    """
    pos_roots = 0
    neg_roots = 0
    pos_roots_position = np.array([0.0])
    neg_roots_position = np.array([0.0])
    # pos_roots_position = []
    # neg_roots_position = []

    if np.abs(x[0]) < 1e-7:
        x[0] = 0.0

    x_pos = x[0]

    for i in range(1, x.size - 1):
        if np.abs(x[i]) < 1e-7:
            x[i] = 0.0

            if x_pos > 0.:
                neg_roots += 1
                neg_roots_position = np.append(
                    neg_roots_position,
                    -x_pos * dt
                    / (x[i] - x_pos) + (i - 1) * dt
                )
            elif x_pos < 0.:
                pos_roots += 1
                pos_roots_position = np.append(
                    pos_roots_position,
                    -x_pos * dt
                    / (x[i] - x_pos) + dt * (i - 1)
                )
            else:
                continue

        elif x_pos * x[i] < 0.:

            if x[i] > x_pos:
                pos_roots += 1
                pos_roots_position = np.append(
                    pos_roots_position,
                    -x_pos * dt
                    / (x[i] - x_pos) + dt * (i - 1)
                )

            elif x[i] < x_pos:
                neg_roots += 1
                neg_roots_position = np.append(
                    neg_roots_position,
                    -x_pos * dt
                    / (x[i] - x_pos) + (i - 1) * dt
                )

        x_pos = x[i]

    if neg_roots == 0 or pos_roots == 0:
        return 0, 0, np.zeros(1), np.zeros(1)

    a = np.delete(pos_roots_position, 0)
    b = np.delete(neg_roots_position, 0)

    return pos_roots, neg_roots, \
           a, b


@njit
def instanton_classical_configuration(tau_pos,
                                      tau_0,
                                      x_potential_minimum):
    """Compute the instanton semi-classical configuration.

    Parameters
    ----------
    tau_pos : ndarray
        Time axis.
    tau_0 : float
        Instanton position (in euclidean time).
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.

    Returns
    -------
    ndarray
        Action.
    """
    return x_potential_minimum * np.tanh(
        2 * x_potential_minimum * (tau_pos - tau_0))


@njit
def second_derivative_action(x_0, x_potential_minimum):
    """Compute the second derivative of the action of the anharmonic
    oscillator.

    Parameters
    ----------
    x_0 : ndarray
        Spatial axis.
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.

    Returns
    -------
    ndarray
        Second derivative.

    Notes
    -------
    We use a system of unit of measurements where h_bar=1.
    """
    return 12 * x_0 * x_0 - 4 * x_potential_minimum * x_potential_minimum


@njit
def periodic_boundary_conditions(x_config):
    """Impose periodic boundary conditions.

    Parameters
    ----------
    x_config : ndarray
        System configuration.

    Returns
    ----------
    None
    """
    x_config[0] = x_config[- 2]
    x_config[-1] = x_config[1]


@njit
def anti_periodic_boundary_conditions(x_config):
    """Impose anti-periodic boundary conditions.

    Parameters
    ----------
    x_config : ndarray
        System configuration.

    Returns
    ----------
    None
    """
    x_config[0] = - x_config[-2]
    x_config[-1] = - x_config[1]


def two_loop_density(x_pot_min):
    """Compute the instanton density at 2-loop.

    Parameters
    ----------
    x_pot_min : float
        Anharmonic potential minimum(a).

    Returns
    -------
    float
        Density.
    """
    action_0 = np.power(x_pot_min, 3) * 4 / 6

    return 8 * pow(x_pot_min, 5 / 2) * pow(2 / np.pi, 1 / 2) \
           * np.exp(-action_0 - 71 / (72 * action_0))
