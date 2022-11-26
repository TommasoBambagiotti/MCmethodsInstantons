import numpy as np
#from numba import jit, njit

#@jit(nopython=True)
def potential_anh_oscillator(x_position,
                             x_potential_minimum):

    return np.square(x_position * x_position
                     - x_potential_minimum * x_potential_minimum)


#@jit(nopython=True)
def potential_0_switching(x_position,
                          x_potential_minimum):

    return np.square(x_position * (4 * x_potential_minimum))/4.


#@jit(nopython=True)
def potential_alpha(x_position,
                    x_potential_minimum,
                    alpha):

    potential_0 = potential_0_switching(x_position,
                                        x_potential_minimum)

    potential_1 = potential_anh_oscillator(x_position,
                                           x_potential_minimum)

    return alpha * (potential_1 - potential_0) + potential_0


#@njit
def gaussian_potential(x_position,
                       x_potential_minimum,
                       a_alpha):

    potential_0 = 1.0 / 2.0 * x_position[2]\
        * np.square(x_position[1] - x_position[0]) \
        + potential_anh_oscillator(x_position[0],
                                   x_potential_minimum)

    potential_1 = potential_anh_oscillator(x_position[1],
                                           x_potential_minimum)

    return a_alpha * (potential_1 - potential_0) + potential_0


#@jit(nopython=True)
def metropolis_question_density_switching(x_config,
                                          x_0_config,
                                          second_der_0,
                                          f_potential,
                                          x_potential_minimum,
                                          dtau,
                                          delta_x,
                                          sector,  # 0-instanton sector or 1-instanton sector
                                          a_alpha=1):

    tau_fixed = int((x_config.size - 1) / 2)

    for i in range(1, x_config.size - 1):
        # for i = n_lattice/2 x is fixed
        if (i == tau_fixed) and (sector == 1):
            continue

        # expanding about the classical config


        x_position = np.array([x_0_config[i], x_config[i], second_der_0[i]])

        action_loc_old = (np.square(x_config[i] - x_config[i - 1])
                          + np.square(x_config[i + 1] - x_config[i])) / (4 * dtau)\
            + dtau * f_potential(x_position,
                                 x_potential_minimum,
                                 a_alpha)

        # Jacobian (constrain)

        if (i == tau_fixed + 1) or (i == tau_fixed - 1) and (sector == 1):

            jacobian = (x_config[tau_fixed+1] -
                        x_config[tau_fixed-1]) / (2*dtau)
            if np.abs(jacobian)  < 0.001:
                jacobian = 0.0
            action_jac = np.log(np.abs(jacobian))
            action_loc_old = action_loc_old - action_jac

        x_new = x_config[i] + np.random.gauss(0, delta_x)
        x_position[1] = x_new

        action_loc_new = (np.square(x_new - x_config[i-1])
                          + np.square(x_config[i+1] - x_new)) / (4 * dtau)\
            + dtau * f_potential(x_position,
                                 x_potential_minimum,
                                 a_alpha)

        if (i == tau_fixed-1) and (sector == 1):

            jacobian = (x_config[tau_fixed+1] - x_new) / (2*dtau)
            if np.abs(jacobian)  < 0.001:
                jacobian = 0.0
            action_jac = np.log(np.abs(jacobian))

            action_loc_new = action_loc_new - action_jac

        if (i == tau_fixed+1) and (sector == 1):

            jacobian = (x_new - x_config[tau_fixed-1]) / (2*dtau)
            if np.abs(jacobian)  < 0.001:
                jacobian = 0.0
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


#@jit(nopython=True)
def metropolis_question(x_config,
                        x_potential_minimum,
                        dtau,
                        delta_x):

    for i in range(1, x_config.size - 1):
        action_loc_old = (np.square(x_config[i] - x_config[i-1])
                          + np.square(x_config[i+1] - x_config[i]))/(4. * dtau)\
            + dtau * potential_anh_oscillator(x_config[i],
                                              x_potential_minimum)

        x_new = x_config[i] + delta_x * \
            (2*np.random.uniform(0.0, 1.0) - 1.)

        action_loc_new = (np.square(x_new - x_config[i-1])
                          + np.square(x_config[i+1] - x_new))/(4. * dtau)\
            + dtau * potential_anh_oscillator(x_new,
                                              x_potential_minimum)

        delta_action_exp = np.exp(action_loc_old-action_loc_new)

        if delta_action_exp > np.random.uniform(0., 1.):
            x_config[i] = x_new

    x_config[0] = x_config[-2]
    x_config[-1] = x_config[1]


#@jit(nopython=True)
def metropolis_question_switching(x_config,
                                  x_potential_minimum,
                                  dtau,
                                  delta_x,
                                  alpha):

    for i in range(1, x_config.size - 1):
        action_loc_old = (np.square(x_config[i] - x_config[i-1])
                          + np.square(x_config[i+1] - x_config[i]))/(4. * dtau)\
            + dtau * potential_alpha(x_config[i],
                                     x_potential_minimum,
                                     alpha)

        x_new = x_config[i] + delta_x * \
            (2*np.random.uniform(0.0, 1.0) - 1.)

        action_loc_new = (np.square(x_new - x_config[i-1])
                          + np.square(x_config[i+1] - x_new))/(4. * dtau)\
            + dtau * potential_alpha(x_new,
                                     x_potential_minimum,
                                     alpha)

        delta_action_exp = np.exp(action_loc_old-action_loc_new)

        if delta_action_exp > np.random.uniform(0., 1.):
            x_config[i] = x_new

    x_config[0] = x_config[-2]
    x_config[-1] = x_config[1]


#@jit(nopython=True)
def return_action(x_config,
                  x_potential_minimum,
                  dtau):

    action = 0.0

    action = np.square(x_config[1:-1] - x_config[0:-2])/(4. * dtau)\
        + dtau * potential_anh_oscillator(x_config[1:-1],
                                          x_potential_minimum)

    return np.sum(action)


def initialize_lattice(n_lattice,
                       x_potential_minimum,
                       i_cold=False,
                       classical_config=False,
                       dtau=0.05):

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
        #x_config[n_lattice - 1] = x_config[0]
        #x_config[n_lattice] = x_config[1]
        periodic_boundary_conditions(x_config)

        return x_config

    elif classical_config is True:
        tau_inst = (dtau*n_lattice)/2
        tau_array = np.linspace(0, n_lattice * dtau, n_lattice + 1)
        x_config = instanton_classical_configuration(tau_array,
                                                     tau_inst,
                                                     x_potential_minimum)
    
        # APBC
        #x_config[1] = - x_config[n_lattice]
        #x_config[0] = x_config[n_lattice-1]
        anti_periodic_boundary_conditions(x_config)
    
        return x_config


#@jit(nopython=True)
def configuration_cooling(x_cold_config,
                          x_potential_minimum,
                          dtau,
                          delta_x):
    n_trials = 10

    for i in range(1, x_cold_config.size - 1):
        action_loc_old = (np.square(x_cold_config[i] - x_cold_config[i-1])
                          + np.square(x_cold_config[i+1] - x_cold_config[i]))/(4. * dtau)\
            + dtau * potential_anh_oscillator(x_cold_config[i],
                                              x_potential_minimum)

        for _ in range(n_trials):
            x_new = x_cold_config[i] + delta_x * \
                (2*np.random.uniform(0.0, 1.0) - 1.)

            action_loc_new = (np.square(x_new - x_cold_config[i-1])
                              + np.square(x_cold_config[i+1] - x_new))/(4. * dtau)\
                + dtau * potential_anh_oscillator(x_new,
                                                  x_potential_minimum)
          
            if action_loc_new < action_loc_old:
                x_cold_config[i] = x_new

    x_cold_config[0] = x_cold_config[-2]
    x_cold_config[-1] = x_cold_config[1]


#@jit(nopython=True)
def find_instantons(x, dt):

    pos_roots = 0
    neg_roots = 0
    pos_roots_position = np.array([0.0])
    neg_roots_position = np.array([0.0])
    #pos_roots_position = []
    #neg_roots_position = []
   

    if np.abs(x[0]) < 1e-7:
        x[0] = 0.0

    x_pos = x[0]

    for i in range(1, x.size-1):
        if np.abs(x[i]) < 1e-7:
            x[i] = 0.0

            if x_pos > 0.:
                neg_roots += 1
                neg_roots_position = np.append(
                    neg_roots_position,
                    -x_pos * (dt)
                    / (x[i] - x_pos) + (i-1) * dt
                )
            elif x_pos < 0.:
                pos_roots += 1
                pos_roots_position = np.append(
                    pos_roots_position,
                    -x_pos * (dt)
                    / (x[i] - x_pos) + dt * (i-1)
                )
            else:
                continue

        elif x_pos * x[i] < 0.:

            if x[i] > x_pos:
                pos_roots += 1
                pos_roots_position = np.append(
                    pos_roots_position,
                    -x_pos * (dt)
                    / (x[i] - x_pos) + dt * (i-1)
                )

            elif x[i] < x_pos:
                neg_roots += 1
                neg_roots_position = np.append(
                    neg_roots_position,
                    -x_pos * (dt)
                    / (x[i] - x_pos) + (i-1) * dt
                )

        x_pos = x[i]

    if neg_roots == 0 or pos_roots == 0:
        return 0, 0, np.zeros(1), np.zeros(1)

    a = np.delete(pos_roots_position, 0)
    b = np.delete(neg_roots_position, 0)

    # a = np.array(pos_roots_position, float)
    # b = np.array(neg_roots_position, float)

    return pos_roots, neg_roots, \
        a, b


def instanton_classical_configuration(tau_pos,
                                      tau_0,
                                      x_potential_minimum):

    return x_potential_minimum * np.tanh(2 * x_potential_minimum * (tau_pos - tau_0))


#@jit(nopython=True)
def second_derivative_action(x_0, x_potential_minimum):

    return 12 * x_0 * x_0 - 4 * x_potential_minimum * x_potential_minimum


#@jit(nopython=True)
def periodic_boundary_conditions(x_config):

    x_config[0] = x_config[- 2]
    x_config[-1] = x_config[1]


#@jit(nopython=True)
def anti_periodic_boundary_conditions(x_config):

    x_config[0] = - x_config[-2]
    x_config[-1] = - x_config[1]
