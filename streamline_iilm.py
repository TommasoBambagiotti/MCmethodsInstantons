'''
Solve the action problem using
the streamline method as a valid
alternative of the classical sum ansatz
'''

import numpy as np
import utility_rilm as rilm
import utility_custom
import utility_monte_carlo as mc


def print_sum_ansatz_ia(n_lattice,
                        x_potential_minimum,
                        dtau):
    """Dependence of the interactive action
    with respect to the instanton-anti instanton
    separation, in the near repulsive case

    Parameters
    ----------
    n_lattice :
    x_potential_minimum :
    dtau :
    """
    output_path = './output_data/output_iilm/streamline'
    utility_custom.output_control(output_path)

    n_inst = int(n_lattice / 4)

    array_ia = np.zeros(n_inst)
    array_int_core = np.zeros(n_inst)
    array_int = np.zeros(n_inst)
    array_ia_0_list = []
    array_int_zero_cross_list = []

    action_0 = 4 / 3 * np.power(x_potential_minimum, 3)

    for n_a in range(n_inst, 2 * n_inst):

        tau_centers_ia = np.array([n_inst * dtau, n_a * dtau])
        tau_array = np.linspace(0., n_lattice * dtau, n_lattice, False)
        x_config = rilm.ansatz_instanton_conf(tau_centers_ia,
                                              tau_array,
                                              x_potential_minimum,
                                              dtau)

        action = mc.return_action(x_config,
                                  x_potential_minimum,
                                  dtau)

        tau_core = 0.3 / x_potential_minimum
        action_core = 3.0

        action_int_core = action_0 * action_core * (np.exp(-(tau_centers_ia[1]
                                                             - tau_centers_ia[
                                                                 0])
                                                           / tau_core)
                                                    + np.exp(
                    -(tau_centers_ia[0]
                      - tau_centers_ia[1]
                      + n_lattice * dtau)
                    / tau_core)
                                                    )

        action_int_core += action

        action_int_core /= action_0
        action /= action_0

        action_int_core -= 2.
        action -= 2.

        array_int_core[n_a - n_inst] = action_int_core
        array_int[n_a - n_inst] = action
        array_ia[n_a - n_inst] = tau_centers_ia[1] - tau_centers_ia[0]

        n_i, n_an, pos_roots, neg_roots = mc.find_instantons(
            x_config, dtau)

        z_ia = 0
        if n_a % 2 == 0 \
                and n_i == n_an \
                and n_i > 0 \
                and n_i == len(pos_roots) \
                and n_an == len(neg_roots):

            if pos_roots[0] < neg_roots[0]:

                for i in range(n_i):
                    if i == 0:
                        zero_m = neg_roots[-1] - n_lattice * dtau
                    else:
                        zero_m = neg_roots[i - 1]

                    z_ia = np.minimum(np.abs(neg_roots[i] - pos_roots[i]),
                                      np.abs(pos_roots[i] - zero_m))

            elif pos_roots[0] > neg_roots[0]:
                for i in range(n_i):
                    if i == 0:
                        zero_p = pos_roots[-1] - n_lattice * dtau
                    else:
                        zero_p = pos_roots[i - 1]

                    z_ia = np.minimum(np.abs(pos_roots[i] - neg_roots[i]),
                                      np.abs(neg_roots[i] - zero_p))

            array_ia_0_list.append(z_ia)
            array_int_zero_cross_list.append(mc.return_action(x_config,
                                                              x_potential_minimum,
                                                              dtau)
                                             )

    array_ia_0 = np.array(array_ia_0_list, float)
    array_int_zero_cross = np.array(array_int_zero_cross_list, float)
    array_int_zero_cross /= 4 / 3 * np.power(x_potential_minimum, 3)
    array_int_zero_cross -= 2

    np.savetxt(output_path + '/array_ia.txt', array_ia)
    np.savetxt(output_path + '/array_ia_0.txt', array_ia_0)
    np.savetxt(output_path + '/array_int.txt', array_int)
    np.savetxt(output_path + '/array_int_core.txt', array_int_core)
    np.savetxt(output_path + '/array_int_zero_cross.txt', array_int_zero_cross)


# Save, for completeness, also the action density

def streamline_method_iilm(r_initial_sep,
                           stream_time_step,
                           n_lattice_half,
                           n_lattice,
                           n_streamline,
                           x_potential_minimum=1.4,
                           dtau=0.05):
    """

    Parameters
    ----------
    r_initial_sep :
    stream_time_step :
    n_lattice_half :
    n_lattice :
    n_streamline :
    x_potential_minimum :
    dtau :
    """
    print_sum_ansatz_ia(n_lattice,
                        x_potential_minimum,
                        dtau)

    output_path = './output_data/output_iilm/streamline'
    utility_custom.output_control(output_path)

    tau_centers_ia = np.array([n_lattice_half * dtau - r_initial_sep / 2.0,
                               n_lattice_half * dtau + r_initial_sep / 2.0])

    tau_array = np.linspace(0.0, n_lattice_half * 2 *
                            dtau, n_lattice_half * 2, False)

    # Initial condition of the streamline
    x_config = rilm.ansatz_instanton_conf(tau_centers_ia,
                                          tau_array,
                                          x_potential_minimum,
                                          dtau)

    x_config = np.insert(x_config, 0, x_config[0])
    x_config = np.insert(x_config, 0, x_config[0])
    x_config[-1] = x_config[-2]
    x_config = np.append(x_config, x_config[-1])

    action_density = np.zeros(2 * n_lattice_half, float)
    # Derivative in streamline parameter
    lambda_derivative = np.zeros(2 * n_lattice_half)

    # np.savetxt(output_path + '/tau_array.txt', tau_array)
    # np.savetxt(output_path + '/streamline_0.txt', x_config[2:-2])

    ansatz_action = 4 / 3 * pow(x_potential_minimum, 3)

    # Streamline evolution

    tau_writer = open(output_path + '/delta_tau_ia.txt', 'w', encoding='utf8')
    act_writer = open(
        output_path + '/streamline_action_int.txt', 'w', encoding='utf8')

    for i_s in range(n_streamline):
        if i_s % 1000 == 0:
            print(f'streamline #{i_s}')
        # Evaluate the derivative of the action
        for i_pos in range(2, 2 * n_lattice_half + 2, 1):
            der_2 = (-x_config[i_pos + 2] + 16 * x_config[i_pos + 1] - 30 *
                     x_config[i_pos]
                     + 16 * x_config[i_pos - 1] - x_config[i_pos - 2]) \
                    / (12 * dtau * dtau)

            lambda_derivative[i_pos - 2] = - der_2 / 2.0 + 4 * x_config[i_pos] \
                                           * (x_config[i_pos]
                                              * x_config[i_pos]
                                              - x_potential_minimum
                                              * x_potential_minimum)

        for i_pos in range(2, 2 * n_lattice_half + 2):
            x_config[i_pos] += -lambda_derivative[i_pos - 2] * stream_time_step

        x_config[0] = x_config[2]
        x_config[1] = x_config[2]
        x_config[-1] = x_config[-3]
        x_config[-2] = x_config[-3]

        for i in range(2, 2 * n_lattice_half + 2):
            v = (x_config[i] ** 2 - x_potential_minimum ** 2) ** 2
            k = (x_config[i + 1] - x_config[i - 1]) / (2. * dtau)
            action_density[i - 2] = k ** 2 / 4. + v

        current_action = mc.return_action(x_config[2:-1],
                                          x_potential_minimum,
                                          dtau)
        n_i, n_a, pos_root, neg_root = mc.find_instantons(
            x_config[2:-2], dtau)

        if i_s > 63000 \
                and i_s < 64000 \
                and i_s % 20 == 0:
            if n_i == n_a \
                    and n_i != 0 \
                    and pos_root.size == n_i and neg_root.size == n_a:
                interactive_action = current_action - 2 * ansatz_action

                tau_i_a = np.abs(pos_root[0] - neg_root[0])

                tau_writer.write(str(tau_i_a) + '\n')

                act_writer.write(
                    str(interactive_action / ansatz_action) + '\n')

        # if current_action > 0.0001:
        #     if current_action / ansatz_action > 1.8:
        #         np.savetxt(output_path + '/streamline_1.txt', x_config[2:-2])
        #     if current_action / ansatz_action > 1.6:
        #         np.savetxt(output_path + '/streamline_2.txt', x_config[2:-2])
        # if current_action / ansatz_action > 1.4:
        #     np.savetxt(output_path + '/streamline_3.txt', x_config[2:-2])
        # if current_action / ansatz_action > 1.2:
        #     np.savetxt(output_path + '/streamline_4.txt', x_config[2:-2])
        # if current_action / ansatz_action > 1.0:
        #    np.savetxt(output_path + '/streamline_5.txt', x_config[2:-2])
        # if current_action / ansatz_action > 0.8:
        #     np.savetxt(output_path + '/streamline_6.txt', x_config[2:-2])
        # if current_action / ansatz_action > 0.5:
        #      np.savetxt(output_path + '/streamline_7.txt', x_config[2:-2])
        # if current_action / ansatz_action > 0.2:
        #     np.savetxt(output_path + '/streamline_8.txt', x_config[2:-2])

    tau_writer.close()
    act_writer.close()
