import numpy as np
import utility_rilm as rilm
import utility_custom
import utility_monte_carlo as mc


def sum_ansatz_ia(n_lattice,
                  x_potential_minimum,
                  dtau):
    """Dependence of the interactive action with respect to the instanton-
    anti-instanton separation, in the near repulsive case. Results are sa-
    ved into files.

    Parameters
    ----------
    n_lattice : int
        Number of lattice point in euclidean time.
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.
    dtau : float
        Lattice spacing.

    Returns
    ----------
    None
    """
    output_path = './output_data/output_iilm/streamline'
    utility_custom.output_control(output_path)

    n_lattice_4 = int(n_lattice / 4)

    # Action
    tau_ia_ansatz = np.zeros(n_lattice_4)
    action_int_ansatz = np.zeros(n_lattice_4)
    tau_ia_zcr_list = []
    action_int_zcr_list = []

    # Semi-classical action for one instanton
    action_0 = 4 / 3 * np.power(x_potential_minimum, 3)

    for n_counter in range(n_lattice_4, 2 * n_lattice_4):
        # Initialize variables
        tau_centers_ia = np.array([n_lattice_4 * dtau, n_counter * dtau])
        tau_array = np.linspace(0., n_lattice * dtau, n_lattice, False)
        x_config = rilm.ansatz_instanton_conf(tau_centers_ia,
                                              tau_array,
                                              x_potential_minimum)
        # Total action
        action = mc.return_action(x_config,
                                  x_potential_minimum,
                                  dtau)
        # Normalization
        action /= action_0
        action -= 2.
        
        action_int_ansatz[n_counter - n_lattice_4] = action
        tau_ia_ansatz[n_counter - n_lattice_4] = tau_centers_ia[1] - tau_centers_ia[0]
        # Tau zero crossing
        n_inst, n_a_inst, pos_roots, neg_roots = mc.find_instantons(
            x_config, dtau)

        if n_counter % 2 == 0 \
                and n_inst == n_a_inst \
                and n_inst > 0 \
                and n_inst == len(pos_roots) \
                and n_a_inst == len(neg_roots):

            
            for i in range(n_inst):
                if i == 0:
                    zero_m = neg_roots[-1] - n_lattice * dtau
                else:
                    zero_m = neg_roots[i - 1]

                z_ia = np.minimum(np.abs(neg_roots[i] - pos_roots[i]),
                                  np.abs(pos_roots[i] - zero_m))

            
            tau_ia_zcr_list.append(z_ia)
            action_int_zcr_list.append(mc.return_action(x_config,
                                                        x_potential_minimum,
                                                        dtau)
                                             )

    tau_ia_zcr = np.array(tau_ia_zcr_list, float)
    action_int_zcr = np.array(action_int_zcr_list, float)
    # Normalization
    action_int_zcr /= 4 / 3 * np.power(x_potential_minimum, 3)
    action_int_zcr -= 2

    # Save action into files
    np.savetxt(output_path + '/tau_ia_ansatz.txt', tau_ia_ansatz)
    np.savetxt(output_path + '/tau_ia_zcr.txt', tau_ia_zcr)
    np.savetxt(output_path + '/action_int_ansatz.txt', action_int_ansatz)
    np.savetxt(output_path + '/action_int_zcr.txt', action_int_zcr)


def streamline_method_iilm(r_initial_sep,
                           stream_time_step,
                           n_lattice_half,
                           n_lattice,
                           n_streamline,
                           print_valley=False,
                           x_potential_minimum=1.4,
                           dtau=0.05):
    """Solve the streamline equation for an instanton/anti-instanton
    pair.

    The streamline equation is solved using the descent method.
    We use a system of unit of measurements where h_bar=1, m=1/2 and
    lambda=1.

    Parameters
    ----------
    r_initial_sep :
        Initial instanton/anti-inst. pair separation.
    stream_time_step :
        Streamline time step.
    n_lattice_half : int
        Effective number of lattice points.
    n_lattice : int
        Number of lattice point in euclidean time.
    n_streamline : int
        Number of iteration in descent method.
    print_valley : bool, default=False
        If True, save into files streamline paths.
    x_potential_minimum : float
        Position of the minimum(a) of the anharmonic potential.
    dtau : float
        Lattice spacing.
    Returns
    ----------
    None
    """
    sum_ansatz_ia(n_lattice,
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
                                          x_potential_minimum)

    x_config = np.insert(x_config, 0, x_config[0])
    x_config = np.insert(x_config, 0, x_config[0])
    x_config[-1] = x_config[-2]
    x_config = np.append(x_config, x_config[-1])

    action_density = np.zeros(2 * n_lattice_half, float)

    # Derivative in streamline parameter
    lambda_derivative = np.zeros(2 * n_lattice_half)

    np.savetxt(output_path + '/tau_array.txt', tau_array)
    np.savetxt(output_path + '/stream_0.txt', x_config[2:-2])
    
    ansatz_action = 4 / 3 * pow(x_potential_minimum, 3)


    tau_writer = open(output_path + '/delta_tau_ia.txt', 'w', encoding='utf8')
    act_writer = open(
        output_path + '/streamline_action_int.txt', 'w', encoding='utf8')

    tau_store = 1.5
    
    # Streamline evolution
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

        if i_s == 0:
            np.savetxt(output_path + '/streamline_action_dens_0.txt', 
                       action_density)

        current_action = mc.return_action(x_config[2:-1],
                                          x_potential_minimum,
                                          dtau)
        n_i, n_a, pos_root, neg_root = mc.find_instantons(
            x_config[2:-2], dtau)

        if 59000 < i_s < 64000 and i_s % 10 == 0:
            if n_i == n_a \
                    and n_i != 0 \
                    and pos_root.size == n_i and neg_root.size == n_a:
                interactive_action = current_action - 2 * ansatz_action

                tau_i_a = np.abs(pos_root[0] - neg_root[0])
                if tau_i_a < tau_store - 0.08:
                    tau_store = tau_i_a
                    tau_writer.write(str(tau_i_a) + '\n')

                    act_writer.write(
                        str(interactive_action / ansatz_action) + '\n')


        if print_valley is True:
            if current_action > 0.0001:
                if current_action / ansatz_action > 1.8:
                    np.savetxt(output_path + '/stream_1.txt', x_config[2:-2])
                    np.savetxt(output_path + '/streamline_action_dens_1.txt', 
                               action_density)
                elif current_action / ansatz_action > 1.6:
                    np.savetxt(output_path + '/stream_2.txt', x_config[2:-2])
                    np.savetxt(output_path + '/streamline_action_dens_2.txt', 
                               action_density)
                elif current_action / ansatz_action > 1.4:
                    np.savetxt(output_path + '/stream_3.txt', x_config[2:-2])
                    np.savetxt(output_path + '/streamline_action_dens_3.txt', 
                               action_density)
                elif current_action / ansatz_action > 1.2:
                    np.savetxt(output_path + '/stream_4.txt', x_config[2:-2])
                    np.savetxt(output_path + '/streamline_action_dens_4.txt', 
                               action_density)
                elif current_action / ansatz_action > 1.0:
                    np.savetxt(output_path + '/stream_5.txt', x_config[2:-2])
                    np.savetxt(output_path + '/streamline_action_dens_5.txt', 
                               action_density)
                elif current_action / ansatz_action > 0.8:
                    np.savetxt(output_path + '/stream_6.txt', x_config[2:-2])
                    np.savetxt(output_path + '/streamline_action_dens_6.txt', 
                               action_density)
                elif current_action / ansatz_action > 0.6:
                    np.savetxt(output_path + '/stream_7.txt', x_config[2:-2])
                    np.savetxt(output_path + '/streamline_action_dens_7.txt', 
                               action_density)
                elif current_action / ansatz_action > 0.4:
                    np.savetxt(output_path + '/stream_8.txt', x_config[2:-2])
                    np.savetxt(output_path + '/streamline_action_dens_8.txt', 
                               action_density)
                elif current_action / ansatz_action > 0.2:
                    np.savetxt(output_path + '/stream_9.txt', x_config[2:-2])
                    np.savetxt(output_path + '/streamline_action_dens_9.txt', 
                               action_density)
                else:
                    np.savetxt(output_path + '/stream_10.txt', x_config[2:-2])
                    np.savetxt(output_path + '/streamline_action_dens_10.txt', 
                               action_density)
    tau_writer.close()
    act_writer.close()
