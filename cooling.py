import numpy as np
import utility_monte_carlo as mc
import utility_custom
import random as rnd
import matplotlib.pyplot as plt

#from scipy import optimize
#from scipy import interpolate
#from scipy.misc import derivative

def cooled_monte_carlo(x_potential_minimum,
                   n_lattice,
                   dtau,
                   n_equil,
                   n_mc_sweeps,
                   delta_x,
                   n_points,
                   n_meas,
                   i_cold,
                   n_sweeps_btw_cooling,
                   n_cooling_sweeps,
                    print_last_config):

    #contatore probabilmente inutile
    n_cooling = 0

    if n_mc_sweeps < n_equil:
        print("too few Monte Carlo sweeps/ N_equilib > N_Monte_Carlo")
        return 0

    # Control output filepath
    output_path = r'.\output_data\output_cooled_monte_carlo'
    utility_custom.output_control(output_path)

    # Correlation functions
    x_cor_sums = np.zeros((3, n_points))
    x2_cor_sums = np.zeros((3, n_points))

    # x position along the tau axis
    x_config = mc.initialize_lattice(n_lattice, i_cold)

    #number of instantons
    n_total_instantons_sum = 0
    n2_total_instantons_sum = 0


    # Monte Carlo sweeps: Principal cycle

    # Equilibration cycle
    for i_equil in range(n_equil):
        x_config = mc.metropolis_question(x_config)

    # Rest of the MC sweeps
    with open(output_path + r'\ground_state_histogram.dat', 'wb') as hist_writer:
        for i_mc in range(n_mc_sweeps - n_equil):
            x_config = mc.metropolis_question(x_config)

            np.save(hist_writer, x_config[0:(n_lattice - 1)])
            for k_meas in range(n_meas):
                i_p0 = int((n_lattice - n_points) * rnd.uniform(0., 1.))
                x_0 = x_config[i_p0]
                for i_point in range(n_points):
                    x_1 = x_config[i_p0 + i_point]

                    x_cor_sums[0, i_point] += x_0 * x_1
                    x_cor_sums[1, i_point] += pow(x_0 * x_1, 2)
                    x_cor_sums[2, i_point] += pow(x_0 * x_1, 3)

                    x2_cor_sums[0, i_point] += pow(x_0 * x_1, 2)
                    x2_cor_sums[1, i_point] += pow(x_0 * x_1, 4)
                    x2_cor_sums[2, i_point] += pow(x_0 * x_1, 6)

            # COOLING

            if bool(print_last_config) and (i_mc == (n_mc_sweeps - n_equil - 1)):

                print(f'cooling in the last montecarlo configuration\n')

                x_cold_config_temp = np.copy(x_config)

                for i in range(n_cooling_sweeps):
                    x_cold_config = mc.configuration_cooling(x_cold_config_temp,
                                                                        dtau,
                                                        delta_x * 0.1,  # prova, come nel codice
                                                        x_potential_minimum)

                    # Find instantons and antiinstantons
                    n_instantons, n_anti_instantons,\
                    pos_instantons, pos_anti_instantons = mc.find_instantons(x_cold_config,
                                                                             n_lattice,
                                                                             dtau)

                    n_total_instantons_sum += (n_instantons + n_anti_instantons)
                    n2_total_instantons_sum += pow((n_instantons + n_anti_instantons), 2)

            elif (not bool(print_last_config)) and ((i_mc % n_sweeps_btw_cooling) == 0):

                # expected number of cooled configuration = n_conf/n_sweeps_btw_cooling

                print(f'cooling #{n_cooling} of {(n_mc_sweeps - n_equil) / n_sweeps_btw_cooling}\n'
                      f'in configuration #{i_mc}')

                x_cold_config_temp = np.copy(x_config)
                n_cooling +=1

                for i_cooling in range(n_cooling_sweeps):

                    x_cold_config = mc.configuration_cooling(x_cold_config_temp,
                                                         dtau,
                                                         delta_x*0.1, #prova, come nel codice
                                                         x_potential_minimum)

                    # Find instantons and antiinstantons
                    n_instantons, n_anti_instantons, \
                    pos_instantons, pos_anti_instantons = mc.find_instantons(x_cold_config,
                                                                            n_lattice,
                                                                            dtau)

                    n_total_instantons_sum += (n_instantons + n_anti_instantons)
                    n2_total_instantons_sum += pow((n_instantons + n_anti_instantons),2)


        #Last configuration

        if bool(print_last_config):

            print(f'final configuration: #{i_mc}\n')

            print(f'number of instantons #{n_instantons}\n'
              f'number of anti-instantons #{n_anti_instantons}\n')

            print(f'instantons positions: [{pos_instantons}]\n'
              f'anti-instantons positions: [{pos_anti_instantons}]\n')

            tau = np.linspace(0,dtau*n_lattice,n_lattice+1)

            plt.plot(tau,x_config, color='red')
            plt.plot(tau, x_cold_config, color='blue')
            plt.show()




#    # Evaluate averages and other stuff, maybe we can create a function
#    x_cor_av, x_cor_err = mc.stat_av_var(x_cor_sums[0],
#                                      x2_cor_sums[0],
#                                      n_meas * (n_mc_sweeps - n_equil))
#    x_cor_2_av, x_cor_2_err = mc.stat_av_var(x_cor_sums[1],
#                                          x2_cor_sums[1],
#                                          n_meas * (n_mc_sweeps - n_equil))
#    x_cor_3_av, x_cor_3_err = mc.stat_av_var(x_cor_sums[2],
#                                          x2_cor_sums[2],
#                                          n_meas * (n_mc_sweeps - n_equil))

#    with open(output_path + r'\tau_array.txt', 'w') as tau_writer:
#        np.savetxt(tau_writer,np.linspace(0, n_points * dtau, n_points, False))
#    with open(output_path + r'\average_x_cor.txt', 'w') as av_writer:
#        np.savetxt(av_writer, x_cor_av)
#    with open(output_path + r'\error_x_cor.txt', 'w') as err_writer:
#        np.savetxt(err_writer, x_cor_err)
#    with open(output_path + r'\average_x_cor_2.txt', 'w') as av_writer:
#        np.savetxt(av_writer, x_cor_2_av)
#    with open(output_path + r'\error_x_cor_2.txt', 'w') as err_writer:
#        np.savetxt(err_writer, x_cor_2_err)
#    with open(output_path + r'\average_x_cor_3.txt', 'w') as av_writer:
#        np.savetxt(av_writer, x_cor_3_av)
#    with open(output_path + r'\error_x_cor_3.txt', 'w') as err_writer:
#        np.savetxt(err_writer, x_cor_3_err)

    return 1