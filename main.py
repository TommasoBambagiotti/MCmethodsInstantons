# This is a sample Python script.

import anh_osc_diag
import monte_carlo_ao
import monte_carlo_ao_switching
import monte_carlo_ao_cooling
import monte_carlo_ao_cooling_density
import monte_carlo_ao_density_switching
import zero_crossing_dist_cooling
import random_instanton_monte_carlo
import random_instanton_monte_carlo_heating
import streamline_iilm
import instanton_interactive_liquid_model
import graph_print



if __name__ == '__main__':

    stop_exec = 0

    while not stop_exec:

        print(f'diagonalize\n'
              f'monte_carlo\n'
              f'free\n'
              f'cooling\n'
              f'cool0\n'
              f'density\n'
              f'density_switch\n'
              f'rilm\n'
              f'heating\n'
              f'streamline\n'
              f'iilm\n'
              f'print\n'
              f'exit\n')

        call = input()

        if call in ['diagonalize']:
            anh_osc_diag.anharmonic_oscillator_diag(1.4, 50, 4 * 1.4)
        elif call in ['monte_carlo']:
            monte_carlo_ao.monte_carlo_ao(
                                        800,  # n_lattice
                                        100,  # n_equil
                                        100000,  # n_mc_sweeps
                                        30,  # n_point
                                        5,  # n_meas
                                        False)
        elif call in ['free']:
            monte_carlo_ao_switching.\
                free_energy_anharm(4,
                                   40.0,
                                   100,
                                   100000,
                                   20,
                                   False)
            
            
        elif call in ['cooling']:
            monte_carlo_ao_cooling.cooled_monte_carlo(
                                        800,  # n_lattice
                                        100,  # n_equil
                                        150000,  # n_mc_sweeps
                                        20,  # n_points
                                        5,  # n_meas
                                        False,  # cold start
                                        10,  # number of mc configurations between successive cooled configurations
                                        200)  # number of mc cooling sweeps

        elif call in ['density']:
            monte_carlo_ao_cooling_density.cooled_monte_carlo_density(800,
                                                                      100,
                                                                      100000,
                                                                      False,
                                                                      20,
                                                                      10,
                                                                      12,
                                                                      0.5)

        elif call in ['density_switch']:
            monte_carlo_ao_density_switching.\
                instantons_density_switching(100,
                                             100,
                                             200000,
                                             20,
                                             4,
                                             1.3)
                
        elif call in ['rilm']:
            random_instanton_monte_carlo.\
            random_instanton_liquid_model(800,
                                          120000, 
                                          20,
                                          5,
                                          10)
            
        elif call in ['heating']:
            random_instanton_monte_carlo_heating.\
                random_instanton_liquid_model_heating(800,
                                                      100000,
                                                      20,
                                                      5, 
                                                      10)

        elif call in ['streamline']:
            streamline_iilm.streamline_method_iilm(1.8,
                                                    0.001,
                                                    50,
                                                    800,
                                                    70001)
            

        elif call in ['iilm']:
            instanton_interactive_liquid_model.inst_int_liquid_model(800,
                                                                     100000,
                                                                     30,
                                                                     5,
                                                                     0.3,
                                                                     3.0,
                                                                     0.5)

        elif call in ['cool0']:
            zero_crossing_dist_cooling.\
                zero_crossing_cooling_density(800,
                                              100,
                                              600000,
                                              False,
                                              5,
                                              10)
                
        elif call in ['print']:
            call2 = input()
            if call2 in ['a']:
                graph_print.print_graph_free_energy()
            elif call2 in ['b']:
                graph_print.print_graph_cool_conf()
            elif call2 in ['c']:
                graph_print.print_graph('output_monte_carlo')
            elif call2 in ['d']:
                graph_print.print_graph('output_cooled_monte_carlo')
            elif call2 in ['e']:
                graph_print.print_density()
            elif call2 in ['f']:
                graph_print.print_graph('output_rilm')
            elif call2 in ['g']:
                graph_print.print_graph('output_rilm_heating')
            elif call2 in ['h']:
                graph_print.print_graph_heat()
            elif call2 in ['i']:
                graph_print.print_iilm()
            elif call2 in ['k']:
                graph_print.print_stream()
            elif call2 in ['j']:
                graph_print.print_zcr_hist()
            elif call2 in ['l']:
                graph_print.print_rilm_conf()
            elif call2 in ['m']:
                graph_print.print_tau_centers()
            elif call2 in ['n']:
                graph_print.print_graph('output_iilm/iilm')
            elif call2 in ['o']:
                graph_print.print_cool_density()
                
        elif call in ['exit']:
            stop_exec = 1

        else:
            print('invalid command, try again\n')
