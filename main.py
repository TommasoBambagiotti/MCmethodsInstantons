# This is a sample Python script.
import cProfile

import anh_osc_diag
import monte_carlo_ao
import monte_carlo_ao_switching
import monte_carlo_ao_cooling
import monte_carlo_ao_cooling_density
import random_instanton_monte_carlo
import random_instanton_monte_carlo_heating
import streamline_iilm
import instanton_interactive_liquid_model
import graph_print



if __name__ == '__main__':

    stop_exec = 0

    while not stop_exec:

        print(f'diagonalize\n'
              f'monte_carlo_simulation\n'
              f'free_energy_int\n'
              f'cooling\n'
              f'density\n'
              f'rilm\n'
              f'heating\n'
              f'streamline\n'
              f'iilm\n'
              f'print\n'
              f'exit\n')

        call = input()

        if call in ['diagonalize']:
            anh_osc_diag.anharmonic_oscillator_diag(1.4, 50, 4 * 1.4)
        elif call in ['monte_carlo_simulation']:
            monte_carlo_ao.monte_carlo_ao(
                                        800,  # n_lattice
                                        100,  # n_equil
                                        40000,  # n_mc_sweeps
                                        20,  # n_point
                                        5,  # n_meas
                                        False)
        elif call in ['free_energy_int']:
            monte_carlo_ao_switching.\
                free_energy_anharm(1,8.0,100,50000,20,False)
            
        elif call in ['cooling']:
            monte_carlo_ao_cooling.cooled_monte_carlo(
                                        200,  # n_lattice
                                        100,  # n_equil
                                        50000,  # n_mc_sweeps
                                        20,  # n_points
                                        5,  # n_meas
                                        False,  # cold start
                                        20,  # number of mc configurations between successive cooled configurations
                                        200)  # number of mc cooling sweeps

        elif call in ['density']:
            monte_carlo_ao_cooling_density.cooled_monte_carlo_density(800,
                                                                      200,
                                                                      100000,
                                                                      False,
                                                                      10,
                                                                      10,
                                                                      1)

        elif call in ['rilm']:
            random_instanton_monte_carlo.\
            random_instanton_liquid_model(800,
                                          100000, 
                                          20,
                                          5)
            
        elif call in ['heating']:
            random_instanton_monte_carlo_heating.\
                random_instanton_liquid_model_heating(800,
                                                      5000,
                                                      20,
                                                      5, 
                                                      10)

        elif call in ['streamline']:
            streamline_iilm.print_sum_ansatz_ia(800)
            streamline_iilm.streamline_method_iilm(1.8,
                                                   0.001,
                                                   50,
                                                   70000)

        elif call in ['iilm']:
            instanton_interactive_liquid_model.inst_int_liquid_model(800,
                                                                     100000,
                                                                     0,
                                                                     0,
                                                                     0.3,
                                                                     3.0,
                                                                     0.1)

        elif call in ['print']:
            call2 = input()
            if call2 in ['a']:
                graph_print.print_graph_free_energy()
            elif call2 in ['b']:
                graph_print.print_graph_cool_conf()
            elif call2 in ['c']:
                graph_print.print_graph_mc()
            elif call2 in ['d']:
                graph_print.print_graph_cool()
            elif call2 in ['e']:
                graph_print.print_density()
            elif call2 in ['f']:
                graph_print.print_graph_rilm()
            elif call2 in ['g']:
                graph_print.print_graph_rilm_heating()
            elif call2 in ['h']:
                graph_print.print_graph_heat()
            elif call2 in ['i']:
                graph_print.print_iilm()
            elif call2 in ['l']:
                graph_print.print_stream()
            elif call2 in ['k']:
                graph_print.print_zcr_hist()
            elif call2 in ['j']:
                graph_print.print_rilm_conf()

        elif call in ['exit']:
            stop_exec = 1

        else:
            print('invalid command, try again\n')
