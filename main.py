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
from utility_custom import graphical_ui
from utility_custom import cor_plot_setup

if __name__ == '__main__':

    stop_exec = 0

    while not stop_exec:

        call = graphical_ui('main')

        if call in ['0']:

            anh_osc_diag.anharmonic_oscillator_diag(50)

        elif call in ['1']:
            monte_carlo_ao.monte_carlo_ao(
                800,  # n_lattice
                100,  # n_equil
                100000,  # n_mc_sweeps
                30,  # n_point
                5,  # n_meas
                False)

        elif call in ['2']:
            monte_carlo_ao_switching. \
                free_energy_anharm(1,
                                   40.0,
                                   100,
                                   100000,
                                   20,
                                   False)

        elif call in ['3']:
            monte_carlo_ao_cooling.cooled_monte_carlo(
                800,  # n_lattice
                100,  # n_equil
                150000,  # n_mc_sweeps
                30,  # n_points
                5,  # n_meas
                False,  # cold start
                10,
                # number of mc configurations between successive cooled configurations
                100)  # number of mc cooling sweeps

        elif call in ['4']:
            monte_carlo_ao_cooling_density. \
                cooled_monte_carlo_density(800,
                                           100,
                                           100000,
                                           False,
                                           50,
                                           200,
                                           4)

        elif call in ['5']:
            monte_carlo_ao_density_switching. \
                instantons_density_switching(80,
                                             100,
                                             50000,
                                             20,
                                             4,
                                             1.3,
                                             delta_x=0.5)

        elif call in ['6']:
            random_instanton_monte_carlo. \
                random_instanton_liquid_model(800,
                                              120000,
                                              30,
                                              5,
                                              10)

        elif call in ['7']:
            random_instanton_monte_carlo_heating. \
                random_instanton_liquid_model_heating(800,
                                                      100000,
                                                      30,
                                                      5,
                                                      10)

        elif call in ['8']:
            streamline_iilm.streamline_method_iilm(1.8,
                                                   0.001,
                                                   50,
                                                   800,
                                                   70001)

        elif call in ['9']:
            instanton_interactive_liquid_model. \
                inst_int_liquid_model(800,
                                      100000,
                                      30,
                                      5,
                                      0.3,
                                      3.0,
                                      0.5)

        elif call in ['10']:
            zero_crossing_dist_cooling. \
                zero_crossing_cooling_density(800,
                                              100,
                                              600000,
                                              False,
                                              5,
                                              10,
                                              600000)

        elif call in ['11']:

            stop_exec_plot = 0
            i_figure = 1

            while not stop_exec_plot:

                call2 = graphical_ui('plots')

                if call2 in ['a']:
                    graph_print.print_graph_cor_func('output_monte_carlo',
                                                     cor_plot_setup(call2),
                                                     i_figure)
                    i_figure += 2
                elif call2 in ['b']:
                    graph_print.print_ground_state(i_figure)
                    i_figure += 1
                elif call2 in ['c']:
                    graph_print.print_graph_free_energy(i_figure)
                    i_figure += 1
                elif call2 in ['d']:
                    graph_print.print_graph_cor_func(
                        'output_cooled_monte_carlo',
                        cor_plot_setup(call2),
                        i_figure)
                    i_figure += 2
                elif call2 in ['e']:
                    graph_print.print_density(i_figure)
                    i_figure += 2
                elif call2 in ['f']:
                    graph_print.print_switch_density(i_figure)
                    i_figure += 1
                elif call2 in ['g']:
                    graph_print.print_configuration(
                        'output_cooled_monte_carlo',
                        i_figure)
                    i_figure += 2
                elif call2 in ['h']:
                    graph_print.print_graph_cor_func('output_rilm',
                                                     cor_plot_setup(call2),
                                                     i_figure)
                    i_figure += 2
                elif call2 in ['i']:
                    graph_print.print_graph_cor_func('output_rilm_heating',
                                                     cor_plot_setup(call2),
                                                     i_figure)
                    i_figure += 2
                elif call2 in ['j']:
                    graph_print.print_configuration('output_rilm_heating',
                                                    i_figure)
                    i_figure += 1
                elif call2 in ['k']:
                    graph_print.print_graph_cor_func('output_iilm/iilm',
                                                     cor_plot_setup(call2),
                                                     i_figure)
                    i_figure += 2
                elif call2 in ['l']:
                    graph_print.print_zcr_hist(i_figure)
                    i_figure += 1
                elif call2 in ['m']:
                    graph_print.print_tau_centers(i_figure)
                    i_figure += 1
                elif call2 in ['n']:
                    graph_print.print_iilm(i_figure)
                    i_figure += 1
                elif call2 in ['o']:
                    graph_print.print_potential(i_figure)
                    i_figure +=1
                elif call2 in ['p']:
                    graph_print.print_streamline(i_figure)
                    i_figure += 2
                elif call2 in ['exit']:
                    stop_exec_plot = 1
                else:
                    print('invalid command, try again\n')

        elif call in ['exit']:
            stop_exec = 1

        else:
            print('invalid command, try again\n')