# This is a sample Python script.
import anh_osc_diag
import monte_carlo_ao
import monte_carlo_ao_switching
import monte_carlo_ao_cooling

if __name__ == '__main__':

    stop_exec = 0

    while not stop_exec:

        print(f'diagonalize\n'
              f'monte_carlo_simulation\n'
              f'free_energy_int\n'
              f'cooling\n'
              f'exit\n')

        call = input()

        if call in ['diagonalize']:
            anh_osc_diag.anharmonic_oscillator_diag(1.4, 50, 4 * 1.4)
        elif call in ['monte_carlo_simulation']:
            monte_carlo_ao.monte_carlo_ao(
                                        800,  # n_lattice
                                        100,  # n_equil
                                        10000,  # n_mc_sweeps
                                        20,  # n_point
                                        5,  # n_meas
                                        False)
        elif call in ['free_energy_int']:
            monte_carlo_ao_switching.free_energy_anharm(1)
        elif call in ['cooling']:
            monte_carlo_ao_cooling.cooled_monte_carlo(
                                        800,  # n_lattice
                                        100,  # n_equil
                                        1000,  # n_mc_sweeps
                                        20,  # n_points
                                        5,  # n_meas
                                        False,  # cold start
                                        20,  # number of mc configurations between successive cooled configurations
                                        50)  # number of mc cooling sweeps
        elif call in ['exit']:
            stop_exec = 1

        else:
            print('invalid command, try again\n')

