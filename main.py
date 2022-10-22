# This is a sample Python script.

import anh_osc_diag
import monte_carlo_ao
import monte_carlo_ao_switching

if __name__ == '__main__':

    print(f'diagonalize\nmonte_carlo_simulation\nfree_energy_int')
    call = input()

    if call in ['diagonalize']:
        anh_osc_diag.anharmonic_oscillator_diag(1.4, 50, 4 * 1.4)
    elif call in ['monte_carlo_simulation']:
        monte_carlo_ao.monte_carlo_ao(1.4, #potential minimum
                   800,  # n_lattice
                   0.05,  # dtau
                   100,  # n_equil
                   20000,  # n_mc_sweeps
                   0.45,  # delta_x
                   20,  # n_point
                   5,  # n_meas
                   False)
    elif call in  ['free_energy_int']:
        monte_carlo_ao_switching.free_energy_anharm(1)
