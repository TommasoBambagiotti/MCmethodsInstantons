import utility_custom
import utility_monte_carlo as mc
import matplotlib.pyplot as plt
import numpy as np

import matplotlib

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.sans-serif': ['Helvetica'],
    'text.usetex': True,
})
plt.style.use('ggplot')

# global variables
filepath = './output_graph'
utility_custom.output_control(filepath)
n_lattice = 800
dtau = 0.05


def print_potential(i_figure):

    # plot setup
    n_eigenvalues = 4
    x_min = -2.5
    x_max = 2.5
    y_min = 0
    y_max = 10

    # create figure
    fig1 = plt.figure(i_figure,
                      facecolor="#fafafa",
                      figsize=(5, 5))

    ax1 = fig1.add_axes((0.13, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    ax1.set_yticks(np.arange(y_min, y_max+1, 1.0))

    # create x_axis and potential
    x_axis = np.linspace(x_min, x_max, 100)
    pot = mc.potential_anh_oscillator(x_axis, 1.4)

    #import #(n_eigenvalues) energy eigenvalues
    count = 0
    en_eigenvalues = np.empty(n_eigenvalues)
    with open('./output_data/output_diag/eigenvalues.txt', 'r') as f_read:
        for line in f_read.readlines():
            if count == (n_eigenvalues):
                continue
            count +=1
            en_eigenvalues[count-1] = float(line)

    # plot
    ax1.plot(x_axis, pot)

    for i in range(n_eigenvalues):
        ax1.hlines(en_eigenvalues[i], x_min, x_max, linestyles='--', color='green')

    ax1.set_xlabel(r'$$x$$')
    ax1.set_ylabel(r'$$V(x)$$')

    # save figure
    fig1.savefig(filepath+'/potential.png', dpi=300)
    plt.show()





def print_ground_state(i_figure):
    filepath_loc = filepath + '/ground_state'
    utility_custom.output_control(filepath_loc)
    utility_custom.clear_folder(filepath_loc)

    fig1 = plt.figure(i_figure,
                      facecolor="#fafafa",
                      figsize=(5, 5))
    ax1 = fig1.add_axes((0.13, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    ax1.set_title('Groundstate distribution $|\psi|^2$',
                  )
    ax1.set_xlabel(f'$x (arb. un.)$')
    ax1.set_ylabel(f'$|\psi|^2$')

    hist = np.loadtxt(
        './output_data/output_monte_carlo/ground_state_histogram.txt',
        float, delimiter=' ')

    x_array = np.loadtxt('./output_data/output_diag/x_position_array.txt',
                         float, delimiter=' ')
    psi_ground_state = np.loadtxt(
        './output_data/output_diag/psi_ground_state.txt',
        float, delimiter=' ')

    psi_simple_model = np.loadtxt(
        './output_data/output_diag/psi_simple_model.txt',
        float, delimiter=' ')

    ax1.set_ylim(0.0, 0.55)
    ax1.hist(hist, 100, (-3, 3), color='blue', label=f'Monte Carlo sim.',
             density=True, histtype='step', linewidth=1)
    ax1.plot(x_array, psi_simple_model, color='red',
             label=f'$\psi$ simple model', linewidth=0.8)
    ax1.plot(x_array, psi_ground_state, color='green',
             label=f'$\psi$ ground state', linewidth=1)

    ax1.legend(
        fontsize=10)

    fig1.savefig(filepath_loc + '/ground_state.png', dpi=300)

    plt.show()


def print_graph_free_energy(i_figure):
    fig1 = plt.figure(i_figure,
                      facecolor="#fafafa",
                      figsize=(5, 5))
    ax1 = fig1.add_axes((0.13, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    ax1.set_title("Free energy F")
    ax1.set_ylabel("F")
    ax1.set_xlabel("Temperature")

    temperature_array = np.loadtxt(
        './output_data/output_monte_carlo_switching/temperature.txt',
        float, delimiter=' ')
    f_energy = np.loadtxt(
        './output_data/output_monte_carlo_switching/free_energy_mc.txt',
        float, delimiter=' ')
    f_energy_err = np.loadtxt(
        './output_data/output_monte_carlo_switching/free_energy_mc_err.txt',
        float, delimiter=' ')

    vir = np.loadtxt(
        './output_data/output_monte_carlo_switching/free_energy_vir.txt',
        float, delimiter=' ')

    vir_err = np.loadtxt(
        './output_data/output_monte_carlo_switching/free_energy_vir_err.txt',
        float, delimiter=' ')

    ax1.errorbar(temperature_array,
                 f_energy,
                 f_energy_err,
                 color='b',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5,
                 label='Monte Carlo switching')

    ax1.errorbar(temperature_array,
                 vir,
                 vir_err,
                 linestyle='',
                 marker='.',
                 color='red',
                 capsize=2.5,
                 elinewidth=0.5,
                 label='Virial theorem')

    temperature_axis = np.loadtxt(
        './output_data/output_diag/temperature.txt', float, delimiter=' ')
    free_energy = np.loadtxt(
        './output_data/output_diag/free_energy.txt', float, delimiter=' ')

    ax1.plot(temperature_axis,
             free_energy,
             color='green',
             label='Free energy')

    ax1.set_ylim(-2.3, -1.6)
    ax1.legend(loc='upper left',
               fontsize=10)
    ax1.set_xscale('log')

    fig1.savefig(filepath + '/free_energy.png', dpi=300)

    plt.show()


def print_configuration(folder, i_figure):
    # check folder
    utility_custom.output_control(filepath + '/' + folder)

    fig = plt.figure(i_figure, facecolor="#fafafa", figsize=(6, 4.5))
    ax = fig.add_axes((0.11, 0.11, 0.8, 0.8), facecolor="#e1e1e1")

    # import data
    tau_array = np.linspace(0, dtau * n_lattice, n_lattice + 1)
    x1_config = np.loadtxt('./output_data/' + folder + '/x1_config.txt')
    x2_config = np.loadtxt('./output_data/' + folder + '/x2_config.txt')

    # plot 1
    ax.set_ylabel(r'$x(\tau)$')
    ax.set_xlabel(r'$\tau$')

    if folder in ['output_cooled_monte_carlo']:
        ax.plot(tau_array, x1_config, color='black', label=r'Monte Carlo')
        ax.plot(tau_array, x2_config, color='green',
                label=r'Cooled Monte Carlo')

    elif folder in ['output_rilm_heating']:
        ax.plot(tau_array, x1_config, color='blue', label=r'RILM',
                linewidth=1.)
        ax.plot(tau_array, x2_config, color='red', label=r'Gaussian heating',
                linewidth=0.8)

    ax.legend()

    plt.savefig(filepath + '/' + folder + '/ ' + 'config.png', dpi=300)
    plt.show()


def print_graph_cor_func(folder, setup, i_figure):
    """

    Parameters
    ----------
    folder :
    """
    utility_custom.output_control(filepath + '/' + folder)

    # axes limits

    x_inf_1 = setup['x_inf_1']
    x_sup_1 = setup['x_sup_1']

    x_inf_2 = setup['x_inf_2']
    x_sup_2 = setup['x_sup_2']

    y_inf_2 = setup['y_inf_2']
    y_sup_2 = setup['y_sup_2']

    # point shift
    cor1_s = setup['cor1_s']
    cor2_s = setup['cor2_s']
    cor3_s = setup['cor3_s']
    cor2_s_fig1 = setup['cor2_s_fig1']

    # Plots

    fig1 = plt.figure(i_figure, facecolor="#fafafa")
    ax1 = fig1.add_axes((0.13, 0.11, 0.8, 0.8), facecolor="#e1e1e1")

    # Set x-axis limit
    ax1.set_xlim(x_inf_1, x_sup_1)

    # Import data
    tau_array = np.loadtxt('./output_data/' + folder + '/tau_array.txt',
                           float, delimiter=' ')
    corr1 = np.loadtxt('./output_data/' + folder + '/average_x_cor_1.txt',
                       float, delimiter=' ')

    corr2 = np.loadtxt('./output_data/' + folder + '/average_x_cor_2.txt',
                       float, delimiter=' ')

    corr3 = np.loadtxt('./output_data/' + folder + '/average_x_cor_3.txt',
                       float, delimiter=' ')

    corr_err1 = np.loadtxt('./output_data/' + folder + '/error_x_cor_1.txt',
                           float, delimiter=' ')

    corr_err2 = np.loadtxt('./output_data/' + folder + '/error_x_cor_2.txt',
                           float, delimiter=' ')

    corr_err3 = np.loadtxt('./output_data/' + folder + '/error_x_cor_3.txt',
                           float, delimiter=' ')

    tau_array_2 = np.loadtxt('./output_data/output_diag/tau_array.txt',
                             float, delimiter=' ')

    corr1_d = np.loadtxt('./output_data/output_diag/corr_function.txt',
                         float, delimiter=' ')

    corr2_d = np.loadtxt('./output_data/output_diag/corr_function2.txt',
                         float, delimiter=' ')

    corr3_d = np.loadtxt('./output_data/output_diag/corr_function3.txt',
                         float, delimiter=' ')
    # Plot
    ax1.plot(tau_array_2[0:61],
             corr1_d[0:61],
             color='b',
             linewidth=0.8,
             linestyle=':')

    ax1.plot(tau_array_2[0:61],
             corr2_d[0:61],
             color='r',
             linewidth=0.8,
             linestyle=':')

    ax1.plot(tau_array_2[0:61],
             corr3_d[0:61],
             color='g',
             linewidth=0.8,
             linestyle=':')

    ax1.errorbar(tau_array[:tau_array.size - cor1_s],
                 corr1[:corr1.size - cor1_s],
                 corr_err1[:corr1.size - cor1_s],
                 color='b',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5,
                 label=r'$<x(\tau)x(0)>$')

    ax1.errorbar(tau_array[:tau_array.size - cor2_s_fig1],
                 corr2[:corr2.size - cor2_s_fig1],
                 corr_err2[:corr2.size - cor2_s_fig1],
                 color='red',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5,
                 label=r'$<x^2(\tau)x^2(0)>$')

    ax1.errorbar(tau_array[:tau_array.size - cor3_s],
                 corr3[:corr3.size - cor3_s],
                 corr_err3[:corr3.size - cor3_s],
                 color='green',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5,
                 label=r'$<x^3(\tau)x^3(0)>$')

    # Labels
    ax1.set_ylabel(r'$<x^n(\tau)x^n(0)>$', rotation='vertical',
                   fontsize=12)

    ax1.set_xlabel(r'$\tau$', fontsize=12)

    # Legend
    ax1.legend(loc='upper right', fontsize=10)
    # Save plot
    fig1.savefig(filepath + '/' + folder + '/x_corr.png', dpi=300)

    # Plot 2

    fig2 = plt.figure(i_figure + 1, facecolor="#fafafa")

    ax2 = fig2.add_axes((0.13, 0.11, 0.8, 0.8), facecolor="#e1e1e1")

    ax2.set_xlim(x_inf_2, x_sup_2)
    ax2.set_ylim(y_inf_2, y_sup_2)

    dcorr1 = np.loadtxt('./output_data/' + folder + '/average_der_log_1.txt',
                        float, delimiter=' ')

    dcorr2 = np.loadtxt('./output_data/' + folder + '/average_der_log_2.txt',
                        float, delimiter=' ')

    dcorr3 = np.loadtxt('./output_data/' + folder + '/average_der_log_3.txt',
                        float, delimiter=' ')

    dcorr_err1 = np.loadtxt('./output_data/' + folder + '/error_der_log_1.txt',
                            float, delimiter=' ')

    dcorr_err2 = np.loadtxt('./output_data/' + folder + '/error_der_log_2.txt',
                            float, delimiter=' ')

    dcorr_err3 = np.loadtxt('./output_data/' + folder + '/error_der_log_3.txt',
                            float, delimiter=' ')

    dcorr1_d = np.loadtxt(
        './output_data/output_diag/av_der_log_corr_funct.txt',
        float, delimiter=' ')

    dcorr2_d = np.loadtxt(
        './output_data/output_diag/av_der_log_corr_funct2.txt',
        float, delimiter=' ')

    dcorr3_d = np.loadtxt(
        './output_data/output_diag/av_der_log_corr_funct3.txt',
        float, delimiter=' ')

    ax2.plot(tau_array_2[0:61],
             dcorr1_d[0:61],
             color='b',
             linewidth=0.8,
             linestyle=':')

    ax2.plot(tau_array_2[0:61],
             dcorr2_d[0:61],
             color='r',
             linewidth=0.8,
             linestyle=':')

    ax2.plot(tau_array_2[0:61],
             dcorr3_d[0:61],
             color='g',
             linewidth=0.8,
             linestyle=':')

    ax2.errorbar(tau_array[0:tau_array.size - cor1_s - 1],
                 dcorr1[:dcorr1.size - cor1_s],
                 dcorr_err1[:dcorr1.size - cor1_s],
                 color='b',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5,
                 label=r'$<x(\tau)x(0)>$')

    ax2.errorbar(tau_array[0:tau_array.size - cor2_s - 1],
                 dcorr2[:dcorr2.size - cor2_s],
                 dcorr_err2[:dcorr2.size - cor2_s],
                 color='red',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5,
                 label=r'$<x^2(\tau)x^2(0)>$')

    ax2.errorbar(tau_array[0:tau_array.size - cor3_s - 1],
                 dcorr3[:dcorr3.size - cor3_s],
                 dcorr_err3[:dcorr3.size - cor3_s],
                 color='green',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5,
                 label=r'$<x^3(\tau)x^3(0)>$')
    # Legend
    ax2.legend(loc='upper right', fontsize=10)

    # Labels
    ax2.set_ylabel(r'$d(log<x^n(\tau)x^n(0)>)/d\tau $',
                   rotation='vertical',
                   fontsize=12)

    ax2.set_xlabel(r'$\tau$', fontsize=12)

    # Save figure
    fig2.savefig(filepath + '/' + folder + '/der_corr.png', dpi=300)

    # plt.show()


def print_density(i_figure):

    # create new figure and plot density
    fig = plt.figure(i_figure, facecolor="#fafafa", figsize=(5, 5))
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    #ax.set_title('Instanton density')

    potential_minima = np.loadtxt(
        'output_data/output_cooled_monte_carlo/potential_minima.txt',
        delimiter=' ')

    n_cooling = np.loadtxt(
        './output_data/output_cooled_monte_carlo/n_cooling.txt',
        delimiter=' ')

    # only for #potential_minima = 4
    colors = ['red', 'green', 'orange', 'blue']

    i = 1
    for pot in np.nditer(potential_minima):
        n_instantons = np.loadtxt(
            f'./output_data/output_cooled_monte_carlo/n_instantons_{i}.txt')
        n_instantons_err = np.loadtxt(
            f'./output_data/output_cooled_monte_carlo/n_instantons_{i}_err.txt')

        ax.errorbar(n_cooling,
                    n_instantons,
                    n_instantons_err,
                    fmt='.',
                    capsize=2.5,
                    elinewidth=0.5,
                    color=colors[pot],
                    label=f'$\eta = {pot}$')

        s0 = 4 / 3 * pow(pot, 3)

        loop_1 = 8 * pow(pot, 5 / 2) \
                 * pow(2 / np.pi, 1 / 2) * np.exp(-s0)

        loop_2 = 8 * pow(pot, 5 / 2) \
                 * pow(2 / np.pi, 1 / 2) * np.exp(-s0 - 71 / (72 * s0))

        ax.hlines([loop_1, loop_2], 0, n_cooling[-1], color='green',
                  linestyle=['dashed', 'solid'], linewidth=0.5)

        i += 1

    # Labels
    ax.set_xlabel(r'$N_{cool}$', labelpad=0)
    ax.set_ylabel(r'$N_{tot}  / \beta $')

    # Legend
    ax.legend()

    # Log scale
    ax.set_xscale('log')

    ax.set_yscale('log')

    # ax.ticklabel_format(axis = 'y', style = 'plain')

    # current_values = ax.get_yticks()
    # ax.set_yticklabels(['{:.2f}'.format(x) for x in current_values])

    fig.set_size_inches(w=7, h=4)
    fig.savefig(filepath + '/n_istantons.png', dpi=300)

    fig2 = plt.figure(i_figure + 1, facecolor="#fafafa", figsize=(5, 5))
    ax2 = fig2.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    i = 1

    for pot in np.nditer(potential_minima):
        action = np.loadtxt(
            f'./output_data/output_cooled_monte_carlo/action_{i}.txt',
            float,
            delimiter=' ')
        action_err = np.loadtxt(
            f'./output_data/output_cooled_monte_carlo/action_err_{i}.txt',
            float,
            delimiter=' ')

        ax2.errorbar(n_cooling,
                     action,
                     action_err,
                     fmt='.',
                     capsize=2.5,
                     elinewidth=0.5,
                     color=colors[pot],
                     label=f'$\eta = {pot}$')

        s0 = 4 / 3 * pow(pot, 3)

        ax2.hlines(s0, 0, n_cooling[-1], color='green',
                   linestyle='dashed', linewidth=0.8)

        i += 1

    ax2.legend()
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylabel(r'S / N_{tot}')
    ax2.set_xlabel(r'N_{cool}', labelpad=0)
    fig2.set_size_inches(w=7, h=4)
    # current_values = ax2.gca().get_yticks()
    # ax2.gca().set_yticklabels([{'0.2f'}.format(x) for x in current_values])

    fig2.savefig(filepath + '/action.png', dpi=300)

    plt.show()


def print_zcr_hist(i_figure):
    fig = plt.figure(i_figure, facecolor="#fafafa", figsize=(6, 4.5))
    ax = fig.add_axes((0.15, 0.13, 0.8, 0.8), facecolor="#e1e1e1")

    ax.set_xlabel(r'$\Delta\tau_{zcr}$')
    ax.set_ylabel(r'$n_{IA}(\tau_{zcr})$')

    zcr = np.loadtxt('./output_data/output_rilm/zcr_hist.txt', float,
                     delimiter=' ')

    zcr_cooling = np.loadtxt(
        './output_data/output_cooled_monte_carlo/zero_crossing/zcr_cooling.txt',
        float, delimiter=' ')

    zcr_int = np.loadtxt('./output_data/output_iilm/iilm/zcr_hist.txt',
                         float, delimiter=' ')

    ax.hist(zcr, 40, (0., 4.), histtype='step',
            color='red', linewidth=1.2,
            label='RILM')
    ax.hist(zcr_int, 40, (0., 4.), histtype='step',
            color='orange', linewidth=1.2,
            label='Core repulsion')  # , density='True')
    ax.hist(zcr_cooling, 40, (0., 4.), histtype='step',
            label='Monte carlo cooling',
            color='blue', linewidth=1.2)  # , density='True')

    ax.legend()

    fig.savefig(filepath + '/zcr_histogram.png', dpi=300)

    # plt.show()


def print_tau_centers(i_figure):
    fig = plt.figure(i_figure, facecolor="#fafafa", figsize=(6, 4.5))
    ax = fig.add_axes((0.11, 0.11, 0.8, 0.8), facecolor="#e1e1e1")

    n_conf = np.loadtxt('./output_data/output_iilm/iilm/n_conf.txt',
                        float, delimiter=' ')
    ax.set_xlabel(r'$N_{conf}$')
    ax.set_ylabel(r'$\tau_{IA}$')
    for n in range(12):
        tau = np.loadtxt(f'./output_data/output_iilm/iilm/center_{n + 1}.txt',
                         float, delimiter=' ')

        if (n % 2) == 0:
            handle_inst, = ax.plot(n_conf, tau, color='blue',
                                   linewidth=0.4,
                                   label=r'Instanton center')
        else:
            handle_a_inst, = ax.plot(n_conf, tau, color='red',
                                     linewidth=0.4,
                                     label=r'Anti-instanton center')

    ax.legend(loc='upper right', handles=[handle_inst,
                                          handle_a_inst])

    ax.set_ylim(0, 47)
    fig.savefig(filepath + '/iilm_config.png', dpi=300)
    # plt.show()


def print_switch_density(i_figure):

    # you have to import delta_e to use this function

    fig = plt.figure(i_figure, facecolor="#fafafa")
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    potential_minima = np.loadtxt(
        'output_data/output_cooled_monte_carlo/potential_minima.txt',
        delimiter=' ')

    dens = np.zeros(potential_minima.size)
    dens_err = np.zeros(potential_minima.size)

    i = 1
    for pot in np.nditer(potential_minima):
        j = 0
        with open(
                f'output_data/output_cooled_monte_carlo/n_instantons_{i}.txt',
                'r') \
                as reader:
            for line in reader:

                if j == 8:
                    line = reader.readline()
                    dens[i - 1] = float(line)
                j += 1
        j = 0
        with open(
                f'output_data/output_cooled_monte_carlo/n_instantons_{i}_err.txt',
                'r') \
                as reader:
            for line in reader:
                if j == 8:
                    dens_err[i - 1] = float(line)
                j += 1

        i += 1
    ax.errorbar(potential_minima,
                dens,
                dens_err,
                color='blue',
                marker='s',
                linestyle='',
                capsize=2.5,
                elinewidth=0.5,
                markersize=4.,
                label='cooling')

    potential = np.linspace(0.1, 2.0)

    potential_e = np.empty(40)
    for i in range(np.size(potential_e)):
        potential_e[i] = 0.05*i

    # import energy eigenvalues
    delta_e = np.loadtxt('output_data/output_monte_carlo_density_switching/delta_e.txt') / 2

    action = 4 / 3 * np.power(potential, 3)

    loop_1 = 8 * np.sqrt(2 / np.pi) \
             * np.power(potential, 5 / 2) * np.exp(-action)

    loop_2 = loop_1 * np.exp(-71 / 72 / action)

    # 1loop plot
    ax.plot(potential, loop_1,
            linestyle='--', color='green', label='1-loop')

    # 2loop plot
    ax.plot(potential, loop_2,
            linestyle='-', color='green', label='2-loop')
    # deltaE/2 plot
    ax.plot(potential_e, delta_e, linestyle='-', color='black',
            label=r'$\Delta E / 2$', linewidth=1)

    # load minima and densities
    potential_minima = np.loadtxt(
        'output_data/output_monte_carlo_density_switching/potential_minima.txt',
        delimiter=' ')

    dens = np.loadtxt(
        'output_data/output_monte_carlo_density_switching/total_density.txt')

    dens_err = np.loadtxt(
        'output_data/output_monte_carlo_density_switching/total_density_err.txt')

    ax.errorbar(
        potential_minima,
        dens,
        dens_err,
        color='red',
        marker='s',
        linestyle='',
        capsize=2.5,
        elinewidth=0.5,
        markersize=4.,
        label='Monte Carlo')

    # create new legend
    ax.legend()

    # axes limits
    ax.set_xlim(0.0, 1.9)
    ax.set_ylim(0.03, 2.)

    # log-scale
    ax.set_yscale('log')

    # labels
    ax.set_xlabel(r'$$f$$')
    ax.set_ylabel(r'$$N_{tot} / \beta$$')

    # save and show
    fig.savefig(filepath + '/density.png', dpi=300)

    plt.show()


def print_iilm(i_figure):
    fig = plt.figure(i_figure, facecolor="#fafafa", figsize=(6, 4.5))

    ax = fig.add_axes((0.11, 0.11, 0.8, 0.8), facecolor="#e1e1e1")

    ax.set_xlabel(r'$\Delta\tau_{IA}$')
    ax.set_ylabel(r'$S_{int}\slash S_{0}$')

    tau_ia = np.loadtxt(
        './output_data/output_iilm/streamline/delta_tau_ia.txt',
        float, delimiter=' ')

    zcr = np.loadtxt('./output_data/output_rilm/zcr_hist.txt', float,
                     delimiter=' ')
    zcr_cooling = np.loadtxt(
        './output_data/output_cooled_monte_carlo/zero_crossing/zcr_cooling.txt',
        float, delimiter=' ')

    hist_1, _ = np.histogram(zcr, 40, range=(0., 4.))
    hist_2, _ = np.histogram(zcr_cooling, 40, range=(0., 4.))

    # bisogna implementare un modo per capire già quanto vale il potentiale?
    # tipo costruire l'istogramma già nella funzione zero_...
    action_ia = -np.log(hist_2[0:10] / hist_1[0:10]) / (
            4 / 3 * np.power(1.4, 3))

    act_int = np.loadtxt(
        './output_data/output_iilm/streamline/streamline_action_int.txt',
        float, delimiter=' ')

    array_ia = np.loadtxt('./output_data/output_iilm/streamline/array_ia.txt',
                          float, delimiter=' ')

    array_ia_0 = np.loadtxt(
        './output_data/output_iilm/streamline/array_ia_0.txt',
        float, delimiter=' ')

    array_int = np.loadtxt(
        './output_data/output_iilm/streamline/array_int.txt',
        float, delimiter=' ')

    array_int_core = np.loadtxt(
        './output_data/output_iilm/streamline/array_int_core.txt',
        float, delimiter=' ')

    array_int_zero_cross = np.loadtxt(
        './output_data/output_iilm/streamline/array_int_zero_cross.txt',
        float, delimiter=' ')

    ax.scatter(tau_ia,
               act_int,
               marker='^',
               color='green',
               label=r'Streamline',
               s=25
               )

    ax.plot(array_ia,
            array_int,
            color='blue',
            label=r'Sum ansatz'
            )

    ax.plot(array_ia,
            array_int_core,
            color='orange',
            label=r'Core repulsion'
            )

    ax.scatter(array_ia_0,
               array_int_zero_cross,
               color='red',
               marker='s',
               label=r'Sum ansatz zero crossing',
               s=25
               )

    ax.scatter(np.linspace(0.1, 1.0, 10, False),
               action_ia,
               marker='s',
               color='cyan',
               label=r'Monte Carlo cooling',
               s=25
               )

    ax.legend()

    ax.set_ylim(-2.1, 0.5)
    ax.set_xlim(-0.05, 2.05)

    fig.savefig(filepath + '/iilm.png', dpi=300)