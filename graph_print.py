import numpy as np
import matplotlib.pyplot as plt
import utility_custom


filepath = './output_graph'
utility_custom.output_control(filepath)


def print_graph_free_energy():
    """

    """
    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig1 = plt.figure(1, facecolor="#f1f1f1")
    ax1 = fig1.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    ax1.set_xlabel(r"$T$")
    ax1.set_ylabel(r"$F$")

    temperature_array = np.loadtxt('./output_data/output_monte_carlo_switching/temperature.txt',
                                   float, delimiter=' ')
    f_energy = np.loadtxt('./output_data/output_monte_carlo_switching/free_energy_mc.txt',
                          float, delimiter=' ')
    f_energy_err = np.loadtxt('./output_data/output_monte_carlo_switching/free_energy_mc_err.txt',
                              float, delimiter=' ')

    vir = np.loadtxt('./output_data/output_monte_carlo_switching/free_energy_vir.txt',
                     float, delimiter=' ')

    vir_err = np.loadtxt('./output_data/output_monte_carlo_switching/free_energy_vir_err.txt',
                         float, delimiter=' ')

    ax1.errorbar(temperature_array,
                 f_energy,
                 f_energy_err,
                 color='b',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    ax1.errorbar(temperature_array,
                 vir,
                 vir_err,
                 linestyle='',
                 marker='.',
                 color='red',
                 capsize=2.5,
                 elinewidth=0.5)

    ax1.set_xscale('log')

    temperature_axis = np.loadtxt(
        './output_data/output_diag/temperature.txt', float, delimiter=' ')
    free_energy = np.loadtxt(
        './output_data/output_diag/free_energy.txt', float, delimiter=' ')

    ax1.plot(temperature_axis,
             free_energy,
             color='green')

    fig1.savefig(filepath + '/free_energy.png', dpi=300)

    plt.show()


def print_graph(folder):
    """

    Parameters
    ----------
    folder :
    """
    utility_custom.output_control(filepath + '/' + folder)

    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig1 = plt.figure(1, facecolor="#f1f1f1")

    ax1 = fig1.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

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

    ax1.errorbar(tau_array,
                 corr1,
                 corr_err1,
                 color='b',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    ax1.errorbar(tau_array,
                 corr2,
                 corr_err2,
                 color='red',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    ax1.errorbar(tau_array,
                 corr3,
                 corr_err3,
                 color='green',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    fig1.savefig(filepath + '/' + folder + '/x_corr.png', dpi=300)

    fig2 = plt.figure(2, facecolor="#f1f1f1")

    ax2 = fig2.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

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

    dcorr1_d = np.loadtxt('./output_data/output_diag/av_der_log_corr_funct.txt',
                          float, delimiter=' ')

    dcorr2_d = np.loadtxt('./output_data/output_diag/av_der_log_corr_funct2.txt',
                          float, delimiter=' ')

    dcorr3_d = np.loadtxt('./output_data/output_diag/av_der_log_corr_funct3.txt',
                          float, delimiter=' ')

    ax2.set_ylim(-1.0, 10.0)

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

    ax2.errorbar(tau_array[0:-1],
                 dcorr1,
                 dcorr_err1,
                 color='b',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    ax2.errorbar(tau_array[0:-1],
                 dcorr2,
                 dcorr_err2,
                 color='red',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    ax2.errorbar(tau_array[0:-1],
                 dcorr3,
                 dcorr_err3,
                 color='green',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    ax2.set_ylim(-0.1, 5.0)

    fig2.savefig(filepath + '/' + folder + '/der_corr.png', dpi=300)

    plt.show()


def print_density():
    """

    """
    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig3 = plt.figure(3, facecolor="#f1f1f1")
    ax3 = fig3.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    potential_minima = np.loadtxt(
        'output_data/output_cooled_monte_carlo/potential_minima.txt',
        delimiter=' ')

    n_cooling = np.loadtxt(
        './output_data/output_cooled_monte_carlo/n_cooling.txt',
        delimiter=' ')

    i = 1
    for pot in np.nditer(potential_minima):
        n_instantons = np.loadtxt(
            f'./output_data/output_cooled_monte_carlo/n_instantons_{i}.txt')
        n_instantons_err = np.loadtxt(
            f'./output_data/output_cooled_monte_carlo/n_instantons_{i}_err.txt')

        ax3.errorbar(n_cooling,
                      n_instantons,
                      n_instantons_err,
                      fmt='.',
                      capsize=2.5,
                      elinewidth=0.5,
                      label=f'$\eta = {pot}$')

        s0 = 4 / 3 * pow(pot, 3)

        loop_1 = 8 * pow(pot, 5 / 2) \
            * pow(2 / np.pi, 1/2) * np.exp(-s0)

        loop_2 = 8 * pow(pot, 5 / 2) \
            * pow(2 / np.pi, 1/2) * np.exp(-s0 - 71 / (72 * s0))

        ax3.hlines([loop_1, loop_2], 0, n_cooling[-1], color='green',
                    linestyle=['dashed', 'solid'], linewidth=0.5)

        i += 1

    ax3.legend()
    ax3.set_xscale('log')

    ax3.set_yscale('log')

    fig3.savefig(filepath + '/n_istantons.png', dpi=300)

    fig4 = plt.figure(4, facecolor="#f1f1f1")
    ax4 = fig4.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    i = 1
    for pot in np.nditer(potential_minima):
        action = np.loadtxt(f'./output_data/output_cooled_monte_carlo/action_{i}.txt',
                            float,
                            delimiter=' ')
        action_err = np.loadtxt(
            f'./output_data/output_cooled_monte_carlo/action_err_{i}.txt',
            float,
            delimiter=' ')

        ax4.errorbar(n_cooling,
                     action,
                     action_err,
                     fmt='.',
                     capsize=2.5,
                     elinewidth=0.5,
                     label=f'$\eta = {pot}$')

        s0 = 4 / 3 * pow(pot, 3)

        ax4.hlines(s0, 0, n_cooling[-1], color='green',
                   linestyle='dashed', linewidth=0.8)

        i += 1

    ax4.legend()
    ax4.set_xscale('log')
    ax4.set_yscale('log')

    fig4.savefig(filepath + '/action.png', dpi=300)

    plt.show()


def print_rilm_conf():
    """

    """
    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig1 = plt.figure(1, facecolor="#f1f1f1")

    ax1 = fig1.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    conf = np.loadtxt('./output_data/output_rilm/conf_test.txt',
                      float, delimiter=' ')

    tau = np.loadtxt('./output_data/output_rilm/tau_test.txt',
                     float, delimiter=' ')

    ax1.plot(tau, conf)

    plt.show()


def print_graph_heat():
    """

    """
    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig = plt.figure(1, facecolor="#f1f1f1")
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    tau_array = np.loadtxt('./output_data/output_rilm_heating/tau_array_conf.txt',
                           float, delimiter=' ')
    config = np.loadtxt('./output_data/output_rilm_heating/configuration.txt',
                        float, delimiter=' ')
    config_heat = np.loadtxt('./output_data/output_rilm_heating/configuration_heated.txt',
                             float, delimiter=' ')

    ax.plot(tau_array, config[0:-1], color='blue')
    ax.plot(tau_array, config_heat[0:-1], color='red')

    fig.savefig(filepath + '/heating.png', dpi=300)

    plt.show()


def print_stream():
    """

    """
    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig1 = plt.figure(1, facecolor="#f1f1f1")

    ax1 = fig1.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    tau = np.loadtxt('./output_data/output_iilm/streamline/tau_array.txt',
                     float, delimiter=' ')

    for i in range(3):
        conf = np.loadtxt(f'./output_data/output_iilm/streamline/streamline_{i}.txt',
                          float, delimiter=' ')
        ax1.plot(tau, conf)
        #print(mc.return_action(conf)/ip.action_0)

    plt.show()


def print_zcr_hist():
    """

    """
    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig = plt.figure(1, facecolor="#f1f1f1")
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    zcr = np.loadtxt('./output_data/output_rilm/zcr_hist.txt', float,
                     delimiter=' ')

    zcr_cooling = np.loadtxt('./output_data/output_cooled_monte_carlo/zero_crossing/zcr_cooling.txt',
                             float, delimiter=' ')

    zcr_int = np.loadtxt('./output_data/output_iilm/iilm/zcr_hist.txt',
                             float, delimiter=' ')
    
    print(zcr_int.size)
    print(zcr.size)
    print(zcr_cooling.size)

    ax.hist(zcr, 39, (0.1, 4.), histtype='step', color = 'red')
    ax.hist(zcr_int, 39, (0.1, 4.), histtype='step', color = 'orange')#, density='True')
    ax.hist(zcr_cooling, 39, (0.1, 4.), histtype='step',
            color='blue')#, density='True')

    fig.savefig(filepath + '/zcr_histogram.png', dpi=300)

    plt.show()


def print_tau_centers():
    """

    """
    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig = plt.figure(1, facecolor="#f1f1f1")
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    n_conf = np.loadtxt('./output_data/output_iilm/iilm/n_conf.txt',
                        float, delimiter=' ')

    for n in range(12):
        tau = np.loadtxt(f'./output_data/output_iilm/iilm/center_{n+1}.txt',
                         float, delimiter=' ')

        if (n % 2) == 0:
            ax.plot(n_conf, tau, color='blue',
                    linewidth=0.4)
        else:
            ax.plot(n_conf, tau, color='red',
                    linewidth=0.4)

    fig.savefig(filepath + '/iilm_config.png', dpi=300)
    plt.show()


def print_cool_density():
    """

    """
    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig = plt.figure(1, facecolor="#f1f1f1")
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    potential_minima = np.loadtxt('output_data/output_cooled_monte_carlo/potential_minima.txt',
                                  delimiter=' ')

    dens = np.zeros(potential_minima.size)
    dens_err = np.zeros(potential_minima.size)

    i = 1
    for pot in np.nditer(potential_minima):
        j = 0
        with open(f'output_data/output_cooled_monte_carlo/n_instantons_{i}.txt', 'r')\
                as reader:
            for line in reader:

                if j == 8:
                    line = reader.readline()
                    dens[i-1] = float(line)
                j += 1
        j = 0
        with open(f'output_data/output_cooled_monte_carlo/n_instantons_{i}_err.txt', 'r')\
                as reader:
            for line in reader:
                if j == 8:
                    dens_err[i-1] = float(line)
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
                label = 'cooling')

    potential = np.linspace(0.1, 2.0)
    action = 4/3 * np.power(potential, 3)

    loop_1 = 8 * np.sqrt(2/np.pi)\
        * np.power(potential, 5/2) * np. exp(-action)

    loop_2 = loop_1 * np.exp(-71/72/action)

    ax.plot(potential, loop_1,
            linestyle='--', color='green', label = '1-loop')

    ax.plot(potential, loop_2,
            linestyle='-', color='green', label = '2-loop')


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
        color = 'red',
        marker='s',
        linestyle='',
        capsize=2.5,
        elinewidth=0.5,
        label = 'Monte Carlo')

    ax.legend()
    ax.set_xlim(0.0, 1.75)
    ax.set_ylim(0.03, 2.)

    ax.set_yscale('log')

    fig.savefig(filepath + '/density.png', dpi = 300)

    plt.show()

def print_iilm():
    """

    """
    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig1 = plt.figure(1, facecolor="#f1f1f1")

    ax1 = fig1.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")
    
    tau_ia = np.loadtxt('./output_data/output_iilm/streamline/delta_tau_ia.txt',
                           float, delimiter=' ')
    
    zcr = np.loadtxt('./output_data/output_rilm/zcr_hist.txt', float,
                     delimiter=' ')
    zcr_cooling = np.loadtxt('./output_data/output_cooled_monte_carlo/zero_crossing/zcr_cooling.txt',
                             float, delimiter=' ')
    
    hist_1, _ = np.histogram(zcr, 40, range =(0.,4.))
    hist_2, _ = np.histogram(zcr_cooling, 40, range =(0.,4.))
    
    # bisogna implementare un modo per capire già quanto vale il potentiale?
    # tipo costruire l'istogramma già nella funzione zero_... 
    action_ia = -np.log(hist_2[0:10]/hist_1[0:10])/(4/3*np.power(1.4,3))
    
    
    act_int = np.loadtxt('./output_data/output_iilm/streamline/streamline_action_int.txt',
                            float, delimiter = ' ')
    
    array_ia = np.loadtxt('./output_data/output_iilm/streamline/array_ia.txt',
                           float, delimiter=' ')
    
    array_ia_0 = np.loadtxt('./output_data/output_iilm/streamline/array_ia_0.txt',
                           float, delimiter=' ')
    
    array_int = np.loadtxt('./output_data/output_iilm/streamline/array_int.txt',
                            float, delimiter = ' ')
    
    array_int_core = np.loadtxt('./output_data/output_iilm/streamline/array_int_core.txt',
                            float, delimiter = ' ')
    
    array_int_zero_cross = np.loadtxt('./output_data/output_iilm/streamline/array_int_zero_cross.txt',
                            float, delimiter = ' ')
    
    
    ax1.plot(tau_ia, 
             act_int,
             marker = '^',
             linestyle ='',
             color = 'green',
             label = 'streamline'
             )
    
    ax1.plot(array_ia,
             array_int,
             color = 'blue',
             label = 'sum ansatz')
    
    ax1.plot(array_ia,
             array_int_core,
             color = 'orange',
             label = 'core repulsion'
             )
    
    ax1.plot(array_ia_0,
             array_int_zero_cross,
             color = 'red',
             marker = 's',
             linestyle = '',
             label = 'zcr sum ansatz'
             )
    
    ax1.plot(np.linspace(0.1, 1.0, 10, False),
             action_ia,
             marker = 's',
             linestyle = '',
             color = 'cyan',
             label = 'monte carlo cooling')
    
    ax1.legend()
    
    ax1.set_ylim(-2.0,0.5)
    ax1.set_xlim(-0.05, 2.0)
    
    fig1.savefig(filepath + '/iilm.png', dpi = 300)
    
    plt.show()