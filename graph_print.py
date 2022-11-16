
import numpy as np
import struct
import matplotlib.pyplot as plt

import utility_custom
import utility_monte_carlo as mc
import input_parameters as ip

filepath = './output_graph'
utility_custom.output_control(filepath)


def print_graph_free_energy():
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

    ax1.errorbar(temperature_array,
                 f_energy,
                 f_energy_err,
                 color='b',
                 fmt='.',
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


def print_graph_cool_conf():

    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig = plt.figure(1, facecolor="#f1f1f1")
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    tau_array = np.loadtxt('./output_data/output_cooled_monte_carlo/tau_array.txt',
                           float, delimiter=' ')
    config = np.loadtxt('./output_data/output_cooled_monte_carlo/configuration.txt',
                        float, delimiter=' ')
    config_cool = np.loadtxt('./output_data/output_cooled_monte_carlo/configuration_cooled.txt',
                             float, delimiter=' ')

    ax.plot(tau_array, config, color='blue')
    ax.plot(tau_array, config_cool, color='red')

    fig.savefig(filepath + '/cooling.png', dpi=300)

    plt.show()


def print_graph_mc():

    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig1 = plt.figure(1, facecolor="#f1f1f1")

    ax1 = fig1.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    tau_array = np.loadtxt('./output_data/output_monte_carlo/tau_array.txt',
                           float, delimiter=' ')
    corr1 = np.loadtxt('./output_data/output_monte_carlo/average_x_cor_1.txt',
                       float, delimiter=' ')

    corr2 = np.loadtxt('./output_data/output_monte_carlo/average_x_cor_2.txt',
                       float, delimiter=' ')

    corr3 = np.loadtxt('./output_data/output_monte_carlo/average_x_cor_3.txt',
                       float, delimiter=' ')

    corr_err1 = np.loadtxt('./output_data/output_monte_carlo/error_x_cor_1.txt',
                           float, delimiter=' ')

    corr_err2 = np.loadtxt('./output_data/output_monte_carlo/error_x_cor_2.txt',
                           float, delimiter=' ')

    corr_err3 = np.loadtxt('./output_data/output_monte_carlo/error_x_cor_3.txt',
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

    fig1.savefig(filepath + '/x_corr.png', dpi=300)

    fig2 = plt.figure(2, facecolor="#f1f1f1")

    ax2 = fig2.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    dcorr1 = np.loadtxt('./output_data/output_monte_carlo/average_der_log_1.txt',
                        float, delimiter=' ')

    dcorr2 = np.loadtxt('./output_data/output_monte_carlo/average_der_log_2.txt',
                        float, delimiter=' ')

    dcorr3 = np.loadtxt('./output_data/output_monte_carlo/average_der_log_3.txt',
                        float, delimiter=' ')

    dcorr_err1 = np.loadtxt('./output_data/output_monte_carlo/error_der_log_1.txt',
                            float, delimiter=' ')

    dcorr_err2 = np.loadtxt('./output_data/output_monte_carlo/error_der_log_2.txt',
                            float, delimiter=' ')

    dcorr_err3 = np.loadtxt('./output_data/output_monte_carlo/error_der_log_3.txt',
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

    fig2.savefig(filepath + '/der_corr.png', dpi=300)

    plt.show()


def print_graph_cool():

    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig1 = plt.figure(1, facecolor="#f1f1f1")

    ax1 = fig1.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    corr1_c = np.loadtxt('./output_data/output_cooled_monte_carlo/average_x_cor_cold_1.txt',
                         float, delimiter=' ')

    corr2_c = np.loadtxt('./output_data/output_cooled_monte_carlo/average_x_cor_cold_2.txt',
                         float, delimiter=' ')

    corr3_c = np.loadtxt('./output_data/output_cooled_monte_carlo/average_x_cor_cold_3.txt',
                         float, delimiter=' ')

    corr_err1_c = np.loadtxt('./output_data/output_cooled_monte_carlo/error_x_cor_cold_1.txt',
                             float, delimiter=' ')

    corr_err2_c = np.loadtxt('./output_data/output_cooled_monte_carlo/error_x_cor_cold_2.txt',
                             float, delimiter=' ')

    corr_err3_c = np.loadtxt('./output_data/output_cooled_monte_carlo/error_x_cor_cold_3.txt',
                             float, delimiter=' ')

    tau_array_2 = np.loadtxt('./output_data/output_diag/tau_array.txt',
                             float, delimiter=' ')

    corr1_d = np.loadtxt('./output_data/output_diag/corr_function.txt',
                         float, delimiter=' ')

    corr2_d = np.loadtxt('./output_data/output_diag/corr_function2.txt',
                         float, delimiter=' ')

    corr3_d = np.loadtxt('./output_data/output_diag/corr_function3.txt',
                         float, delimiter=' ')

    tau_array_c = np.linspace(0.0, 20 * 0.05, 20, False)

    ax1.plot(tau_array_2[0:41],
             corr1_d[0:41],
             color='b',
             linewidth=0.8,
             linestyle=':')

    ax1.plot(tau_array_2[0:41],
             corr2_d[0:41],
             color='r',
             linewidth=0.8,
             linestyle=':')

    ax1.plot(tau_array_2[0:41],
             corr3_d[0:41],
             color='g',
             linewidth=0.8,
             linestyle=':')

    ax1.errorbar(tau_array_c,
                 corr1_c,
                 corr_err1_c,
                 color='b',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    ax1.errorbar(tau_array_c,
                 corr2_c,
                 corr_err2_c,
                 color='red',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    ax1.errorbar(tau_array_c,
                 corr3_c,
                 corr_err3_c,
                 color='green',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    fig1.savefig(filepath + '/x_corr_cold.png', dpi=300)

    fig2 = plt.figure(2, facecolor="#f1f1f1")

    ax2 = fig2.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    dcorr1_c = np.loadtxt('./output_data/output_cooled_monte_carlo/average_der_log_cold_1.txt',
                          float, delimiter=' ')

    dcorr2_c = np.loadtxt('./output_data/output_cooled_monte_carlo/average_der_log_cold_2.txt',
                          float, delimiter=' ')

    dcorr3_c = np.loadtxt('./output_data/output_cooled_monte_carlo/average_der_log_cold_3.txt',
                          float, delimiter=' ')

    dcorr_err1_c = np.loadtxt('./output_data/output_cooled_monte_carlo/error_der_log_cold_1.txt',
                              float, delimiter=' ')

    dcorr_err2_c = np.loadtxt('./output_data/output_cooled_monte_carlo/error_der_log_cold_2.txt',
                              float, delimiter=' ')

    dcorr_err3_c = np.loadtxt('./output_data/output_cooled_monte_carlo/error_der_log_cold_3.txt',
                              float, delimiter=' ')

    dcorr1_d = np.loadtxt('./output_data/output_diag/av_der_log_corr_funct.txt',
                          float, delimiter=' ')

    dcorr2_d = np.loadtxt('./output_data/output_diag/av_der_log_corr_funct2.txt',
                          float, delimiter=' ')

    dcorr3_d = np.loadtxt('./output_data/output_diag/av_der_log_corr_funct3.txt',
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

    ax2.errorbar(tau_array_c[0:-1],
                 dcorr1_c,
                 dcorr_err1_c,
                 color='b',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    ax2.errorbar(tau_array_c[0:-5],
                 dcorr2_c[0:-4],
                 dcorr_err2_c[0:-4],
                 color='red',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    ax2.errorbar(tau_array_c[0:-1],
                 dcorr3_c,
                 dcorr_err3_c,
                 color='green',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    ax2.set_ylim(-1.0, 10.0)

    fig2.savefig(filepath + '/der_corr_cold.png', dpi=300)


def print_density():

    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig3 = plt.figure(3, facecolor="#f1f1f1")
    ax3 = fig3.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")



    # n_cooling = np.loadtxt(
    #     './output_data/output_cooled_monte_carlo/n_cooling.txt')
    n_instantons1 = np.loadtxt(
        './output_data/output_cooled_monte_carlo/n_instantons_1.txt')
    n_instantons_err1 = np.loadtxt(
        './output_data/output_cooled_monte_carlo/n_instantons_1_err.txt')
    n_instantons2 = np.loadtxt(
        './output_data/output_cooled_monte_carlo/n_instantons_2.txt')
    n_instantons_err2 = np.loadtxt(
        './output_data/output_cooled_monte_carlo/n_instantons_2_err.txt')
    n_instantons3 = np.loadtxt(
        './output_data/output_cooled_monte_carlo/n_instantons_3.txt')
    n_instantons_err3 = np.loadtxt(
        './output_data/output_cooled_monte_carlo/n_instantons_3_err.txt')

    n_cooling = np.linspace(1 , n_instantons1.size, n_instantons1.size)

    # n_instantons1 *= 2
    # n_instantons2 *= 2
    # n_instantons3 *= 2
    ax3.errorbar(n_cooling,
                 n_instantons1,
                 n_instantons_err1,
                 color='b',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5,
                 label=r'$\eta = 1.4$')

    ax3.errorbar(n_cooling,
                 n_instantons2,
                 n_instantons_err2,
                 color='r',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5,
                 label=r'$\eta = 1.5$')

    ax3.errorbar(n_cooling,
                 n_instantons3,
                 n_instantons_err3,
                 color='orange',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5,
                 label=r'$\eta = 1.6$')

    ax3.legend()

    # potential_minima = np.loadtxt('./output_data/output_cooled_monte_carlo/potential_minima.txt',
    #                               float, delimiter=' ')

    loop_1 = np.zeros(3, float)
    loop_2 = np.zeros(3, float)
    color_array = ['blue', 'red', 'orange']

    c = 0
    for l in [1.4,1.5, 1.6]: #np.nditer(np.array([1.4, 1.5, 1.6])):

        s0 = 4 / 3 * pow(l, 3)

        loop_1 = 8 * pow(l, 5 / 2) \
            * pow(2 / np.pi, 1/2) * np.exp(-s0)

        loop_2 = 8 * pow(l, 5 / 2) \
            * pow(2 / np.pi, 1/2) * np.exp(-s0 - 71 / (72 * s0))

        ax3.hlines([loop_1, loop_2], 0, 200, color_array[c],
                   ['dashed', 'solid'], linewidth=0.5)
        c += 1

    ax3.set_xscale('log')

    ax3.set_yscale('log')

    fig3.savefig(filepath + '/n_istantons.png', dpi=300)

    fig4 = plt.figure(4, facecolor="#f1f1f1")
    ax4 = fig4.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    action1 = np.loadtxt('./output_data/output_cooled_monte_carlo/action_1.txt',
                         float,
                         delimiter=' ')
    action1_err = np.loadtxt('./output_data/output_cooled_monte_carlo/action_err_1.txt',
                              float,
                              delimiter=' ')
    action2 = np.loadtxt('./output_data/output_cooled_monte_carlo/action_2.txt',
                         float,
                         delimiter=' ')
    action2_err = np.loadtxt('./output_data/output_cooled_monte_carlo/action_err_2.txt',
                              float,
                              delimiter=' ')

    action3 = np.loadtxt('./output_data/output_cooled_monte_carlo/action_3.txt',
                         float,
                         delimiter=' ')
    action3_err = np.loadtxt('./output_data/output_cooled_monte_carlo/action_err_3.txt',
                              float,
                              delimiter=' ')

    c = 0
    for l1 in {1.4,1.5,1.6}:
        print(l1)
        s0 = (4 / 3) * pow(l1, 3)

        ax4.hlines(s0, 0, 200, color_array[c], 'dashed',
                    linewidth = 0.8)
        c+=1

    ax4.errorbar(n_cooling, action1, action1_err,
                  linestyle='',
                  color='b',
                  marker='x',
                  capsize=2.5,
                  elinewidth=0.5,
                  label=r'$\eta = 1.4$')

    ax4.errorbar(n_cooling, action2, action2_err,
                  linestyle='',
                  color='r',
                  marker='x',
                  capsize=2.5,
                  elinewidth=0.5,
                  label=r'$\eta = 1.5$')
    ax4.errorbar(n_cooling, action3, action3_err,
                  color='orange',
                  linestyle='',
                  marker='x',
                  capsize=2.5,
                  elinewidth=0.5,
                  label=r'$\eta = 1.6$')


    ax4.legend()
    ax4.set_xscale('log')
    ax4.set_yscale('log')

    fig4.savefig(filepath + '/action.png', dpi=300)

    plt.show()


def print_graph_rilm():
    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig1 = plt.figure(1, facecolor="#f1f1f1")

    ax1 = fig1.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    tau_array = np.loadtxt('./output_data/output_rilm/tau_array.txt',
                           float, delimiter=' ')
    corr1 = np.loadtxt('./output_data/output_rilm/average_x_cor_1.txt',
                       float, delimiter=' ')

    corr2 = np.loadtxt('./output_data/output_rilm/average_x_cor_2.txt',
                       float, delimiter=' ')

    corr3 = np.loadtxt('./output_data/output_rilm/average_x_cor_3.txt',
                       float, delimiter=' ')

    corr_err1 = np.loadtxt('./output_data/output_rilm/error_x_cor_1.txt',
                           float, delimiter=' ')

    corr_err2 = np.loadtxt('./output_data/output_rilm/error_x_cor_2.txt',
                           float, delimiter=' ')

    corr_err3 = np.loadtxt('./output_data/output_rilm/error_x_cor_3.txt',
                           float, delimiter=' ')

    tau_array_2 = np.loadtxt('./output_data/output_diag/tau_array.txt',
                             float, delimiter=' ')

    corr1_d = np.loadtxt('./output_data/output_diag/corr_function.txt',
                         float, delimiter=' ')

    corr2_d = np.loadtxt('./output_data/output_diag/corr_function2.txt',
                         float, delimiter=' ')

    corr3_d = np.loadtxt('./output_data/output_diag/corr_function3.txt',
                         float, delimiter=' ')

    ax1.plot(tau_array_2[0:60],
             corr1_d[0:60],
             color='b',
             linewidth=0.8,
             linestyle=':')

    ax1.plot(tau_array_2[0:60],
             corr2_d[0:60],
             color='r',
             linewidth=0.8,
             linestyle=':')

    ax1.plot(tau_array_2[0:60],
             corr3_d[0:60],
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

    fig1.savefig(filepath + '/x_corr_rilm.png', dpi=300)

    fig2 = plt.figure(2, facecolor="#f1f1f1")

    ax2 = fig2.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    dcorr1 = np.loadtxt('./output_data/output_rilm/average_der_log_1.txt',
                        float, delimiter=' ')

    dcorr2 = np.loadtxt('./output_data/output_rilm/average_der_log_2.txt',
                        float, delimiter=' ')

    dcorr3 = np.loadtxt('./output_data/output_rilm/average_der_log_3.txt',
                        float, delimiter=' ')

    dcorr_err1 = np.loadtxt('./output_data/output_rilm/error_der_log_1.txt',
                            float, delimiter=' ')

    dcorr_err2 = np.loadtxt('./output_data/output_rilm/error_der_log_2.txt',
                            float, delimiter=' ')

    dcorr_err3 = np.loadtxt('./output_data/output_rilm/error_der_log_3.txt',
                            float, delimiter=' ')

    dcorr1_d = np.loadtxt('./output_data/output_diag/av_der_log_corr_funct.txt',
                          float, delimiter=' ')

    dcorr2_d = np.loadtxt('./output_data/output_diag/av_der_log_corr_funct2.txt',
                          float, delimiter=' ')

    dcorr3_d = np.loadtxt('./output_data/output_diag/av_der_log_corr_funct3.txt',
                          float, delimiter=' ')

    ax2.set_ylim(-1.0, 6.0)

    ax2.plot(tau_array_2[0:60],
             dcorr1_d[0:60],
             color='b',
             linewidth=0.8,
             linestyle=':')

    ax2.plot(tau_array_2[0:60],
             dcorr2_d[0:60],
             color='r',
             linewidth=0.8,
             linestyle=':')

    ax2.plot(tau_array_2[0:60],
             dcorr3_d[0:60],
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

    fig2.savefig(filepath + '/der_corr_rilm.png', dpi=300)

    plt.show()
    

def print_rilm_conf():
    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig1 = plt.figure(1, facecolor="#f1f1f1")

    ax1 = fig1.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")
    
    conf  = np.loadtxt('./output_data/output_rilm/conf_test.txt',
                           float, delimiter=' ')
    
    tau = np.loadtxt('./output_data/output_rilm/tau_test.txt',
                           float, delimiter=' ')

    ax1.plot(tau, conf)
    
    plt.show()
    
def print_graph_rilm_heating():
    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig1 = plt.figure(1, facecolor="#f1f1f1")

    ax1 = fig1.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    tau_array = np.loadtxt('./output_data/output_rilm_heating/tau_array.txt',
                           float, delimiter=' ')
    corr1 = np.loadtxt('./output_data/output_rilm_heating/average_x_cor_1.txt',
                       float, delimiter=' ')

    corr2 = np.loadtxt('./output_data/output_rilm_heating/average_x_cor_2.txt',
                       float, delimiter=' ')

    corr3 = np.loadtxt('./output_data/output_rilm_heating/average_x_cor_3.txt',
                       float, delimiter=' ')

    corr_err1 = np.loadtxt('./output_data/output_rilm_heating/error_x_cor_1.txt',
                           float, delimiter=' ')

    corr_err2 = np.loadtxt('./output_data/output_rilm_heating/error_x_cor_2.txt',
                           float, delimiter=' ')

    corr_err3 = np.loadtxt('./output_data/output_rilm_heating/error_x_cor_3.txt',
                           float, delimiter=' ')

    tau_array_2 = np.loadtxt('./output_data/output_diag/tau_array.txt',
                             float, delimiter=' ')

    corr1_d = np.loadtxt('./output_data/output_diag/corr_function.txt',
                         float, delimiter=' ')

    corr2_d = np.loadtxt('./output_data/output_diag/corr_function2.txt',
                         float, delimiter=' ')

    corr3_d = np.loadtxt('./output_data/output_diag/corr_function3.txt',
                         float, delimiter=' ')

    ax1.plot(tau_array_2[0:60],
             corr1_d[0:60],
             color='b',
             linewidth=0.8,
             linestyle=':')

    ax1.plot(tau_array_2[0:60],
             corr2_d[0:60],
             color='r',
             linewidth=0.8,
             linestyle=':')

    ax1.plot(tau_array_2[0:60],
             corr3_d[0:60],
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

    fig1.savefig(filepath + '/x_corr_rilm_heat.png', dpi=300)

    fig2 = plt.figure(2, facecolor="#f1f1f1")

    ax2 = fig2.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    dcorr1 = np.loadtxt('./output_data/output_rilm_heating/average_der_log_1.txt',
                        float, delimiter=' ')

    dcorr2 = np.loadtxt('./output_data/output_rilm_heating/average_der_log_2.txt',
                        float, delimiter=' ')

    dcorr3 = np.loadtxt('./output_data/output_rilm_heating/average_der_log_3.txt',
                        float, delimiter=' ')

    dcorr_err1 = np.loadtxt('./output_data/output_rilm_heating/error_der_log_1.txt',
                            float, delimiter=' ')

    dcorr_err2 = np.loadtxt('./output_data/output_rilm_heating/error_der_log_2.txt',
                            float, delimiter=' ')

    dcorr_err3 = np.loadtxt('./output_data/output_rilm_heating/error_der_log_3.txt',
                            float, delimiter=' ')

    dcorr1_d = np.loadtxt('./output_data/output_diag/av_der_log_corr_funct.txt',
                          float, delimiter=' ')

    dcorr2_d = np.loadtxt('./output_data/output_diag/av_der_log_corr_funct2.txt',
                          float, delimiter=' ')

    dcorr3_d = np.loadtxt('./output_data/output_diag/av_der_log_corr_funct3.txt',
                          float, delimiter=' ')

    ax2.set_ylim(-1.0, 6.0)

    ax2.plot(tau_array_2[0:60],
             dcorr1_d[0:60],
             color='b',
             linewidth=0.8,
             linestyle=':')

    ax2.plot(tau_array_2[0:60],
             dcorr2_d[0:60],
             color='r',
             linewidth=0.8,
             linestyle=':')

    ax2.plot(tau_array_2[0:60],
             dcorr3_d[0:60],
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

    fig2.savefig(filepath + '/der_corr_rilm_heat.png', dpi=300)

    plt.show()

    
def print_graph_heat():
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


def print_iilm():
    
    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig1 = plt.figure(1, facecolor="#f1f1f1")

    ax1 = fig1.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")
    
    tau_ia = np.loadtxt('./output_data/output_iilm/streamline/delta_tau_ia.txt',
                           float, delimiter=' ')
    
    act_int = np.loadtxt('./output_data/output_iilm/streamline/streamline_action_int.txt',
                            float, delimiter = ' ')
    
    array_ia = np.loadtxt('./output_data/output_iilm/streamline/array_ia.txt',
                           float, delimiter=' ')
    
    array_int = np.loadtxt('./output_data/output_iilm/streamline/array_int.txt',
                            float, delimiter = ' ')
    
    array_int_core = np.loadtxt('./output_data/output_iilm/streamline/array_int_core.txt',
                            float, delimiter = ' ')
    
    
    ax1.plot(tau_ia, 
             act_int,
             marker = '^',
             linestyle ='',
             color = 'green'
             )
    
    ax1.plot(array_ia,
             array_int,
             color = 'b')
    
    ax1.plot(array_ia,
             array_int_core,
             color = 'orange')
    
    ax1.set_ylim(-2.05, 0.05)
    ax1.set_xlim(-0.05, 2.0)
    
    fig1.savefig(filepath + '/iilm.png', dpi = 300)
    
    plt.show()
    
def print_stream():
    
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
        print(mc.return_action(conf)/ip.action_0)
        
    plt.show()
    
    
def print_zcr_hist():
    
    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig = plt.figure(1, facecolor="#f1f1f1")
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")
    
    zcr = np.loadtxt('./output_data/output_rilm/zcr_hist.txt', float, 
                     delimiter =' ')
    
    zcr_cooling = np.loadtxt('./output_data/output_cooled_monte_carlo/zero_crossing/zcr_cooling.txt',
                             float, delimiter =' ')
    
    print(zcr.size)
    print(zcr_cooling.size)
    
    ax.hist(zcr, 390, (0.1, 4.), histtype = 'step',density='True')
    ax.hist(zcr_cooling, 390, (0.1,4.), histtype = 'step', color ='blue', density='True')
    
    fig.savefig(filepath + '/zcr_histogram.png', dpi = 300)
    
    plt.show()
    
    
def print_tau_centers():
    
    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig = plt.figure(1, facecolor="#f1f1f1")
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    n_conf = np.loadtxt('./output_data/output_iilm/iilm/n_conf.txt',
                             float, delimiter =' ')

    for n in range(12):
        tau = np.loadtxt(f'./output_data/output_iilm/iilm/center_{n+1}.txt',
                                 float, delimiter =' ')

        if (n % 2) == 0:
            ax.plot(n_conf, tau, color = 'blue',
                    linewidth = 0.4)
        else:
            ax.plot(n_conf, tau, color = 'red',
                    linewidth = 0.4)

    fig.savefig(filepath + '/iilm_config.png', dpi = 300)
    plt.show()
    
    
def print_iilm_graph():
    
    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    fig1 = plt.figure(1, facecolor="#f1f1f1")

    ax1 = fig1.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    corr1_c = np.loadtxt('./output_data/output_iilm/iilm/average_x_cor_1.txt',
                         float, delimiter=' ')

    corr2_c = np.loadtxt('./output_data/output_iilm/iilm/average_x_cor_2.txt',
                         float, delimiter=' ')

    corr3_c = np.loadtxt('./output_data/output_iilm/iilm/average_x_cor_3.txt',
                         float, delimiter=' ')

    corr_err1_c = np.loadtxt('./output_data/output_iilm/iilm/error_x_cor_1.txt',
                             float, delimiter=' ')

    corr_err2_c = np.loadtxt('./output_data/output_iilm/iilm/error_x_cor_2.txt',
                             float, delimiter=' ')

    corr_err3_c = np.loadtxt('./output_data/output_iilm/iilm/error_x_cor_3.txt',
                             float, delimiter=' ')

    tau_array_2 = np.loadtxt('./output_data/output_diag/tau_array.txt',
                             float, delimiter=' ')

    corr1_d = np.loadtxt('./output_data/output_diag/corr_function.txt',
                         float, delimiter=' ')

    corr2_d = np.loadtxt('./output_data/output_diag/corr_function2.txt',
                         float, delimiter=' ')

    corr3_d = np.loadtxt('./output_data/output_diag/corr_function3.txt',
                         float, delimiter=' ')

    tau_array_c = np.linspace(0.0, 20 * 0.05, 20, False)

    ax1.plot(tau_array_2[0:41],
             corr1_d[0:41],
             color='b',
             linewidth=0.8,
             linestyle=':')

    ax1.plot(tau_array_2[0:41],
             corr2_d[0:41],
             color='r',
             linewidth=0.8,
             linestyle=':')

    ax1.plot(tau_array_2[0:41],
             corr3_d[0:41],
             color='g',
             linewidth=0.8,
             linestyle=':')

    ax1.errorbar(tau_array_c,
                 corr1_c,
                 corr_err1_c,
                 color='b',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    ax1.errorbar(tau_array_c,
                 corr2_c,
                 corr_err2_c,
                 color='red',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    ax1.errorbar(tau_array_c,
                 corr3_c,
                 corr_err3_c,
                 color='green',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    fig1.savefig(filepath + '/x_corr_cold.png', dpi=300)

    fig2 = plt.figure(2, facecolor="#f1f1f1")

    ax2 = fig2.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    dcorr1_c = np.loadtxt('./output_data/output_iilm/iilm/average_der_log_1.txt',
                          float, delimiter=' ')

    dcorr2_c = np.loadtxt('./output_data/output_iilm/iilm/average_der_log_2.txt',
                          float, delimiter=' ')

    dcorr3_c = np.loadtxt('./output_data/output_iilm/iilm/average_der_log_3.txt',
                          float, delimiter=' ')

    dcorr_err1_c = np.loadtxt('./output_data/output_iilm/iilm/error_der_log_1.txt',
                              float, delimiter=' ')

    dcorr_err2_c = np.loadtxt('./output_data/output_iilm/iilm/error_der_log_2.txt',
                              float, delimiter=' ')

    dcorr_err3_c = np.loadtxt('./output_data/output_iilm/iilm/error_der_log_3.txt',
                              float, delimiter=' ')

    dcorr1_d = np.loadtxt('./output_data/output_diag/av_der_log_corr_funct.txt',
                          float, delimiter=' ')

    dcorr2_d = np.loadtxt('./output_data/output_diag/av_der_log_corr_funct2.txt',
                          float, delimiter=' ')

    dcorr3_d = np.loadtxt('./output_data/output_diag/av_der_log_corr_funct3.txt',
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

    ax2.errorbar(tau_array_c[0:-1],
                 dcorr1_c,
                 dcorr_err1_c,
                 color='b',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    ax2.errorbar(tau_array_c[0:-5],
                 dcorr2_c[0:-4],
                 dcorr_err2_c[0:-4],
                 color='red',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    ax2.errorbar(tau_array_c[0:-1],
                 dcorr3_c,
                 dcorr_err3_c,
                 color='green',
                 fmt='.',
                 capsize=2.5,
                 elinewidth=0.5)

    ax2.set_ylim(-1.0, 10.0)

    fig2.savefig(filepath + '/der_corr.png', dpi=300)
    
    plt.show()