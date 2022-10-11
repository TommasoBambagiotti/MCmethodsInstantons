
import struct
import time
import random as rnd
import numpy as np
import matplotlib.pyplot as plt


def stat_av_var(observable, observable_sqrd, n_data):
    '''Evaluate the average and the variance of the average of a set of data,
    expressed in an array, directly as the sum and the sum of squares.
    We use the formula Var[<O>] = (<O^2> - <O>^2)/N'''
    # Control
    n_array = 0

    if observable.size == observable_sqrd.size:
        n_array = observable.size
    else:
        return None, None

    observable_av = np.empty((n_array), float)
    observable_err = np.empty((n_array), float)
    var_temp = 0.0

    for i_array in range(n_array):
        observable_av[i_array] = observable[i_array]/n_data
        var_temp = observable_sqrd[i_array]/(n_data*n_data) - \
            pow(observable_av[i_array], 2)/n_data
        if var_temp > 0.0:
            observable_err[i_array] = pow(var_temp, 1/2)
        else:
            observable_err[i_array] = 0.

    return observable_av, observable_err


def log_central_der_alg(corr_funct, corr_funct_err, delta_step):
    '''Log derivative of the correlation functions.
    We can not use the analytic formula because
    we do not know the energy eignevalues.'''
    n_array = 0

    if corr_funct.size == corr_funct_err.size:
        n_array = corr_funct.size
        if n_array < 2:
            print("No diff possible")
            return None, None
    else:
        return None, None

    # Reference method
    derivative_log = np.empty((n_array-1), float)
    derivative_log_err = np.empty((n_array-1), float)

    for i_array in range(n_array-1):
        derivative_log[i_array] = - ((corr_funct[i_array+1] -
                                      corr_funct[i_array]) /
                                     (corr_funct[i_array]*delta_step))

        derivative_log_err[i_array] = pow(pow(corr_funct_err[i_array+1] /
                                              corr_funct[i_array], 2) +
                                          pow(corr_funct_err[i_array] *
                                              corr_funct[i_array+1] /
                                              pow(corr_funct[i_array], 2),
                                              2), 1/2) / delta_step

    return derivative_log, derivative_log_err


def monte_carlo_ao(x_potential_minimum,  # potential well position
                   n_lattice,  # size of the grid
                   dtau,  # grid spacing in time
                   n_equil,  # equilibration sweeps
                   n_mc_sweeps,  # monte carlo sweeps
                   delta_x,  # width of gauss. dist. for update x
                   n_point,  #
                   n_meas,  #
                   icold):  # cold/hot start):
    '''Solve the anharmonic oscillator through
    Monte Carlo technique on an Euclidian Axis'''

    if n_mc_sweeps < n_equil:
        print("too few Monte Carlo sweeps/ N_equilib > N_Monte_Carlo")

    # Correlation functions
    x_cor_sum = np.zeros((n_point))
    x_cor_2_sum = np.zeros((n_point))
    x_cor_3_sum = np.zeros((n_point))

    x2_cor_sum = np.zeros((n_point))
    x2_cor_2_sum = np.zeros((n_point))
    x2_cor_3_sum = np.zeros((n_point))

    # Averages and errors
    x_cor_av = np.zeros((n_point))
    x_cor_err = np.zeros((n_point))

    x_cor_2_av = np.zeros((n_point))
    x_cor_2_err = np.zeros((n_point))

    x_cor_3_av = np.zeros((n_point))
    x_cor_3_err = np.zeros((n_point))

    derivative_log_corr_funct = np.empty((n_point-1))
    derivative_log_corr_funct_2 = np.empty((n_point))
    derivative_log_corr_funct_3 = np.empty((n_point-1))

    derivative_log_corr_funct_err = np.empty((n_point-1))
    derivative_log_corr_funct_2_err = np.empty((n_point))
    derivative_log_corr_funct_3_err = np.empty((n_point-1))

    # x position along the tau axis
    x_config = np.zeros((n_lattice+1))

    # initialize the lattice
    rnd_uniform = rnd.Random(time.time())
    rnd_gauss = rnd.Random(time.time())

    if icold is True:
        for i in range(n_lattice):
            x_config[i] = -x_potential_minimum
    else:
        rnd.seed()
        for i in range(n_lattice):
            x_config[i] = 2 * \
                rnd_uniform.uniform(0., 1.)*x_potential_minimum \
                - x_potential_minimum

    # impose periodic boundary conditions and
    # understand the way in which they are imposed in the reference file
    x_config[n_lattice-1] = x_config[0]
    x_config[n_lattice] = x_config[1]

    # Monte Carlo sweeps: Principal cycle

    # Counters for different parts
    n_accept = 0
    n_hit = 0
    n_config = 0
    n_corr = 0

    # Action, Kin en and Potential en
    action_total = 0.0  # total action
    kin_en_total = 0.0
    pot_en_total = 0.0

    # Temporary variables
    kin_en_temp = 0.0
    pot_en_temp = 0.0

    # Output file hist
    with open('hist.dat', 'wb') as writer:

        # Equilibration cycle
        while n_config < n_equil:
            for i in range(1, n_lattice):
                # we apply Metropolis algorithm for each site tau_k:
                # Secundary cycle
                n_hit += 1

                action_loc_old = (
                    pow((x_config[i] - x_config[i-1])/(2*dtau), 2) +
                    pow((x_config[i+1] - x_config[i])/(2*dtau), 2) +
                    pow(x_config[i] * x_config[i] -
                        x_potential_minimum*x_potential_minimum, 2)
                )*dtau

                x_new = x_config[i] + rnd_gauss.gauss(0, delta_x)

                action_loc_new = (
                    pow((x_new - x_config[i-1])/(2*dtau), 2) +
                    pow((x_config[i+1] - x_new)/(2*dtau), 2) +
                    pow(x_new * x_new -
                        x_potential_minimum*x_potential_minimum, 2))*dtau

                delta_action = action_loc_new - action_loc_old

                # we put a bound on the value of delta_S
                # because we need the exp.
                delta_action = max(delta_action, -70.0)
                delta_action = min(delta_action, 70.0)
                # Metropolis question:
                if np.exp(-delta_action) > rnd_uniform.uniform(0., 1.):
                    x_config[i] = x_new
                    n_accept += 1

            x_config[0] = x_config[n_lattice-1]
            x_config[n_lattice] = x_config[1]

            n_config += 1

        n_config = 0

        # Rest of the MC sweeps
        while (n_config + n_equil) < n_mc_sweeps:
            for i in range(1, n_lattice):
                n_hit += 1

                action_loc_old = (
                    pow((x_config[i] - x_config[i-1])/(2*dtau), 2) +
                    pow((x_config[i+1] - x_config[i])/(2*dtau), 2) +
                    pow(x_config[i] * x_config[i] -
                        x_potential_minimum*x_potential_minimum, 2)
                )*dtau

                x_new = x_config[i] + rnd_gauss.gauss(0, delta_x)

                action_loc_new = (
                    pow((x_new - x_config[i-1])/(2*dtau), 2) +
                    pow((x_config[i+1] - x_new)/(2*dtau), 2) +
                    pow(x_new * x_new -
                        x_potential_minimum*x_potential_minimum, 2))*dtau

                delta_action = action_loc_new - action_loc_old

                delta_action = max(delta_action, -70.0)
                delta_action = min(delta_action, 70.0)
                # Metropolis question:
                if np.exp(-delta_action) > rnd_uniform.uniform(0., 1.):
                    x_config[i] = x_new
                    n_accept += 1

                writer.write(x_config[i])

            x_config[0] = x_config[n_lattice-1]

            writer.write(x_config[0])

            x_config[n_lattice] = x_config[1]

            # Calculate the total action:
            for j in range(n_lattice-1):
                kin_en_temp = pow((x_config[j+1] - x_config[j])/(2*dtau), 2)
                pot_en_temp = pow(x_config[j] * x_config[j] -
                                  x_potential_minimum*x_potential_minimum, 2)

                kin_en_total += kin_en_temp
                pot_en_total += pot_en_temp
                action_total += (kin_en_temp + pot_en_temp)*dtau

                kin_en_temp = 0.0
                pot_en_temp = 0.0

            for k_meas in range(n_meas):
                n_corr += 1

                i_p0 = int((n_lattice - n_point) * rnd_uniform.uniform(0., 1.))
                x_0 = x_config[i_p0]
                for i_corr in range(n_point):
                    x_1 = x_config[i_p0 + i_corr]

                    x_cor_sum[i_corr] += x_0 * x_1
                    x_cor_2_sum[i_corr] += pow(x_0 * x_1, 2)
                    x_cor_3_sum[i_corr] += pow(x_0 * x_1, 3)

                    x2_cor_sum[i_corr] += pow(x_0 * x_1, 2)
                    x2_cor_2_sum[i_corr] += pow(x_0 * x_1, 4)
                    x2_cor_3_sum[i_corr] += pow(x_0 * x_1, 6)
                    # Close the output file
            
            n_config += 1
    # Monte Carlo End
    # Evaluate averages and other stuff, maybe we can create a function
    x_cor_av, x_cor_err = stat_av_var(x_cor_sum, x2_cor_sum, n_corr)
    x_cor_2_av, x_cor_2_err = stat_av_var(x_cor_2_sum, x2_cor_2_sum, n_corr)
    x_cor_3_av, x_cor_3_err = stat_av_var(x_cor_3_sum, x2_cor_3_sum, n_corr)

    # Correlation function Log
    derivative_log_corr_funct, derivative_log_corr_funct_err = \
        log_central_der_alg(x_cor_av, x_cor_err, dtau)

    # In the case of log <x^2x^2> the constant part <x^2>
    # is circa the average for the greatest tau

    derivative_log_corr_funct_2, derivative_log_corr_funct_2_err = \
        log_central_der_alg(
            x_cor_2_av - x_cor_2_av[n_point-1],
            np.sqrt(x_cor_2_err * x_cor_2_err + pow(x_cor_2_err, 2)),
            dtau)

    derivative_log_corr_funct_3, derivative_log_corr_funct_3_err = \
        log_central_der_alg(x_cor_3_av, x_cor_3_err, dtau)

    # Control Acceptance ratio
    print('acceptance ratio = {n}'.format(n=n_accept/n_hit))

    # In order for matplotlib to print text in LaTeX
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    # Print correlation function plot
    tau_corr_axis = np.linspace(0, n_point*dtau, n_point, False)

    fig1 = plt.figure(1, figsize=(8, 2.5), facecolor="#f1f1f1")
    ax1 = fig1.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    ax1.set_xlabel(r"$\tau$")
    ax1.set_ylabel(r"$<x^{n}(0) x^{n}(\tau)>$")

    ax1.errorbar(tau_corr_axis, x_cor_av, x_cor_err, color='blue',
                 fmt='o', label=r"$<x(0)x(\tau)>$")
    ax1.errorbar(tau_corr_axis, x_cor_2_av, x_cor_2_err, color='red',
                 fmt='o', label=r"$<x^{2}(0)x^{2}(\tau)>$")
    ax1.errorbar(tau_corr_axis, x_cor_3_av, x_cor_3_err, color='green',
                 fmt='o', label=r"$<x^{3}(0)x^{3}(\tau)>$")
    ax1.legend(loc=1)

    fig1.savefig("Graph_corr_fun.png", dpi=200)

    # Print log correlation function plot
    fig2 = plt.figure(2, figsize=(8, 2.5), facecolor="#f1f1f1")
    ax2 = fig2.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    ax2.set_xlabel(r"$\tau$")
    ax2.set_ylabel(r"$-d \log \big( <x^{n}(0) x^{n}(\tau)> \big)/ \, d\tau$")

    ax2.errorbar(tau_corr_axis[0:(n_point-1)],
                 derivative_log_corr_funct,
                 derivative_log_corr_funct_err,
                 color='blue',
                 fmt='o',
                 label=r"$-d \log \big( <x(0) x(\tau)> \big) / \, d\tau$")
    ax2.errorbar(tau_corr_axis[0:(n_point-1)],
                 derivative_log_corr_funct_2,
                 derivative_log_corr_funct_2_err,
                 color='red',
                 fmt='o',
                 label=r"$-d \log \big( <x^2(0) x^2(\tau)> \big) / \, d\tau$")
    ax2.errorbar(tau_corr_axis[0:(n_point-1)],
                 derivative_log_corr_funct_3,
                 derivative_log_corr_funct_3_err,
                 color='green',
                 fmt='o',
                 label=r"$-d \log \big( <x^3(0) x^3(\tau)> \big) / \, d\tau$")
    ax2.set_ylim((0.0, 5.0))
    ax2.legend(loc=1)

    fig2.savefig("Graph_log_corr_fun.png", dpi=200)

    # Read the output file in order to print the histogram
    with open('hist.dat', 'rb') as reader:
        # Binary content of the file, we saved data as double
        hist_raw = reader.read((n_mc_sweeps - n_equil) * n_lattice * 8)

    x_config_hist = np.array(struct.unpack(
            'd'*(n_mc_sweeps-n_equil)*n_lattice, hist_raw))

    # Print histogram for wave function dist.
    fig3 = plt.figure(3, figsize=(8, 2.5), facecolor="#f1f1f1")
    ax3 = fig3.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    ax3.set_xlabel("x")
    ax3.set_ylabel("P(x)")

    # number of bins
    n_bin_hist = 100

    ax3.hist(x_config_hist, n_bin_hist,
             (-1.5*x_potential_minimum, 1.5*x_potential_minimum),
             density=True,
             color='b')

    fig3.savefig("Hist_prob_dens.png", dpi=200)

    plt.show()


if __name__ == '__main__':

    monte_carlo_ao(1.4,  # x_potential_minimum
                   800,  # n_lattice
                   0.05,  # dtau
                   100,  # n_equil
                   20000,  # n_mc_sweeps
                   0.45,  # delta_x
                   20,  # n_point
                   5,  # n_meas
                   False)
