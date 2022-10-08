
import struct
import time
import random as rnd
import numpy as np
import matplotlib.pyplot as plt


def monte_carlo_ao(x_potential_minimum,
                   n_lattice,
                   dtau,
                   n_equil,
                   n_mc_sweeps,
                   delta_x,
                   n_point,
                   n_meas,
                   n_print,
                   icold):
    '''Solve the anharmonic oscillator through Monte Carlo technique on an Euclidian Axis'''
    if n_mc_sweeps < n_equil:
        print("too few Monte Carlo sweeps/ N_equilib > N_Monte_Carlo")

    # Correlation
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

    # x position along the tau axis
    x_config = np.zeros((n_lattice+1))

    #histogram for the wave function

    n_bin_hist = 100

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
                rnd_uniform.uniform(0., 1.)*x_potential_minimum - x_potential_minimum

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
                #we apply Metropolis algorithm for each site tau_k:
                #Secundary cycle
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
                    pow(x_new * x_new-x_potential_minimum*x_potential_minimum, 2)
                )*dtau

                delta_action = action_loc_new - action_loc_old

                # we put a bound on the value of delta_S because we need the exp.
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

        #Rest of the MC sweeps
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
                    pow(x_new * x_new-x_potential_minimum*x_potential_minimum, 2)
                )*dtau

                delta_action = action_loc_new - action_loc_old

                # we put a bound on the value of delta_S because we need the exp.
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
                pot_en_temp = pow(
                    x_config[j] * x_config[j]-x_potential_minimum*x_potential_minimum, 2)

                kin_en_total += kin_en_temp
                pot_en_total += pot_en_temp
                action_total += (kin_en_temp + pot_en_temp)*dtau

                kin_en_temp = 0.0
                pot_en_temp = 0.0

            # output control at each n_print step

            #if( (k % n_print) == 0 ):
                #print('acceptance ratio = {n}'.format(n= n_accept/n_hit))

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
    #Evaluate averages and other stuff, maybe we can create a function
    var = 0.0
    var2 = 0.0
    var3 = 0.0

    for i_corr in range(n_point):
        x_cor_av[i_corr] = x_cor_sum[i_corr]/n_corr
        var = x2_cor_sum[i_corr]/(n_corr*n_corr) - \
            pow(x_cor_av[i_corr], 2)/n_corr

        x_cor_2_av[i_corr] = x_cor_2_sum[i_corr]/n_corr
        var2 = x2_cor_2_sum[i_corr] / (n_corr*n_corr) - \
            pow(x_cor_2_av[i_corr], 2)/n_corr

        x_cor_3_av[i_corr] = x_cor_3_sum[i_corr]/n_corr
        var3 = x2_cor_3_sum[i_corr] / (n_corr*n_corr) - \
            pow(x_cor_3_av[i_corr], 2)/n_corr

        if var > 0.0:
            x_cor_err[i_corr] = pow(var, 1/2)
        else:
            x_cor_err[i_corr] = 0.

        if var2 > 0.0:
            x_cor_2_err[i_corr] = pow(var2, 1/2)
        else:
            x_cor_2_err[i_corr] = 0.

        if var3 > 0.0:
            x_cor_3_err[i_corr] = pow(var3, 1/2)
        else:
            x_cor_3_err[i_corr] = 0.

    #Control Acceptance ratio
    print('acceptance ratio = {n}'.format(n=n_accept/n_hit))

    # In order for matplotlib to print text in LaTeX
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    #Print correlation function plot
    tau_corr_axis = np.arange(0, n_point*dtau, dtau)
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

    # Read the output file in order to print the histogram

    with open('hist.dat', 'rb') as reader:
    # Binary content of the file, we saved data as double
        hist_raw = reader.read((n_mc_sweeps-n_equil)*n_lattice*8)

    x_config_hist = np.array(struct.unpack(
            'd'*(n_mc_sweeps-n_equil)*n_lattice, hist_raw))

    #Print histogram for wave function dist.
    fig2 = plt.figure(2)

    ax2 = fig2.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    ax2.set_xlabel("x")
    ax2.set_ylabel("P(x)")

    ax2.hist(x_config_hist, n_bin_hist,
            (-1.5*x_potential_minimum, 1.5*x_potential_minimum),
             density=True,
             color='b')

    fig2.savefig("Hist_prob_dens.png", dpi=200)

    plt.show()

if __name__ == '__main__':

    monte_carlo_ao(1.4, 800, 0.05, 100, 10000, 0.5, 20, 5, 100, False)
