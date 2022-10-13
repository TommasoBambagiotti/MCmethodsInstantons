# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:12:54 2022

@author: Federico
"""
import time
import random as rnd
import numpy as np
import matplotlib.pyplot as plt

def free_energy_harmonic_osc(beta, w_omega0):
    '''
    Return the Free Helmoltz energy for an harmonic oscillator
    '''
    return -np.log(2.0 * np.sinh(beta * w_omega0 / 2.0)) / beta


def stat_av_var(observable, observable_sqrd, n_data):
    '''
    Evaluate the average and the variance of the average of a set of data,
    expressed in an array, directly as the sum and the sum of squares.
    We use the formula Var[<O>] = (<O^2> - <O>^2)/N
    '''
    # Control
    n_array = 0

    if observable.size == observable_sqrd.size:
        n_array = observable.size
    else:
        return None, None

    observable_av = observable / n_data

    observable_err = np.empty((n_array), float)
    var_temp = 0.0

    for i_array in range(n_array):
        var_temp = observable_sqrd[i_array]/(n_data * n_data) - \
            pow(observable_av[i_array], 2) / n_data
        if var_temp > 0.0:
            observable_err[i_array] = var_temp  # INPUT OF THE VARIANCES
        else:
            observable_err[i_array] = 0.0

    return observable_av, observable_err


def monte_carlo_ao_switching(x_potential_minimum,  # potential well position
                             n_lattice,  # size of the grid
                             dtau,  # grid spacing in time
                             n_equil,  # equilibration sweeps
                             n_mc_sweeps,  # monte carlo sweeps
                             delta_x,  # width of gauss. dist. for update x
                             n_switching,  #
                             w_omega0,  #
                             icold):  # cold/hot start
    '''
    Solve the anharmonic oscillator through
    Monte Carlo technique on an Euclidian Axis
    '''
    # Variables for switching algorithm
    d_alpha = 1.0 / n_switching

    delta_s_alpha = np.zeros((2*n_switching + 1))
    delta_s_alpha2 = np.zeros((2*n_switching + 1))


    # x position along the tau axis
    x_config = np.zeros((n_lattice + 1))

    # Now the principal cycle is over the coupling constant alpha
    for i_switching in range(2 * n_switching + 1):

        if icold is True:
            for i in range(1, n_lattice):
                x_config[i] = - x_potential_minimum
        else:
            for i in range(1, n_lattice):
                x_config[i] = 2 * \
                    rnd.uniform(0., 1.) * x_potential_minimum \
                    - x_potential_minimum
        # impose periodic boundary conditions
        x_config[0] = x_config[n_lattice - 1]
        x_config[n_lattice] = x_config[1]


        if i_switching <= n_switching:
            a_alpha = i_switching * d_alpha
        else:
            a_alpha = 2.0 - (i_switching) * d_alpha

        # Monte Carlo sweeps

        # Counters for different parts
        n_accept = 0
        n_hit = 0
        n_config = n_mc_sweeps - n_equil

        # Monte Carlo
        for i_mc in range(n_mc_sweeps):

            delta_s_alpha_temp = 0.0

            for i in range(1, n_lattice):

                n_hit += 1

                v0 = pow(w_omega0 * x_config[i], 2) / 4.0
                v1 = pow((pow(x_config[i], 2) - pow(x_potential_minimum, 2)), 2)

                action_loc_old = (
                    pow((x_config[i] - x_config[i-1])/(2 * dtau), 2) +
                    pow((x_config[i+1] - x_config[i])/(2 * dtau), 2) +
                    (v1 - v0) * a_alpha + v0  # Potential V_alpha
                    ) * dtau

                x_new = x_config[i] + rnd.gauss(0, delta_x)

                v0 = pow(w_omega0 * x_new, 2) / 4.0
                v1 = pow((pow(x_new, 2) - pow(x_potential_minimum, 2)), 2)

                action_loc_new = (
                    pow((x_new - x_config[i-1])/(2*dtau), 2) +
                    pow((x_config[i+1] - x_new)/(2*dtau), 2) +
                    (v1 - v0) * a_alpha + v0  # Potential V_alpha
                    ) * dtau

                delta_action = action_loc_new - action_loc_old

                # we put a bound on the value of delta_S
                # because we need the exp.
                delta_action = max(delta_action, -70.0)
                delta_action = min(delta_action, 70.0)
                # Metropolis question:
                if np.exp(-delta_action) > rnd.uniform(0., 1.):
                    x_config[i] = x_new
                    n_accept += 1

            x_config[0] = x_config[n_lattice - 1]
            x_config[n_lattice] = x_config[1]

            if i_mc >= n_equil:
                delta_v_temp = 0.0

                for j in range(n_lattice):
                    v0 = pow(w_omega0 * x_config[i], 2) / 4.0
                    v1 = pow((pow(x_config[i], 2) - pow(x_potential_minimum, 2)), 2)
                    delta_v_temp += (v1 - v0) * dtau

                delta_s_alpha_temp += delta_v_temp

            delta_s_alpha[i_switching] += delta_s_alpha_temp
            delta_s_alpha2[i_switching] += pow(delta_s_alpha_temp, 2)

        # Monte Carlo End
        # Control Acceptance ratio
        #print('acceptance ratio = {n}'.format(n=n_accept/n_hit))

    delta_s_alpha_av, delta_s_alpha_err = stat_av_var(delta_s_alpha, delta_s_alpha2, n_config)
    #delta_s_alpha_err = np.power(delta_s_alpha_err,2)

    # Integration over alpha in <deltaS>
    integral_01 = np.trapz(delta_s_alpha_av[0:(n_switching + 1)], dx = d_alpha)
    integral_10 = np.trapz(delta_s_alpha_av[n_switching:(2*n_switching + 1)], dx = d_alpha)

    integral_01_err = np.trapz(delta_s_alpha_err[0:(n_switching + 1)], dx = d_alpha)
    integral_10_err = np.trapz(delta_s_alpha_err[n_switching:(2*n_switching + 1)], dx = d_alpha)

    trapezoidal_error = np.abs(
        (delta_s_alpha_av[n_switching] - delta_s_alpha_av[n_switching-1] +
          delta_s_alpha_av[1] - delta_s_alpha_av[0]) /
        (d_alpha * 12 * n_switching * n_switching ) )

    propagation_error = np.sqrt(integral_01_err + integral_10_err)
    hysteresis_error = np.abs(integral_01 - integral_10)


    with open('data.txt','a') as writer:
       np.savetxt(writer, delta_s_alpha_av)
       writer.write('Errors:\n')
       np.savetxt(writer, delta_s_alpha_err)
       writer.write('prop error:\n')
       writer.write(str(propagation_error))
       writer.write('\ntrap error:\n')
       writer.write(str(trapezoidal_error))
       writer.write('\nhys error:\n')
       writer.write(str(hysteresis_error))
       writer.write('\nintegral 0->1:\n')
       writer.write(str(integral_01))
       writer.write('\nintegral 1->0:\n')
       writer.write(str(integral_10))
       writer.write('\n#######################\n')


    return -(integral_01 + integral_10) / 2.0, np.sqrt(pow(propagation_error, 2) +
                                                      pow(hysteresis_error, 2) +
                                                      pow(trapezoidal_error, 2))


def free_energy_anharm(n_beta):

    rnd.seed(time.time())

    # Free Helmoltz energy for the anharmonic oscillator
    beta_axis = np.linspace(1.0, 8.0, n_beta - 1)
    temperature_axis = 1.0 / beta_axis

    dtau = 0.05

    free_energy = np.empty((n_beta), float)
    free_energy_err = np.empty((n_beta), float)

    for i_beta in range(n_beta):

        n_lattice = int(beta_axis[i_beta]/dtau)

        free_energy[i_beta], free_energy_err[i_beta] = \
        monte_carlo_ao_switching(1.4,
                                 n_lattice,  # n_lattice
                                 dtau,  # dtau
                                 100,  # n_equil
                                 100000,  # n_mc_sweeps-
                                 0.45,
                                 20,
                                 1.4 * 4,
                                 False)
        free_energy[i_beta] /= beta_axis[i_beta]
        free_energy_err[i_beta] /= beta_axis[i_beta]
        free_energy[i_beta] += free_energy_harmonic_osc(beta_axis[i_beta], 1.4*4)

    print(free_energy)

    fig1 = plt.figure(1, facecolor="#f1f1f1")
    ax1 = fig1.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    ax1.set_xlabel(r"$T$")
    ax1.set_ylabel(r"$F$")

    ax1.errorbar(temperature_axis,
                  free_energy,
                  free_energy_err,
                  color='b',
                  fmt='.')

    ax1.set_xscale('log')

    temperature_axis = np.loadtxt('temp.txt', float, delimiter = '\n')
    free_energy = np.loadtxt('free_energy.txt', float, delimiter = '\n')

    ax1.plot(temperature_axis,
            free_energy,
            color='green')


    ax1.plot(temperature_axis,
            free_energy_harmonic_osc(1.0/temperature_axis, 1.4 * 4),
            color='red')

    fig1.savefig("Free_energy_mc.png", dpi = 200)

    plt.show()


if __name__ == '__main__':

    free_energy_anharm(4)
