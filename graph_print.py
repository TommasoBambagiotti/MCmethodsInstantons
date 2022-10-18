 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 18:48:01 2022

@author: federico_
"""
import numpy as np
import matplotlib.pyplot as plt

def print_graph():
    plt.style.use('ggplot')
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
    
    fig1 = plt.figure(1, facecolor="#f1f1f1")
    ax1 = fig1.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")

    ax1.set_xlabel(r"$T$")
    ax1.set_ylabel(r"$F$")

    temperature_array = np.loadtxt('./output_data/output_monte_carlo_switching/temperature.txt',
                                  float, delimiter = '\n')
    f_energy = np.loadtxt('./output_data/output_monte_carlo_switching/free_energy_mc.txt',
                         float, delimiter = '\n')
    f_energy_err = np.loadtxt('./output_data/output_monte_carlo_switching/free_energy_mc_err.txt',
                             float, delimiter = '\n')
    
    
    ax1.errorbar(temperature_array,
                 f_energy,
                 f_energy_err,
                 color='b',
                 fmt='.',
                 capsize = 5,
                 elinewidth = 2)

    ax1.set_xscale('log')

    temperature_axis = np.loadtxt('./output_data/output_diag/temperature.txt', float, delimiter = '\n')
    free_energy = np.loadtxt('./output_data/output_diag/free_energy.txt', float, delimiter = '\n')

    ax1.plot(temperature_axis,
            free_energy,
            color='green')

    fig1.savefig('free_energy.png', dpi = 300)    
    
    
    fig2 = plt.figure(1, facecolor="#f1f1f1")
    ax2 = fig2.add_axes((0.1, 0.1, 0.8, 0.8), facecolor="#e1e1e1")


    # tau_array = np.loadtxt('./output_data/output_monte_carlo_cooling/tau_array.txt',
    #                               float, delimiter = '\n')
    # config = np.loadtxt('./output_data/output_monte_carlo_cooling/configuration.txt',
    #                      float, delimiter = '\n')
    # config_cool= np.loadtxt('./output_data/output_monte_carlo_cooling/configuration_cooled.txt',
    #                          float, delimiter = '\n')
    
    # ax2.plot(tau_array, config, color = 'blue')
    # ax2.plot(tau_array, config_cool, color = 'red')
    
    plt.show()
    
    
print_graph()