# This is a sample Python script.

import matplotlib.pyplot as plt
import numpy as np
from numpy import math
from numpy import linalg as LA
from numpy.polynomial import hermite


def anharmonic_oscillator_diag(f, n_dim, w0):

    #Units of measurement: h_bar=1, m=1/2, lambda=1

    # parameters
    n_max = n_dim + 4
    #n_max = 100
    eps = pow(1, -30)

    # euclidean time coordinate
    tau_max = 2.5
    n_tau = 100
    dtau = tau_max / n_tau


    tau_axis = np.arange(0, tau_max, dtau)

    # space coordinate
    x_max = 2 * f
    n_x = 100
    dx = (2 * x_max) / n_x

    x_axis = np.arange(-x_max, x_max, dx)
    # x_axis = np.append(x_axis,x_max)

    # hamiltonian
    h = np.zeros((n_max, n_max))

    # eigenvalues
    e = np.zeros(n_max)

    # eigenvectors coeff.
    v = np.zeros((n_max, n_max))

    # ground state wave function
    psi = np.zeros(n_max)

    # density
    rho = np.zeros(n_dim)
    rho2 = np.zeros(n_dim)
    rho3 = np.zeros(n_dim)

    # correlation function

    corr_funct = np.zeros(n_tau)
    corr_funct2 = np.zeros(n_tau)
    corr_funct3 = np.zeros(n_tau)

    derivative_log_corr_funct = np.zeros(n_tau)
    derivative_log_corr_funct2 = np.zeros(n_tau)
    derivative_log_corr_funct3 = np.zeros(n_tau)

    # mass???

    m = 0.5

    # anharmonic potential coeff.
    A = 1.0
    B = -2.0 * pow(f, 2) - m * pow(w0, 2) / 2
    C = pow(f, 4)

    c_w0 = pow(m * w0, -1 / 2)  # normalizzazione?

    c22 = c_w0 * c_w0 / 2
    c44 = pow(c_w0, 4) / 4
    c68 = pow(c_w0, 6) / 8

    # Hamiltonian, symmetric n_dim x n_dim matrix

    for n in range(n_dim):
        # n=k-1
        # <n|h|n>
        x4 = c44 * 3 * (pow(n + 1, 2) + pow(n, 2))
        x2 = c22 * (2 * n + 1)
        e0 = w0 * (n + 0.5) + C

        h[n, n] = A * x4 + B * x2 + e0

        # <n|h|n+2>
        x4 = c44 * pow(((n + 1) * (n + 2)), 1 / 2) * (4 * n + 6)
        x2 = c22 * pow(((n + 1) * (n + 2)), 1 / 2)

        hh = A * x4 + B * x2
        h[n, n + 2] = hh
        h[n + 2, n] = hh

        # <n|h|n+4>
        x4 = c44 * pow(((n + 1) * (n + 2) * (n + 3) * (n + 4)), 1 / 2)

        h[n, n + 4] = x4
        h[n + 4, n] = x4

    # Diagononilzation of the Hamiltonian
    e, v = LA.eigh(h)

    # Eigenvalues and Eigenvectors check
    #print(e)

    # WE consider also the positive eigenvalues: for some reason the first 4 e are < 0.
    # So we neglect the first 4 eigenvectors and consider the groundstate v[:,4]=> because the eigenvectors are the columns.
    e = np.delete(e, [0, 1, 2, 3])
    v = np.delete(v, [0, 1, 2, 3], 1)  # we eliminate the first 4 columns

    print(e)

    # Evaluate the energy dist. rho, rho2, rho2 and matrix elements < 0|x^i|n > , i= 1,2,3
    # Control the results and the cycle indices
    # We use the convenction for the ladder operators: a+ = (mw/2h)^1/2 (x^ +i/mw p^) and a = (mw/2h)^1/2 (x^ - i/mw p^)
    for n in range(n_dim):
        # C_n = | <0|x|n> |^2
        # D_n = | <0|x^2|n> |^2
        # E_n = | <0|x^3|n> |^2

        C_n = 0.0
        D_n = 0.0
        E_n = 0.0

        for k in range(n_dim):
            k_minus_3 = max(k - 3, 0)
            k_minus_2 = max(k - 2, 0)
            k_minus_1 = max(k - 1, 0)
            k_plus_1 = min(k + 1, n_dim - 1)
            k_plus_2 = min(k + 2, n_dim - 1)
            k_plus_3 = min(k + 3, n_dim - 1)

            C_n += (pow(k + 1, 1 / 2) * v[k_plus_1, 0] + pow(k, 1 / 2) * v[k_minus_1, 0]) * v[k, n]

            D_n += (pow(k * (k - 1), 1 / 2) * v[k_minus_2, 0] + (2 * k + 1) * v[k, 0] + pow((k + 1) * (k + 2), 1 / 2) *
                    v[k_plus_2, 0]) * v[k, n]

            E_n += (pow(k * (k - 1) * (k - 2), 1 / 2) * v[k_minus_3, 0] + 3 * k * pow(k, 1 / 2) * v[
                k_minus_1, 0] + 3 * (k + 1) * pow(k + 1, 1 / 2) * v[k_plus_1, 0] + pow((k + 1) * (k + 2) * (k + 3),
                                                                                       1 / 2) * v[k_plus_3, 0]) * v[
                       k, n]



        rho[n] = c22 * pow(C_n, 2)
        rho2[n] = c44 * pow(D_n, 2)
        rho3[n] = c68 * pow(E_n, 2)

    # Groundstate wave function and its properties

    x = 0.0

    # Definition of a basis of Hermite pol.

    Hermite_coeff = np.zeros((n_dim))

    xnorm = 0.0  # norm with Trapezoidal rule
    xnorm1 = 0.0  # norm with naive sum dx * psi_x^2
    xnorm2 = 0.0  # norm of psi(0)

    Psi_X2 = np.zeros((n_x))
    Psi0_ar = np.zeros((n_x))

    for k in range(n_x):
        x = -x_max + k * dx
        Psi_x = 0.0
        val = 1.0

        for n in range(n_dim):
            # follow the formula for the Eig state of the harmonic osc. : 1/(l^1/2 pi^1/4 (2^n * n!)^1/2) * exp(-x^2/l^2 /2) * H_n(x/l)
            # length scale of the harmonic oscillator: l=c=(1/mw0)^1/2
            val = pow(np.pi * c_w0 * c_w0, -1 / 4)

            val *= pow(2.0, -n / 2)  # 1^ normalization
            # val *= pow(2.0, -3*n/2) # 2^ normalization

            val *= pow(math.factorial(n), -1 / 2)  # 1 normalization
            # val *= pow(math.factorial(n), -1)

            val *= np.exp(-x * x / (2.0 * c_w0 * c_w0))

            Hermite_coeff[n] = val
            psi[n] = hermite.hermval(x / c_w0, Hermite_coeff)

            Hermite_coeff[n] = 0.0
            val = 1.0

            Psi_x += v[n, 0] * psi[n]  # * termine corr[n]


        Psi_X2[k] = Psi_x * Psi_x

        # Compare to the simple model
        Psi_p = pow((2.0 * f / np.pi), 1 / 4) * np.exp(-f * pow(x - f, 2))
        Psi_m = pow((2.0 * f / np.pi), 1 / 4) * np.exp(-f * pow(x + f, 2))
        Psi_0 = (Psi_p + Psi_m) * pow(2, -1 / 2)

        Psi0_ar[k] = Psi_0 * Psi_0

        # Check Normalization
        xnorm1 += dx * pow(Psi_x, 2)

    # xnorm = np.trapz(Psi_X2, x_axis, dx)
    # print('xnorm for Psi_x in dim = {first}'.format(first=n_dim))
    # print(xnorm)
    # print('xnorm1 for Psi_x in dim = {first}'.format(first=n_dim))
    # print(xnorm1)

    # Map potential and Psi_x vs. (Psi+ + Psi-)
    Fig1 = plt.figure(1)

    V_x = np.zeros((n_x))

    for i in range(n_x):
        V_x[i] = pow(x_axis[i] * x_axis[i] - f * f, 2)

    for i in range(5):
        plt.axhline(y=e[i], color='g', linestyle='--')

    plt.plot(x_axis, V_x)

    Fig2 = plt.figure(2)

    plt.plot(x_axis, Psi_X2, 'b', x_axis, Psi0_ar, '-r')

    # Coordinate space correlator

    # Groundstate eigenvalue

    E_0 = e[0]

    # Temporal coord.
    tau = 0.0
    # temporary var
    xCor_temp1 = 0.0
    xCor_temp2 = 0.0
    xCor_temp3 = 0.0

    for i in range(n_tau):
        tau = i * dtau
        for j in range(n_dim):
            xCor_temp1 += rho[j] * np.exp(-(e[j] - E_0) * tau)
            xCor_temp2 += rho2[j] * np.exp(-(e[j] - E_0) * tau)
            xCor_temp3 += rho3[j] * np.exp(-(e[j] - E_0) * tau)

        corr_funct[i] = xCor_temp1
        corr_funct2[i] = xCor_temp2
        corr_funct3[i] = xCor_temp3

        xCor_temp1 = 0.0
        xCor_temp2 = 0.0
        xCor_temp3 = 0.0



    Fig3 = plt.figure(3)

    plt.plot(tau_axis, corr_funct, 'b', linestyle='--')
    plt.plot(tau_axis, corr_funct2, color='r', linestyle='-')
    plt.plot(tau_axis, corr_funct3, color='g', linestyle='dashdot')
    plt.show()

    # Logarithmic derivative of the correlators

    # Python methods
    py_derivative_log_corr_funct = np.gradient(-corr_funct,dtau)
    py_derivative_log_corr_funct2 = np.gradient(-corr_funct2, dtau)
    py_derivative_log_corr_funct3 = np.gradient(-corr_funct3, dtau)

    plt.plot(tau_axis,py_derivative_log_corr_funct,color='blue', linestyle='--')
    plt.plot(tau_axis, py_derivative_log_corr_funct2, color='red',linestyle='-')
    plt.plot(tau_axis, py_derivative_log_corr_funct3, color='green', linestyle='dashdot')
    plt.axis([0, 1.5, 0, 8]) # plt.axis([xmin,xmax,ymin,ymax] set axes limits
    plt.title('d/dt[log<O(0)O(t)>] - Numpy')
    plt.show()

    # Forward difference

    fd_derivative_log_corr_funct = np.zeros(n_tau) #da spostare all'inizio del codice
    fd_derivative_log_corr_funct2 = np.zeros(n_tau)
    fd_derivative_log_corr_funct3 = np.zeros(n_tau)

    for i in range(n_tau-1):
        fd_derivative_log_corr_funct[i] = - (np.log(corr_funct[i+1])-np.log(corr_funct[i]))/dtau
        fd_derivative_log_corr_funct2[i] = - (np.log(corr_funct2[i + 1]) - np.log(corr_funct2[i])) / dtau
        fd_derivative_log_corr_funct3[i] = - (np.log(corr_funct3[i + 1]) - np.log(corr_funct3[i])) / dtau

    plt.plot(tau_axis,fd_derivative_log_corr_funct,color='blue', linestyle='--')
    plt.plot(tau_axis, fd_derivative_log_corr_funct2, color='red',linestyle='-')
    plt.plot(tau_axis, fd_derivative_log_corr_funct3, color='green', linestyle='dashdot')
    plt.axis([0, 1.5, 0, 8]) # plt.axis([xmin,xmax,ymin,ymax] set axes limits
    plt.title('d/dt[log<O(0)O(t)>] - Forward difference')
    plt.show()

    # Analytic formula

    # Time variable
    tau = 0.0

    # Temporary variables
    derivative_xCor_temp1 = 0.0
    derivative_xCor_temp2 = 0.0
    derivative_xCor_temp3 = 0.0

    for i in range(n_tau):

        tau = i*dtau

        for j in range(n_dim):

            derivative_xCor_temp1 = derivative_xCor_temp1 + rho[j] * (e[j] - E_0) * np.exp(-(e[j] - E_0) * tau)
            derivative_xCor_temp2 = derivative_xCor_temp2 + rho2[j] * (e[j] - E_0) * np.exp(-(e[j] - E_0) * tau)
            derivative_xCor_temp3 = derivative_xCor_temp3 + rho3[j] * (e[j] - E_0) * np.exp(-(e[j] - E_0) * tau)

        #Derivatives of the log of the correlation functions
        derivative_log_corr_funct[i] = derivative_xCor_temp1 / corr_funct[i]
        derivative_log_corr_funct2[i] = derivative_xCor_temp2 / (corr_funct2[i] - rho2[0]) #subtract the offset |<0|x^2|0>|^2 to have the right convergence
        derivative_log_corr_funct3[i] = derivative_xCor_temp3 / corr_funct3[i]

        derivative_xCor_temp1 = 0.0
        derivative_xCor_temp2 = 0.0
        derivative_xCor_temp3 = 0.0


    plt.plot(tau_axis,derivative_log_corr_funct,color='blue', linestyle='--')
    plt.plot(tau_axis, derivative_log_corr_funct2, color='red',linestyle='-')
    plt.plot(tau_axis, derivative_log_corr_funct3, color='green', linestyle='dashdot')
    plt.axis([0, 1.5, 0, 8]) # plt.axis([xmin,xmax,ymin,ymax] set axes limits
    plt.title('d/dt[log<O(0)O(t)>] - Analytic formula')
    plt.show()


if __name__ == '__main__':
    anharmonic_oscillator_diag(1.4, 50, 4 * 1.4)