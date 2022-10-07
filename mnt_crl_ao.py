import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import random as rnd

import time

# Solve the anharmonic oscillator through Monte Carlo technique on an Euclidian Axis 

def Monte_Carlo_AO (f,nE_lattice, dtau, nEquil, nMCsweeps, delta_x, nxCorr, nMeas, nPri,icold):

    if(nMCsweeps < nEquil):
        print("too few Monte Carlo sweeps/ N_equilib > N_Monte_Carlo")

    # tau spacing
    #tau_max = nE_lattice* dtau

    tau_axis = np.arange(0,dtau*nE_lattice,dtau)

    # Correlation

    xCor_sum = np.zeros((nxCorr))
    xCor_2_sum = np.zeros((nxCorr))
    xCor_3_sum = np.zeros((nxCorr))

    x2Cor_sum = np.zeros((nxCorr))
    x2Cor_2_sum = np.zeros((nxCorr))
    x2Cor_3_sum = np.zeros((nxCorr))

    # Averages and errors

    xCor_av = np.zeros((nxCorr))
    xCor_err = np.zeros((nxCorr))

    xCor_2_av = np.zeros((nxCorr))
    xCor_2_err = np.zeros((nxCorr))

    xCor_3_av = np.zeros((nxCorr))
    xCor_3_err = np.zeros((nxCorr))


    # x position along the tau axis
    x_Config = np.zeros((nE_lattice+1))

    #histogram for the wave function

    x_nHist = 100
    x_Config_hist = np.empty((nE_lattice),float)
    x_Config_hist_temp = np.empty((nE_lattice),float)

    # initialize the lattice

    rnd_Uniform = rnd.Random(time.time())

    rnd_Gauss = rnd.Random(time.time())


    if(icold == True):
        for i in range(nE_lattice):
            x_Config[i] = -f

    else:
        rnd.seed()
        for i in range(nE_lattice):
            x_Config[i] = 2*rnd_Uniform.uniform(0.,1.)*f -f

    # impose periodic boundary conditions and understand the way in which they are imposed in the reference file
    x_Config[nE_lattice-1] = x_Config[0]
    x_Config[nE_lattice] = x_Config[1]

    #plt.plot(tau_axis, x_axis)
    #plt.show()

    # Evaluate the initial action

    S_initial = 0.0

    for i in range(nE_lattice-1):
        T = pow((x_Config[i+1] - x_Config[i])/(2*dtau),2)
        V = pow(x_Config[i]* x_Config[i]-f*f,2)
        S_initial += (T+V)*dtau

    print(S_initial)

    # Monte Carlo sweeps: Principal cycle

    nAccept = 0
    nHit = 0
    nConfig = 0
    nCorr = 0

    # Local exp(-S) for each point x_k


    S_total = 0.0 # total action
    T = 0.0
    V = 0.0


    # Equilibrization cycle

    for k in range (nEquil):

        for i in range(1,nE_lattice): #we apply Metropolis algorithm for each site tau_k: Secundary cycle

            nHit += 1

            t0m = pow((x_Config[i] - x_Config[i-1])/(2*dtau),2)
            t0p = pow((x_Config[i+1] - x_Config[i])/(2*dtau),2)
            v = pow(x_Config[i]* x_Config[i]-f*f,2)

            sLoc_now = (t0m + t0p + V)*dtau

            x_New = x_Config[i] + rnd_Gauss.gauss(0,delta_x)

            t0m = pow((x_New - x_Config[i-1])/(2*dtau),2)
            t0p = pow((x_Config[i+1] - x_New)/(2*dtau),2)
            v = pow(x_New* x_New-f*f,2)

            sLoc_new = (t0m + t0p + v)*dtau

            delta_sLoc = sLoc_new - sLoc_now

            P = rnd_Uniform.uniform(0.,1.)

            # we put a bound on the value of delta_S because we need the exp.
            delta_sLoc = max(delta_sLoc,-70.0)
            delta_sLoc = min(delta_sLoc,70.0)
            # Metropolis question:
            if( np.exp(-delta_sLoc) > P):
                x_Config[i] = x_New
                nAccept += 1


        x_Config[0] = x_Config[nE_lattice-1]
        x_Config[nE_lattice] = x_Config[1]

        # Calculate the total action:

        for j in range (nE_lattice-1):
            T = pow((x_Config[j+1] - x_Config[j])/(2*dtau),2)
            V = pow(x_Config[j]* x_Config[j]-f*f,2)
            S_total += (T+V)*dtau

        S_total = 0.0

    #Rest of the MC sweeps

    for k in range(nEquil, nMCsweeps):

        nConfig += 1

        for i in range(1,nE_lattice): #we apply Metropolis algorithm for each site tau_k: Secundary cycle

            nHit += 1

            t0m = pow((x_Config[i] - x_Config[i-1])/(2*dtau),2)
            t0p = pow((x_Config[i+1] - x_Config[i])/(2*dtau),2)
            v = pow(x_Config[i]* x_Config[i]-f*f,2)

            sLoc_now = (t0m + t0p + V)*dtau

            x_New = x_Config[i] + rnd_Gauss.gauss(0,delta_x)

            t0m = pow((x_New - x_Config[i-1])/(2*dtau),2)
            t0p = pow((x_Config[i+1] - x_New)/(2*dtau),2)
            v = pow(x_New* x_New-f*f,2)

            sLoc_new = (t0m + t0p + v)*dtau

            delta_sLoc = sLoc_new - sLoc_now

            P = rnd_Uniform.uniform(0.,1.)

            # we put a bound on the value of delta_S because we need the exp.
            delta_sLoc = max(delta_sLoc,-70.0)
            delta_sLoc = min(delta_sLoc,70.0)
            # Metropolis question:
            if( np.exp(-delta_sLoc) > P):
                x_Config[i] = x_New
                nAccept += 1

        x_Config[0] = x_Config[nE_lattice-1]
        x_Config[nE_lattice] = x_Config[1]


        x_Config_hist_temp = x_Config_hist

        x_Config_hist = np.append(x_Config_hist_temp, x_Config[0:(nE_lattice-1)])

        # Calculate the total action:

        for j in range (nE_lattice-1):
            T = pow((x_Config[j+1] - x_Config[j])/(2*dtau),2)
            V = pow(x_Config[j]* x_Config[j]-f*f,2)
            S_total += (T+V)*dtau


        # output control at each nPri step

        #if( (k % nPri) == 0 ):
            #print('acceptance ratio = {n}'.format(n= nAccept/nHit))

        S_total = 0.0

        for i_meas in range (nMeas):

            nCorr += 1

            i_p0 = int( (nE_lattice - nxCorr) * rnd_Uniform.uniform(0.,1.) )

            x_0 = x_Config[i_p0]

            for i_Corr in range (nxCorr):

                x_1 = x_Config[i_p0 + i_Corr]

                #x_1 = x_Config[i_Corr]

                xCor = x_0 * x_1
                xCor_2 = pow(xCor,2)
                xCor_3 = pow(xCor,3)

                xCor_sum[i_Corr] += xCor
                xCor_2_sum[i_Corr] += xCor_2
                xCor_3_sum[i_Corr] += xCor_3

                x2Cor_sum[i_Corr] += pow(xCor,2)
                x2Cor_2_sum[i_Corr] += pow(xCor_2,2)
                x2Cor_3_sum[i_Corr] += pow(xCor_3,2)


# Monte Carlo End

    #Evaluate averages and other stuff, maybe we can create a function

    Var = 0.0
    Var2 = 0.0
    Var3 = 0.0

    for i_Corr in range (nxCorr):
        xCor_av[i_Corr] = xCor_sum[i_Corr]/nCorr
        Var = x2Cor_sum[i_Corr]/(nCorr*nCorr) - pow(xCor_av[i_Corr],2)/nCorr

        xCor_2_av[i_Corr] = xCor_2_sum[i_Corr]/nCorr
        Var2 = x2Cor_2_sum[i_Corr]/(nCorr*nCorr) - pow(xCor_2_av[i_Corr],2)/nCorr

        xCor_3_av[i_Corr] = xCor_3_sum[i_Corr]/nCorr
        Var3 = x2Cor_3_sum[i_Corr]/(nCorr*nCorr) - pow(xCor_3_av[i_Corr],2)/nCorr

        if(Var > 0.):
            xCor_err[i_Corr] = pow(Var,1/2)
        else:
            xCor_err[i_Corr] = 0.

        if(Var2 > 0.0):
            xCor_2_err[i_Corr] = pow(Var2,1/2)
        else:
            xCor_2_err[i_Corr] = 0.

        if(Var3 > 0.0):
            xCor_3_err[i_Corr] = pow(Var3,1/2)
        else:
            xCor_3_err[i_Corr] = 0.

    #Correlation function


    tau_corr_axis = np.arange(0,nxCorr*dtau,dtau)


    print('acceptance ratio = {n}'.format(n= nAccept/nHit))

    # In order for matplotlib to print text in LaTeX
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    Fig1 = plt.figure(1, figsize=(8, 2.5), facecolor="#f1f1f1")
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax1 = Fig1.add_axes((left,bottom,width,height), facecolor="#e1e1e1")

    ax1.set_xlabel(r"$\tau$")
    ax1.set_ylabel(r"$<x^{n}(0) x^{n}(\tau)>$")

    ax1.errorbar(tau_corr_axis,xCor_av,xCor_err,color = 'blue',fmt = 'o', label = r"$<x(0)x(\tau)>$")
    ax1.errorbar(tau_corr_axis,xCor_2_av,xCor_2_err,color = 'red',fmt = 'o', label = r"$<x^{2}(0)x^{2}(\tau)>$")
    ax1.errorbar(tau_corr_axis,xCor_3_av,xCor_3_err,color = 'green',fmt = 'o', label = r"$<x^{3}(0)x^{3}(\tau)>$")
    ax1.legend(loc = 1)

    Fig1.savefig("Graph_corr_fun.png", dpi = 200)

    Fig2 = plt.figure(2)

    ax2 = Fig2.add_axes((left,bottom,width,height), facecolor="#e1e1e1")

    ax2.set_xlabel("x")
    ax2.set_ylabel("P(x)")

    ax2.hist(x_Config_hist,x_nHist,(-1.5*f,1.5*f), density = True, color = 'b')

    Fig2.savefig("Hist_prob_dens.png", dpi = 200)

    plt.show()

if __name__ == '__main__':

        Monte_Carlo_AO(1.4,800,0.05,100,100000,0.5,20,5,100,False)
