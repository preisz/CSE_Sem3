#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal.windows
from matplotlib import rc

# rc('text', usetex=True)
from scipy.fft import rfft, rfftfreq

import scipy
from scipy.fft import rfft, rfftfreq

def gen_polarplots(data_path, filename, plt_freqList, title, indices=None, Herz = None):
    """ create a polar plot of the acoustic pressure at
    the radius of 1 m around the excitation and compare it to sound hard BC
    (at characteristic frequencies)"""

    ts = 1e-3  # Sample time in s
    plotname = "/home/petrar/PetraMaster/WS23/CSE_Sem3/FEM-Multiphys/Aucoustic/images/RigidvsNofilename"+ filename + ".png"

    flag_savePlots = True

    # - Read Mic data ----------------------------------------------------------------------------------

    coord = []
    p = []
    if indices is None:
        for i in range(1, len(plt_freqList) + 1):
            file = f"{data_path}/{filename}-{i}"
            coord.append(np.loadtxt(f"{data_path}/{filename}-{i}", usecols=[1,2], delimiter=',', dtype=float, skiprows=1))
            p.append(np.loadtxt(f"{data_path}/{filename}-{i}", usecols=[3], delimiter=',', dtype=float, skiprows=1))
    else:
        for i in indices:
            file = f"{data_path}/{filename}-{i}"
            coord.append(np.loadtxt(f"{data_path}/{filename}-{i}", usecols=[1,2], delimiter=',', dtype=float, skiprows=1))
            p.append(np.loadtxt(f"{data_path}/{filename}-{i}", usecols=[3], delimiter=',', dtype=float, skiprows=1))
        

    p = np.array(p)

    # - Calculate FFT ----------------------------------------------------------------------------------

    startStep = 0

    yf = []

    win = scipy.signal.windows.hann(p.shape[0] - startStep, True)

    for i in range(p.shape[1]):
        y = p[startStep:, i] * win
        yf.append(abs(rfft(y)))

    yf = np.array(yf)
    xf = rfftfreq(p.shape[0] - startStep, ts)

    # - Directivity plot ----------------------------------------------------------------------------------

    theta = np.arange(0, np.pi, np.pi / p.shape[1])
    #ax = plt.subplot(111, projection='polar')
    plt.figure(figsize=(3,5))

    for plt_freq in plt_freqList:
        index_plt = np.argmin(abs(xf - plt_freq))
        yf_plt = yf[:, index_plt] / max(yf[:, index_plt])
        plt.polar(theta, yf_plt)

    plt.title( Herz+ title)

    plt.savefig(plotname, bbox_inches='tight', transparent=False)

    #plt.show()






data_path = "/home/petrar/PetraMaster/WS23/CSE_Sem3/FEM-Multiphys/Aucoustic/mics_dpdt"  # Path to mics folder

files_Coupled = "micArrayResults_1m_acouPressureCoupled"
files__Rigid = "micArrayResults_1m_acouPressureRigid"

files_Noscat = "micArrayResults_1m_acouPressureNoScatterer"
files_Scat = "micArrayResults_1m_acouPressureScatterer"

freqlist1 = [100]; freqlist2 = [400]; freqlist3 = [700]; freqlist4 = [1000]
indices_list = [[1],[2],[3],[4]]

info = ["100 Hz ", "400 Hz ","700 Hz ", "1000 Hz "]


for (i, fi) in enumerate([ freqlist1, freqlist2, freqlist3, freqlist4  ]):
    print(fi)
    gen_polarplots(data_path, files_Scat, freqlist2, "Harmonic analysis, rigid scatterer", indices_list[i], info[i])  
    gen_polarplots(data_path, files_Noscat, freqlist2, "Harmonic analysis, no scatterer",indices_list[i], info[i])