#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal.windows
from matplotlib import rc

# rc('text', usetex=True)
from scipy.fft import rfft, rfftfreq

plt.rc('font', family='serif')
rc('font', size=16.0)
rc('lines', linewidth=1.5)
rc('legend', fontsize='medium', numpoints=1)  # framealpha=1.0,
rc('svg', fonttype='none')

# - User Input ----------------------------------------------------------------------------------

def gen_polarplots(data_path, filename, plt_freqList, title):
    """ create a polar plot of the acoustic pressure at
    the radius of 1 m around the excitation and compare it to sound hard BC
    (at characteristic frequencies)"""

    ts = 1e-3  # Sample time in s
    plotname = filename + ".png"

    flag_savePlots = True

    # - Read Mic data ----------------------------------------------------------------------------------

    coord = []
    p = []
    #for i in range(1, 21):
    for i in range(1, len(plt_freqList) + 1):
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
    ax = plt.subplot(111, projection='polar')

    for plt_freq in plt_freqList:
        #if (plt_freq == 100 or plt_freq ==1000):
            #continue

        index_plt = np.argmin(abs(xf - plt_freq))
        yf_plt = yf[:, index_plt] / max(yf[:, index_plt])

        ax.plot(theta, yf_plt, label="Size 1m")

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=10.0)

    legendEntries = []
    for plt_freq in plt_freqList:
        #if (plt_freq == 100 or plt_freq ==1000):
            #continue
        legendEntries.append(f'{plt_freq} Hz')

    plt.legend(legendEntries, loc=3)
    plt.title(title)

    if flag_savePlots:
        plt.savefig(plotname, bbox_inches='tight', transparent=False)

    plt.show()


data_path = "/home/petrar/PetraMaster/WS23/CSE_Sem3/FEM-Multiphys/Aucoustic/mics_dpdt"  # Path to mics folder
files_Coupled = "micArrayResults_1m_acouPressureCoupled"
files__Rigid = "micArrayResults_1m_acouPressureCoupled"

files_Noscat = "micArrayResults_1m_acouPressureNoScatterer"
files_Scat = "micArrayResults_1m_acouPressureScatterer"
freqlist1 = [100, 400, 700, 1000]; title1 = "Harmonic analysis, no scatterer"; title2 = "Harmonic analysis, rigid scatterer"
#gen_polarplots(data_path, files_Coupled)
#gen_polarplots(data_path, files__Rigid)

gen_polarplots(data_path, files_Noscat, freqlist1, title1)
gen_polarplots(data_path, files_Scat, freqlist1, title2)

