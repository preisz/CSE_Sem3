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

def gen_polarplots(data_path, filename):

    ts = 1e-3  # Sample time in s
    plt_freqList = [95,190]  # Frequency to plot in Hz
    plotname = filename + ".png"

    flag_savePlots = True

    # - Read Mic data ----------------------------------------------------------------------------------

    coord = []
    p = []
    for i in range(1, 21):
        #coord.append(np.loadtxt(f'{data_path}/micArrayResults_1m_acouPressure-1', usecols=[1,2], delimiter=',', dtype=float, skiprows=1))
        #p.append(np.loadtxt(f'{data_path}/micArrayResults_1m_acouPressure-1', usecols=[3], delimiter=',', dtype=float, skiprows=1))
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

        index_plt = np.argmin(abs(xf - plt_freq))
        yf_plt = yf[:, index_plt] / max(yf[:, index_plt])

        ax.plot(theta, yf_plt, label="Size 1m")

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=10.0)

    legendEntries = []
    for plt_freq in plt_freqList:
        legendEntries.append(f'{plt_freq} Hz')

    plt.legend(legendEntries, loc=3)

    if flag_savePlots:
        plt.savefig(plotname, bbox_inches='tight', transparent=False)

    plt.show()


data_path = "/home/petrar/PetraMaster/WS23/CSE_Sem3/FEM-Multiphys/Aucoustic/mics_dpdt"  # Path to mics folder
files_Coupled = "micArrayResults_1m_acouPressureCoupled"
files__Rigid = "micArrayResults_1m_acouPressureCoupled"

gen_polarplots(data_path, files_Coupled)
gen_polarplots(data_path, files__Rigid)

Nsteps = 319; deltaT = 1/(624 * 15)
data_path = "/home/petrar/PetraMaster/WS23/CSE_Sem3/FEM-Multiphys/Aucoustic/history"
name1 = "mic1-soundhard-trans"; name2="mic2-soundhard-trans"
name1t = "mic1-flexiblescat_trans"; name2t="mic2-flexiblescat_trans"; name3t = "mic3-flexiblescat_trans"

p1 = []; p2 = []; t = []
p1tr = []; p2tr =[]; p3tr = []

for i in range(1, Nsteps + 1):
    #realPress = np.loadtxt(file_path, usecols=[2], delimiter='\t', dtype=float, skiprows=1)
    p1.append(np.loadtxt(f"{data_path}/{name1}-{i}", usecols=[3], delimiter=',', dtype=float, skiprows=1))
    p2.append(np.loadtxt(f"{data_path}/{name2}-{i}", usecols=[3], delimiter=',', dtype=float, skiprows=1))
    
    p1tr.append(np.loadtxt(f"{data_path}/{name1t}-{i}", usecols=[3], delimiter=',', dtype=float, skiprows=1))
    p2tr.append(np.loadtxt(f"{data_path}/{name2t}-{i}", usecols=[3], delimiter=',', dtype=float, skiprows=1))
    displ = np.loadtxt(f"{data_path}/{name3t}-{i}", usecols=range(3, 5), delimiter=',', dtype=float, skiprows=1)
    p3tr.append( np.linalg.norm(displ) )   #mech displ
    
    t.append(i *deltaT)

p1 = np.array(p1); p2 = np.array(p2); 

plt.plot(t, p1, label="Mic1 rigid"); plt.plot(t, p2, label="Mic2 rigid ");
plt.plot(t, p1tr, label="Mic1 coupled"); plt.plot(t, p2tr, label="Mic2 coupled");plt.legend(); 
plt.grid(); plt.xlabel("Time $t$"); plt.ylabel("AcouPressure [Pascal]")
plt.title("Evolution of acoustic pressure"); plt.show()

plt.plot(t, p1tr, label="Mic1"); plt.plot(t, p2tr, label="Mic2");plt.legend(); 
plt.grid(); plt.xlabel("Time $t$"); plt.ylabel("AcouPressure [Pascal]")
plt.title("Flexible scatterer"); plt.show()
    
plt.plot(t, p1, label="Mic1"); plt.plot(t, p2, label="Mic2");plt.legend(); 
plt.grid(); plt.xlabel("Time $t$"); plt.ylabel("AcouPressure [Pascal]")
plt.title("Rigid scatterer"); plt.show()

plt.plot(t, p1tr, label="Mic1"); plt.plot(t, p2tr, label="Mic2");plt.legend(); 
plt.grid(); plt.xlabel("Time $t$"); plt.ylabel("AcouPressure [Pascal]")
plt.title("Flexible scatterer"); plt.show()

plt.plot(t, p3tr, label="Mic3") ;plt.legend(); 
plt.grid(); plt.xlabel("Time $t$"); plt.ylabel("MechDispl [meters]")
plt.title("Flexible scatterer"); plt.show()
