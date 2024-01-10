#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:24:11 2020

@author: INTERN+feberhar
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal.windows
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch

file1 = r"../3_ca/history/propagation_dpdt-acouPotentialD1-node-7730-mic1.hist"
file2 = r"../3_ca/history/propagation_dpdt-acouPotentialD1-node-131-mic2.hist"

rho = 1.204
# c = 343.5

p0 = 20e-6  # reference pressure in Pa

flag_savePlots = True

# - Read File 1 ----------------------------------------------------------------------------------

fobj = open(file1, "r")
cont = fobj.read()
fobj.close()
txt = cont.split("\n")
t1 = []
y1 = []

for i in range(len(txt) - 4):
    line = txt[i + 3].split("  ")
    t1.append(float(line[0]))
    yvalue = float(line[1])
    yvalue = yvalue * rho
    y1.append(yvalue)

# - Read File 2 -----------------------------------------------------------------------------------

fobj = open(file2, "r")
cont = fobj.read()
fobj.close()
txt = cont.split("\n")
t2 = []
y2 = []

for i in range(len(txt) - 4):
    line = txt[i + 3].split("  ")
    t2.append(float(line[0]))
    yvalue = float(line[1])
    yvalue = yvalue * rho
    y2.append(yvalue)

# - Plot time series ----------------------------------------------------------------------------------

plt.plot(t1, y1)
plt.plot(t2, y2)
plt.grid()
# plt.title("mic1")
plt.ylabel("$p^\mathrm{a}$ in Pa")
plt.xlabel("time in s")
plt.legend(('Mic 1', 'Mic 2'))
plt.tight_layout()
if flag_savePlots:
    picname = "mic_data_dpdt.png"
    plt.savefig(picname, dpi=100)

plt.show()

# - Cut data ----------------------------------------------------------------------------------
startStep = 0

y1 = np.array(y1)
y2 = np.array(y2)

y1 = y1[startStep:]
y2 = y2[startStep:]


# - Calculate power spectral density ----------------------------------------------------------------------------------

fs1 = 1.0 / (t1[1] - t1[0])
fs2 = 1.0 / (t2[1] - t2[0])

nblock = 128
overlap = 32
# win = hanning(nblock, True)
win = scipy.signal.windows.hann(nblock, True)

f1, Pxxf1 = welch(y1, fs1, window=win, noverlap=overlap, nfft=nblock, return_onesided=True, detrend=False)
f2, Pxxf2 = welch(y2, fs2, window=win, noverlap=overlap, nfft=nblock, return_onesided=True, detrend=False)

# - Plot SPL ----------------------------------------------------------------------------------

plt.semilogy(f1, np.sqrt(Pxxf1), '-')
plt.semilogy(f2, np.sqrt(Pxxf2), '-')

plt.grid()
plt.xlim((0,500))
plt.ylabel("Amplitude spectral density in Pa/Hz")
plt.xlabel("frequency in Hz")
plt.legend(('Mic 1', 'Mic 2'))
plt.tight_layout()

plt.show()

plt.plot(f1, 20 * np.log10(np.sqrt(Pxxf1) / np.sqrt(2) / p0) , '-')
plt.plot(f2, 20 * np.log10(np.sqrt(Pxxf2) / np.sqrt(2) / p0) , '-')

plt.grid()
plt.xlim((0,500))
plt.ylim([-60, 60])
plt.ylabel("Sound pressure level in dB")
plt.xlabel("frequency in Hz")
plt.legend(('Mic 1', 'Mic 2'))
plt.tight_layout()

if flag_savePlots:
    picname = "mic_SPL_dpdt.png"
    plt.savefig(picname, dpi=100)

plt.show()

# - FFT -----------------
ts1 = (t1[1] - t1[0])
ts2 = (t2[1] - t2[0])

win = scipy.signal.windows.hann(len(y1), True)

yf1 = abs(rfft(win*y1))
xf1 = rfftfreq(len(y1), ts1)

yf2 = abs(rfft(win*y2))
xf2 = rfftfreq(len(y2), ts2)

Yf1 = 20 * np.log10(yf1 / p0)
Yf2 = 20 * np.log10(yf2 / p0)

plt.plot(xf1, Yf1)
plt.plot(xf2, Yf2)

plt.grid()
plt.xlim((0,500))
plt.ylabel("F{$p^\mathrm{a}$} in dB")
plt.xlabel("frequency in Hz")
plt.legend(('Mic 1', 'Mic 2'))
plt.tight_layout()

plt.show()
