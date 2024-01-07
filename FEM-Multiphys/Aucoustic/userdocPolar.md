```xml

<?xml version="1.0" encoding="UTF-8"?>
<cfsSimulation xmlns="http://www.cfs++.org/simulation" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	xsi:schemaLocation="http://www.cfs++.org/simulation https://opencfs.gitlab.io/cfs/xml/CFS-Simulation/CFS.xsd">
	<documentation>
		<title>Sound propagation simulation</title>
		<authors>
			<author>Andreas Wurzinger</author>
		</authors>
		<date>2022-01-13</date>
		<keywords>
			<keyword>aeroacoustics</keyword>
			<keyword>nonmatching grids</keyword>
			<keyword>pml</keyword>
			<keyword>transient</keyword>
		</keywords>
		<references></references>
		<isVerified>yes</isVerified>
		<description>
			Propagation simulation solving the aeroacoustic wave equation for the acoustic scalar potential
		</description>
	</documentation>
	<fileFormats>
		<!-- Data Input -->
		<input>
			<hdf5 fileName="../2_sourceInterpolation/results_hdf5/source_acouRhsLoad_dpdt.cfs" />
			<hdf5 fileName="../0_mesh/PML.cfs" id="id2" />
			<hdf5 fileName="../0_mesh/propagationRegion.cfs" id="id3" />
		</input>
		<!-- Data Output -->
		<output>
			<hdf5 />
			<text id="txt" />
		</output>
		<!-- Material File -->
		<materialData file="air.xml" format="xml" />
	</fileFormats>

	<domain geometryType="plane">
		<!-- User-defined Variables -->
		<variableList>
			<!-- Material parameters for source term scaling -->
			<var name="rho" value="1.204" />
			<var name="c" value="343.5" />
			<!-- Parameter for spatial blending function -->
			<var name="sigma" value="0.6" />
			<!-- Parameter for temporal blending function -->
			<var name="tau" value="20e-3" />
		</variableList>
		<!-- Volume Regions -->
		<regionList>
			<region name="internal" material="air_20deg" />
			<region name="prop" material="air_20deg" />
			<region name="pml" material="air_20deg" />
		</regionList>
		<!-- Surface Regions -->
		<surfRegionList>
			<!-- <surfRegion name="BC_IFin" /> -->
			<!-- <surfRegion name="BC_IFout" /> -->
			<!-- <surfRegion name="bc" /> -->
			<!-- <surfRegion name="ifborder" /> -->
		</surfRegionList>
		<!-- Non-conforming Interfaces -->
		<ncInterfaceList>
			<ncInterface name="NC1" masterSide="bc" slaveSide="BC_IFin" />
			<ncInterface name="NC2" masterSide="BC_IFout" slaveSide="ifborder" />
		</ncInterfaceList>
		<!-- User-defined Points -->
		<elemList>
			<elems name="mic1e">
				<coord x="1" y="0" />
			</elems>
			<elems name="mic2e">
				<coord x="0" y="3" />
			</elems>
		</elemList>
		<nodeList>
			<nodes name="mic1">
				<coord x="1" y="0" />
			</nodes>
			<nodes name="mic2">
				<coord x="0" y="3" />
			</nodes>
		</nodeList>
	</domain>

	<sequenceStep>
		<!-- Analysis Type -->
		<analysis>
			<transient>
				<numSteps>201</numSteps>
				<deltaT>1e-3</deltaT>
			</transient>
		</analysis>

		<!-- PDE Definition -->
		<pdeList>
			<acoustic formulation="acouPotential">
				<!-- Computation Domain -->
				<regionList>
					<region name="internal" />
					<region name="prop" />
					<region name="pml" dampingId="myPml" />
				</regionList>

				<!-- Non-conforming Interface Definition -->
				<ncInterfaceList>
					<ncInterface name="NC1" formulation="Nitsche" />
					<ncInterface name="NC2" formulation="Nitsche" />
				</ncInterfaceList>

				<!-- Damping Model -->
				<dampingList>
					<pml id="myPml">
						<type>inverseDist</type>
						<dampFactor>1</dampFactor>
					</pml>
				</dampingList>

				<!-- Boundary Conditions and Volume Loads -->
				<bcsAndLoads>
					<rhsValues name="internal">
						<grid>
							<defaultGrid quantity="acouRhsLoad" dependtype="GENERAL">
								<globalFactor>
									<!-- Scaling Factor from PDE --> <!-- Temporal blending fadeIn(duration, mode, timeval) --> <!-- Spatial blending -->
									(-1.0 / (rho*c^2)) * fadeIn(tau, 1, t) * ((sqrt(x^2+y^2) lt 1.9)? exp(-0.5*(x^2+y^2)/sigma^2) : 0)
								</globalFactor>
							</defaultGrid>
						</grid>
					</rhsValues>
				</bcsAndLoads>

				<!-- Results Definition -->
				<storeResults>
					<!-- Export point array -->
					<sensorArray fileName="mics_dpdt/micArrayResults_2m_acouPotentialD1" csv="yes" type="acouPotentialD1">
						<coordinateFile fileName="MicPos.csv" xCoordColumn="1" yCoordColumn="2" delimiter="," />
					</sensorArray>
					<sensorArray fileName="mics_dpdt/micArrayResults_2m_acouIntensity" csv="yes" type="acouIntensity" delimiter=",">
						<coordinateFile fileName="MicPos.csv" xCoordColumn="1" yCoordColumn="2" delimiter="," />
					</sensorArray>
					<!-- Export node result -->
					<nodeResult type="acouPotentialD1">
						<allRegions />
						<nodeList>
							<nodes name="mic1" />
							<nodes name="mic2" />
						</nodeList>
					</nodeResult>
					<!-- Export element result -->
					<elemResult type="acouIntensity">
						<allRegions />
						<elemList>
							<elems name="mic1e" />
							<elems name="mic2e" />
						</elemList>
					</elemResult>
				</storeResults>
			</acoustic>
		</pdeList>
	</sequenceStep>
</cfsSimulation>

```

```python
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


data_path = "../3_ca/mics_dpdt"  # Path to mics folder

ts = 1e-3  # Sample time in s
plt_freqList = [95,190]  # Frequency to plot in Hz

flag_savePlots = True

# - Read Mic data ----------------------------------------------------------------------------------

coord = []
p = []
for i in range(1, 202):
    coord.append(np.loadtxt(f'{data_path}/micArrayResults_2m_acouPotentialD1-{i}', usecols=[1,2], delimiter=',', dtype=float, skiprows=1))
    p.append(np.loadtxt(f'{data_path}/micArrayResults_2m_acouPotentialD1-{i}', usecols=[3], delimiter=',', dtype=float, skiprows=1))

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

theta = np.arange(0, 2 * np.pi, 2 * np.pi / p.shape[1])
ax = plt.subplot(111, projection='polar')

for plt_freq in plt_freqList:

    index_plt = np.argmin(abs(xf - plt_freq))
    yf_plt = yf[:, index_plt] / max(yf[:, index_plt])

    ax.plot(theta, yf_plt, label="Size 2m")

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=10.0)

legendEntries = []
for plt_freq in plt_freqList:
    legendEntries.append(f'{plt_freq} Hz')

plt.legend(legendEntries, loc=3)

if flag_savePlots:
    plt.savefig("mic_dpdt.pdf", bbox_inches='tight', transparent=True)

plt.show()

```
