# Problem Description
This homework investigates acoustic wave propagation within an open domain featuring a solid scatterer. Through the designated tasks, acoustic scattering is studied in the time and frequency domain. One can observe the influence of
both stiff and flexible solids on the wave propagation.

The problem’s geometry is illustrated in the accompanying sketch, with the solid
scatterer situated in the middle of the acoustic domain. The excitation boundary
condition is characterized by a uniform normal velocity with an amplitude of 1 at $\Gamma_{e}$. Absorbing boundary condition (ABC) is assumed to simulate free field condition at $\Gamma_{ABC}$.

The interested frequency range of the study is **100 to 1000 Hz**. The density and compression modulus of air are **1.225 kg/m^3** and **1.4271 · 10E5 Pa**. The flexible
body is constructed from foam with density, Young modulus and Poisson number of **320 kg/m3**, **35.3 · 10E6 Pa** and **0.383**, respectively.

# 1. Create a suitable mesh for the problem
• _define the necessary regions and assign meaningful names._
• _estimate the necessary spatial discretization. (1 Point)_
• _create an image of your mesh and describe how you choose the discretization (1 Point)_

The wave equation that describes the propagation of acoustic waves, has smooth solution. For this reason, I used higher order elements (quadratic _QUAD_). One should to try avoidung mesh distortions, for this reason I used laplacian smoothing.  <br>

The required discretization is frequency dependent. The meshize $d$ is determined by the highest excitation frequency $f_{max}$ according to 

$$ d \approx \frac{1}{k} \lambda \, , \, \, k \approx 10-20 \\ .\\
\lambda = c T = \frac{c}{f_{max}}$$
the speed of sound in air is given as

$$ c = \sqrt{\frac{K}{\rho}} =  \sqrt{\frac{K}{\rho}} = 341.31797 m/s$$

With $K$ being the compression modulus and $\rho$ the densiy of air. Since we consider harmonic analysis between 100 and 1000Hz, $f_{max}= 1000 Hz$ and for $k$ I took $k=10$:

$$ d = \frac{1}{k} \cdot \frac{c}{f_{max}} \approx 0.034m$$.

# 2. Setup harmonic analysis
_Perform harmonic simulation_
_1. without the solid scatterer, i.e., radiation into the free field;_
_2. with rigid scatterer_

The simulation setup without the rigid scatterer can be found in [simulation-no-scaterrer.xml](simulation-no-scaterrer.xml). When having the rigid scatterer, one can still only go for the solution of the acoustic PDE, with sound hard boundary conditions on the boundary between the air and scatterer domain. This corresponds to the homogenous (natural) Neumann boundary condition.

#### Compare the results by answering the following questions.
- _plot the pressure field at 100, 400, 700 and 1000 Hz with and without scattering. (1 Point)_
On the graphics below, the magnitudde of the acousic pressure is illustrated.

- _at the mentioned frequencies, create polar plots of the acoustic pressure amplitude at a radius of 1 m around the excitation_

