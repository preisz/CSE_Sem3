# HeatMech | Drilling Hole
## Problem Description

Investigate the part under different drilling conditions. The problem is depicted below:

![](./pics/image1.png){width=40%}

The geometry of the model is the quadratic part given below:

![](./pics/part_reworked.png){width=40%}

The whole part is made of steel (S235JR). A through bore is drilled in the middle of the part. Due to drilling, heat power $P$ is transferred into the part through the surface of the hole ($S_{hole}$). It is assumed that the heat power homogeneously contributes over the hole surface. During the drilling, the part is clamped on two opposite sides ($S_{fixed}$). The clamping must be modelled using zero displacement boundary conditions (fixed). The fixation is initially stress-free at ambient air temperature $T_r$. The part gets cooled by water, with the temperature $T_w$. The water runs over the top and unclamped side surfaces ($S_{cooled}$). On the bottom ($S_{air}$), it's surrounded by an airstream with an ambient temperature of $T_r$. Additionally, we neglect all heat transfer through the clamped sides and assume that the drilling speed is low compared to the heat propagation.

Create an appropriate FE model and investigate the following questions:

* What is the maximal temperature under normal operation?
* What is the maximal temperature if the water cooling fails?
* What are the maximum occurring thermal deformations in the above two cases?
* Could temperature-induced stresses pose a problem?

Use the given values:

|Variable|Value|
|--|:--:|
|drill hole radius $R$ in mm|5|
|Height $H$ in mm|30|
|Lenght $L$ in mm|100|
|Produced heat power in W|200|
|Heat transfer coefficient steel-air $\alpha_{air}$ in W/(m$^2$·K)|20|
|Heat transfer coefficient steel-water $\alpha_{water}$ in W/(m$^2$·K)|600|
|Density of S235JR $\rho$ in kg/m³|7850|
|E-modulus of S235JR in N/m²|2.00E+11|
|Thermal expansion of S235JR in 1/K |12E-6|
|Poisson number of S235JR |0.29|
|Heat capacity of S235JR in J/K|461|
|Heat conductivity of S235JR in W/mK |50|
|Ambient temperatue of air $T_r$ in °C|20|
|Temperature of water $T_w$ in °C |20|

To complete the homework assignment, follow the tasks and questions below.

## Assignment

### 1. Create a suitable mesh for the problem **(4 Points)**
* Create a hexahedral mesh.
* Define the necessary regions and assign meaningful names.
* Create the .cdb file in cubit and convert it to .cfs fromat by ‘cfs -g geometry‘ command (geometry.xml file is provided with the assignment). **(2.0 Points)** 
* Load the converted file in Paraview and create an image showing the mesh size and the different regions. **(1.5 Points)** 
* How many nodes and elements does your mesh have? **(0.5 Points)**

### 2. Modelling Assumptions **(6 Points)**

2.1. What are the modelling assumptions in the thermal model? Consider the material, analysis type, pde, etc. State at least 4 assumptions and justify them briefly. (**3.0 Points**)
-Temperature field is independent of the mechanical fiels $\Longrightarrow$ forward coupling
-Perfect thermal isolation on the surfaces $S_{fixed} \Longrightarrow$ we can apply Neumann boundary conditions
-Fourier law holds $\Longrightarrow$ no radiation, no convection
-We have conservation of energy, described by the equation (strong form of PDE): 
  $$  \rho c_m \frac{ \partial T(x,t) }{\partial t} - \nabla \cdot ( k \nabla T )= \dot{q}(x,t) $$
-Time-indipendent and isotropic material properties


2.2. What are the modelling assumptions in the mechanical model and for the coupling? Consider the material, analysis type, pde, etc. State at least 4 assumptions and justify them briefly.
(**3.0 Points**)
-Only small displacements
-Drilling affects only the temperature by heat production, but causes no additional stresses and strains
-Due to small displacements, we can apply Hook's law $$ \sigma = C : s $$
-Linear elastic material behavior
-We make static analysis, since drilling is a lot smaller than time needed to reach the equilibrium temperature
-We have the stong form of the PDE: 
$$ \rho \frac{\partial ² \bold{u}}{\partial t²} - \nabla \cdot \sigma = \bold{g}  $$
$\bold{u}$ denotes the displacement (vector quantity) and $\bold{g}$ denotes the volume force density
-We can neglect traction between material and air/water
-We can neglect the affect of deformation field on the temperature field $\Longrightarrow$ only forward coupling

### 3. Drilling at normal operation **(11 Points)**

3.1. Setup an appropriate simulation input for CFS. You need to:

  - define the domain and assign a material,
  - specify an appropriate PDE and analysis type,
  - define the required boundary conditions,
  - specify postprocessing results.

Describe briefly chosen PDEs, analysis types, BCs and coupiling. (**2.0 Points**)

3.2. Give the unit of the prescribed heat flux density $\dot{q}_s$ **(0.5 Points)**

3.3. Compute the heat flux density, which has to be prescribed at the source surface **(0.5 Points)**

3.4. Visualize the temperature distribution in paraview (with only 10 colors in the palette) **(1.0 Point)**

3.5. Visualize the vector field of the heat flux density with a glyph-plot in the z-x plane (y=0) (**1.5 Points**)

3.6. Why is an output of heat flux density defined on elements and not at nodes? Can we compute a heat flux at a node on the reference element? **(1.5 Points)**

3.7. What do you notice when looking at the flux vectors? To be more specific, what does the direction of the vectors tell us? **(1.0 Point)**

3.8. What is the highest temperature? And where does it occur? Is this temperature safe for the structure in terms of material transformations? **(1.5 Points)**

3.9. How much heat is transfered due to the water cooling and how much due to the air? How much is this in percentage? **(1.5 Points)**

### 4. Drilling without water cooling **(3 Points)**

 Hint: if water does not cover the part, something will still cover it.

4.1. Which boundary condition do you need to change? What kind of BC do you set now? **(0.5 Points)**

4.2. Visualize the temperature distribution in paraview (no more than 10 colours) **(1.0 Point)**

4.3. What is the highest temperature? And where does it occur? Is it safe in terms of material transformations? **(1.0 Point)**

4.4. How much heat is now transfered due to the air? How much is this in percentage? **(0.5 Points)**

### 5. Heat-Mechanic Coupling **(6 Points)**
Determine the effect of the temperature change on the mechanical behaviour of the structure.
Use the case with the water cooling from above.
Assume the structure is free of internal stress in the given initial configuration at a room temperature of 293K.

Generate a suitable simulation input for the coupled simulation to determine the thermal deformations and stresses.

5.1. How shall the (mechanic) boundary condition be chosen in order to fit the description above? **(1.5 Points)**

5.2. Visualize the deformed structure (mind the number of colours) **(1.5 Points)**

5.3. What is the maximum deflection, and where does it occur? **(0.5 Points)**

5.4. Visualize the von-Mises stress **(1.0 Point)**

5.5. What is the maximum von-Mises stress and where does it occur? Can it be a problem for the structure? **(1.5 Points)**
