## 1. Create a suitable mesh for the problem

* Create hexahedral mesh (2 elements per layer in thickness direction should be sufficient). **(1 Point)**.
* Use 2nd order elements (element type HEX20). **(0.5 Points)**.
* Define the necessary regions, assign meaningful names and show the mesh.  **(0.5 Points)** 

To reproduce the geomertry, just take the file *hw2.jou*

### Modelling assumptions

What are the modelling assumptions in the conducted simulation? Consider PDEs, boundary conditions, material models, and analysis type, ... (**2 Points**)

- PDEs: We solve mechanic- electric PDEs. We have a mechanical and an electrical PDE
  - Electrical: $ \nabla \times E  = 0 \Longleftrightarrow E = - \nabla \Phi $ and $\nabla \cdot D = 0$
  - Mechanical: $ \rho \cdot \frac{\partial² u}{\partial t²} = \nabla \cdot \sigma + g  $
- The two equations are coupled via **volume coupling** by inserting the following terms for $\sigma , D$:
  - $ \sigma = C^E : s - e \cdot E $ (mechanical field couples on E-field)
  - $ D = e : s + \epsilon^S \cdot E $ (electrical field couples on mechanical strains)
- We instroduce Ansatz functions for both the unknowns to solve:
  - For the mechanical displacement $u$:    $u = \sum_i x_i u^*_i $ with unknown coefficients $x_i$ and known shape functions $u^*_i$ (note that the displacement is a vector quantity)
  - For the electrostatic potential $\Phi$:  $\Phi = \sum_i \phi_i u^*_i $ with unknown coefficients $\phi_i$ and known shape functions $u^*_i$
- We can apply Galerkin procedure and thaat leads to a coupled system of ordinary differential equations.
  

### 3. Material data

The material data of the piezoceramic PIC 255 are given below,

$S_{11} = 16.1\cdot 10^{-12}\frac{m^2}{N},$
$S_{12} = -4.7\cdot 10^{-12}\frac{m^2}{N}$, 
$S_{13} = -8.5\cdot 10^{-12}\frac{m^2}{N}$,
$S_{33} = 20.7\cdot 10^{-12}\frac{m^2}{N}$, 
$S_{44} = 42\cdot 10^{-12}\frac{m^2}{N}$, 

$d_{31}=-180\cdot 10^{-12}\frac{C}{N},$ 
$d_{33}= 400\cdot 10^{-12}\frac{C}{N},$ and
$d_{15}= 550\cdot 10^{-12}\frac{C}{N},$ 

$\epsilon_{11}= 1.4069\cdot 10^{-8}\frac{As}{Vm}$,
$\epsilon_{33}= 1.5494\cdot 10^{-8}\frac{As}{Vm}$,

Density $\rho = 7800 \frac{kg}{m^{3}}$


The piezoceramic PIC255 is polarized in the thickness (3) direction. The blocking force of the piezoelectric patch for an applied voltage of $500V$ is $F_{max} = 256N$ in the length direction and the strain in the width direction is $-650 \mu m/m$. The stress-free nominal displacement of the patch under the applied voltage is $\Delta L_{0} = -27 \mu m$ in the length direction. 
 
Calculate the significant effective material properties $d^{simp}_{31}$, $S^{simp}_{11}$, $S^{simp}_{12}$ and $\epsilon^{simp}_{33}$ for the simplified model of the patch. Calculate the percentage change in effective material properties from the PIC255 material data and apply the change to other unknown material properties using the hints below:
 
* Find $d^{simp}_{31}$ using the nominal displacement and the applied electric field $E_{3}$. Apply the change to $d^{simp}_{33}$ and $d^{simp}_{15}$ (Hint: Use the d-form of the constitutive relation). (**1 point**)

We can calculate as following:
- stress free nominal displacement $\Delta L_0 = -27 \mu m$ in the length direction. 
- So, the stress $\sigma = 0$ (since stress-free configuration)
- Let's take the d-form of the constitutive equation:
$$ s_1 = S^E_{1j}\sigma_j + d_{1j} E_j   $$
Here we took the vector notation of stresses and strains, where $s = (s_1, s_2, s_3, s_4, s_5, s_6 ) = (s_{11}, s_{22}, s_{33},s_{23},s_{13},s_{12}) $.
- We know that $s_1 = \Delta L_0 /l, \, \, \sigma_j =0 \forall j, \, \, E_1=E_2=0$ and $E_3=0$. Thereby, we can write:

$$ s_1 = \frac{\Delta L_0}{l_p} = d_{13}E_3 = d_{13} \cdot \frac{V}{t_p}$$

using symmetry $d_{31} = d_{13}$, 

$$ d^{simp}_{31}  = \frac{\Delta L_0/l_p}{V/t_p} = \frac{4.42\times10^{-4}}{1250 V/mm} = 3.536 \times 10^{-7}\frac{mm}{V} = 3.536 \times 10^{-7}\cdot \frac{ C }{ N\cdot m} \cdot 10^{-10} m$$

Using that one Volt is $1V = 1J/C$. It yields
$$ d^{simp}_{31} = 3.536 \times 10^{-10}\cdot \frac{ C }{ N}$$

This is a change of factor:

$$ k= \frac{d^{simp}_{31}}{d_{31}}= 
\frac{ 3.536 \times 10^{-10}C/N}{1.80\cdot 10^{-10}C/N} = 1.964$$

This results in

$d^{simp}_{31}=-180\cdot 10^{-12}\frac{C}{N}\cdot k = 3.536 \times 10^{-10}C/N,$ 
$d^{simp}_{33}= 400\cdot 10^{-12}\frac{C}{N} \cdot k = 785.77 \cdot 10^{-12}\frac{C}{N},$


 and
$d^{simp}_{15}= 550\cdot 10^{-12}\frac{C}{N} = 1080.44\cdot 10^{-12}\frac{C}{N}$ .