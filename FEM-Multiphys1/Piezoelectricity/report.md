<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });</script>


## 1. Create a suitable mesh for the problem

* Create hexahedral mesh (2 elements per layer in thickness direction should be sufficient). **(1 Point)**.
* Use 2nd order elements (element type HEX20). **(0.5 Points)**.
* Define the necessary regions, assign meaningful names and show the mesh.  **(0.5 Points)** 

To reproduce the geomertry, just take the file [homework2.jou](homework2.jou)
The mesh of the structure looks as following:<br><br>
<img src="images1/mesh_upView.png" alt="Image 1" style="width:450px; height:290px; text-align:center" />
Mesh of the PMMA Plate (left) and piezo patches (roght, only one is displayed, since they are meshed the same way)
<img src="images1/meshPlate.png" alt="Image 1" style="width:350px; height:200px;" /> <img src="images1/meshPatch.png" alt="Image 1" style="width:350px; height:200px;" />





### Modelling assumptions

What are the modelling assumptions in the conducted simulation? Consider PDEs, boundary conditions, material models, and analysis type, ... (**2 Points**)

**1. PDEs**

  - We solve mechanic- electric PDEs. We have a mechanical and an electrical PDE
  - Electrical: $ \nabla \times E  = 0 \Longleftrightarrow E = - \nabla \Phi $ and $\nabla \cdot D = 0$
  - Mechanical: $ \rho \cdot \frac{\partial² u}{\partial t²} = \nabla \cdot \sigma + g  $
- The two equations are coupled via **volume coupling** by inserting the following terms for $\sigma , D$:
  - $ \sigma = C^E : s - e \cdot E $ (mechanical field couples on E-field)
  - $ D = e : s + \epsilon^S \cdot E $ (electrical field couples on mechanical strains)
- We instroduce Ansatz functions for both the unknowns to solve:
  - For the mechanical displacement $u$:    $u = \sum_i x_i u^*_i $ with unknown coefficients $x_i$ and known shape functions $u^*_i$ (note that the displacement is a vector quantity)
  - For the electrostatic potential $\Phi$:  $\Phi = \sum_i \phi_i u^*_i $ with unknown coefficients $\phi_i$ and known shape functions $u^*_i$
- We can apply Galerkin procedure and that leads to a coupled system of ordinary differential equations.
  
**2. Modelling assumptions**

1. Thickness of electrodes very small compared to dimensions of the plate/patches $\Longrightarrow$ electrodes are  mechanically not significant.
2. Electrodes are perfect conductors: conducttance $\infty \Longrightarrow$ constant electric potential at electrodes.
3. Linear piezoelectricity we neglect hyserisis of the the polarisation
4. Perfect, homogeneous polarization.
5. Small strains $\Longrightarrow$ linearized strain-displacement relation

**3. Boundary conditions**
   1. Constant electric potential at electrodes, thereby at the top and bottom of the patches (Electrodes are perfect conductors)
   2. The "ground" $\Phi =0$ is at the bottom of the piezo patches: we have the freedom to set the 0-point of the electric potential ((homogenous Dirichlet BC))
   3. The plate is fixed on the sides $\Longrightarrow$ zero mechanical displacement of the plate at the slides $\Gamma_s$ (homogenous Dirichlet BC)

**4. Analysis types**
We can perform static (steady -state), eigenfrequency, harmonic and transient analysis