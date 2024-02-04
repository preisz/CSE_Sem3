We have little or no information on different types of BC input.
For example, it is not precisely clear from the userdocu how `<grid>`  works.

```xml
<displacement name="V_plate">
  <grid dofs="all">
    <defaultGrid quantity="mechDisplacement" dependtype="GENERAL">
      <globalFactor>1</globalFactor>
    </defaultGrid>
  </grid>
</displacement>
``````
There is a missing description for many others ( `<fileData>`` and so on).
We propose to create a subsection with basic descriptions of all (or all most confusing) BC input methods.
![Alt text](image.png)
<br>

![Alt text](image-1.png)
<br>

![Alt text](image-2.png)
<br>

![Alt text](image-3.png)
<br>

### The file is here [link](https://gitlab.com/openCFS/userdocu/-/blob/fileDataAndGridInput/docs/Tutorials/Features/bcsAndLoadsFeatures.md)


# Boundary Condition Input

A number of different methods of applying boundary conditions are provided.
Besides specifying constant scalar, vector, or tensor values as `value="..." [phase=""]`,
you can prescribe field values from other PDEs, previous simulations, or other external sources.
Just use the corresponding XML tags within the boundary condition tag.

## fileData

`<fileData>` can be used to define the right-hand side vector nodal quantities. For this purpose, one needs to provide a dat-file with node numbers and corresponding RHS values.

Only `<force>` in MechPDE was tested for this feature. Dirichlet BC in MechPDE was proven not to work correctly with it.
For further information, please refer to the test case.

```           
<force name="inclusion_nodes">
  <fileData file="nodal.dat"/>
</force>
```

## grid

The `<grid>` input allows specifying inhomogeneous Dirichlet BC. 
One may use the output results of openCFS (`.cfs`) as input for another model.

Another possible application would be to modify the openCFS result file (with python script) and specify needed BC manually for each node. For this purpose, a "fake" results file should be generated, modified, and finally applied as a BC to the matching model using the following code structures.

```
<displacement name="V_plate">
  <grid dofs="all">
    <defaultGrid quantity="mechDisplacement" dependtype="GENERAL">
      <globalFactor>1</globalFactor>
    </defaultGrid>
  </grid>
</displacement>
```

The grid input can be used e.g. as `<displacement>` BC in SmoothPDE and MechPDE.