# 4. Static analysis
_Test the model in static analysis by applying a pressure of magnitude p = 1000 N/m² on the back of the plate as shown in ﬁgure 2.
Show the deformed shape (appropriately scaled) and visualize the computed electric potential in the piezoelectric patches._

How applying fix BCs and pressure load can be found in the "simulation_input_template.xml" file at sequency step 1.

# 5. Eigenvalue analysis

_Compute the ﬁrst 10 eigenmodes of the PMMA plate with piezoelectric patches for open electrode conﬁguration. Visualize the eigenmodes and provide the corresponding natural frequency_

The computed eigenfrequencies can be found in the file "results2.info.xml" and they are:

     Frequency in Hz |           Errorbound
----------------------------------------
             101.359 |          2.78408e-09
             145.105 |          1.16825e-09
             223.335 |           9.7656e-10
             253.611 |          1.72425e-10
              296.15 |          3.76079e-10
             329.469 |          2.50672e-10
             366.416 |          3.16655e-10
             467.077 |          6.67623e-10
             469.653 |          1.62797e-08
             486.036 |          2.27193e-09

