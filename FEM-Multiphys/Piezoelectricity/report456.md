# 4. Static analysis
_Test the model in static analysis by applying a pressure of magnitude p = 1000 N/m² on the back of the plate as shown in ﬁgure 2.
Show the deformed shape (appropriately scaled) and visualize the computed electric potential in the piezoelectric patches._

How applying fix BCs and pressure load can be found in the "simulation_input_template.xml" file at sequency step 1.

# 5. Eigenvalue analysis

_Compute the ﬁrst 10 eigenmodes of the PMMA plate with piezoelectric patches for open electrode conﬁguration. Visualize the eigenmodes and provide the corresponding natural frequency_

The computed eigenfrequencies can be found in the file "results2.info.xml" and they are (in Herz):

    mode nr="1" frequency="101.530173725666" errorbound="1.3247e-10"/>
    mode nr="2" frequency="145.350161122659" errorbound="1.1625e-10"/>
    mode nr="3" frequency="227.170466703951" errorbound="6.3503e-11"/>
    mode nr="4" frequency="253.640631452802" errorbound="2.7788e-11"/>
    mode nr="5" frequency="296.35625271638" errorbound="7.3648e-11"/>
    mode nr="6" frequency="333.819525748314" errorbound="2.0888e-11"/>
    mode nr="7" frequency="367.619631591616" errorbound="7.4705e-11"/>
    mode nr="8" frequency="470.179361309502" errorbound="4.0783e-10"/>
    mode nr="9" frequency="478.520899595335" errorbound="3.5021e-09"/>
    <mode nr="10" frequency="486.621404171708" errorbound="8.1299e-09"/>
 