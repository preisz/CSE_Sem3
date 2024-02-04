import numpy as np
import matplotlib.pyplot as plt

data_path = "/home/petrar/PetraMaster/WS23/CSE_Sem3/FEM-Multiphys/Aucoustic/mics_dpdt"  # Path to mics folder


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
p1tr = np.array(p1tr); p2 = np.array(p2tr); p3tr = np.array(p3tr)

#Mic 1------------------------------------------------------------------------
plt.figure(dpi = 115)
plt.plot(t, p1, label="Mic1 rigid")
plt.plot(t, p1tr, label="Mic1 coupled")
plt.legend(); plt.grid(); plt.xlabel("Time $t$"); plt.ylabel("AcouPressure [Pascal]")
plt.title("Transient acoustic pressure, mic1")
plt.savefig("images/ptrans-mic1.png");plt.show()

# Plot for the first half
fig, ax = plt.subplots(2, 1, figsize=(6, 8))# Get the first half of the data
half_index = len(t) // 2
ax[0].plot(t[:half_index], p1[:half_index], label="Mic1 rigid")
ax[0].plot(t[:half_index], p1tr[:half_index], label="Mic1 coupled")
ax[0].grid()
ax[0].set_ylabel("Acoustic Pressure [Pascal]")
ax[0].legend()

# Plot for the second half
ax[1].plot(t[half_index:], p1[half_index:], label="Mic1 rigid")
ax[1].plot(t[half_index:], p1tr[half_index:], label="Mic1 coupled")
ax[1].grid()
ax[1].set_xlabel("Time $t$")
ax[1].set_ylabel("Acoustic Pressure [Pascal]")
ax[1].legend()
plt.savefig("images/ptrans-mic1-sublots.png");plt.show()


#mic2------------------------------------------------------------------------------
plt.figure(dpi = 115)
plt.plot(t, p2, label="Mic2 rigid ")
plt.plot(t, p2tr, label="Mic2 coupled")
plt.grid(); plt.xlabel("Time $t$"); plt.ylabel("AcouPressure [Pascal]")
plt.title("Transient acoustic pressure, mic2")
plt.savefig("images/ptrans-mic2.png");plt.show()

fig, ax = plt.subplots(2, 1, figsize=(6, 8))# Get the first half of the data
half_index = len(t) // 2
# Plot for the first half
ax[0].plot(t[:half_index], p2[:half_index], label="Mic2 coupled")
ax[0].plot(t[:half_index], p2tr[:half_index], label="Mic2 coupled")
ax[0].grid()
ax[0].set_ylabel("Acoustic Pressure [Pascal]")
ax[0].legend()

# Plot for the second half
ax[1].plot(t[half_index:], p2[half_index:], label="Mic2 rigid")
ax[1].plot(t[half_index:], p2tr[half_index:], label="Mic2 coupled")
ax[1].grid()
ax[1].set_xlabel("Time $t$")
ax[1].set_ylabel("Acoustic Pressure [Pascal]")
ax[1].legend()

plt.title("Transient acoustic pressure, mic2")
plt.savefig("images/ptrans-mic2-sublots.png");plt.show()

    
plt.plot(t, p3tr, label="Mic3") ;plt.legend(); 
plt.grid(); plt.xlabel("Time $t$"); plt.ylabel("MechDispl Amplitude [meters]")
plt.title("Transient diplacement of mic3")
plt.savefig("images/displtrans-mic3.png");plt.show()
