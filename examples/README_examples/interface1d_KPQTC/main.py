import os ; import sys
sys.path.append(os.getcwd()+"/../../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 12 # exponential length, leads to 2**L sites
H = hamiltonians.chain(L) # get the Hamiltonian

def f(r):
    """Modulation of the hopping"""
    omega1 = np.pi*2.*np.sqrt(2.)/(2**L/10) # left frequency
    omega2 = np.sqrt(3)*omega1 # right frequency
    if r[0]<0.: return 0.2*np.sin(omega1*r[0]) # return correction to the hopping
    else: return 0.1*np.sin(omega2*r[0]) # return correction to the hopping
H.modify_hopping(f) # modify the hopping 

SCF = H.get_SCF_Hubbard(U=3.0) # generate a selfconsistent object

SCF.solve(use_qtci=True,use_kpm=True,info=True,info_qtci=True,delta=1e-1,
        maxite=10,use_dynamical_qtci=False,chiral_AF=True)
Mz = SCF.Mz # selfconsistent magnetization


import matplotlib.pyplot as plt

fig = plt.figure(figsize=[6,3])

plt.subplot(1,2,2)
plt.plot(H.R[:,0],np.abs(Mz),c="blue")
plt.title("Selfconsistent solution")
plt.xlabel("position") ; plt.ylabel("|Magnetization|")
xlim = np.max(H.R[:,0])/2.
plt.xlim([-xlim,xlim])


plt.subplot(1,2,1)
s = H.get_moire() ; s = s - np.mean(s) 
plt.plot(H.R[:,0],s,c="red")
plt.title("Moire Hamiltonian")
plt.xlabel("position") ; plt.ylabel("Strain")
xlim = np.max(H.R[:,0])/2.
plt.xlim([-xlim,xlim])
plt.ylim([-1,1])

plt.tight_layout()

plt.savefig("SCF.png",dpi=500)
plt.show()


