import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 8 # exponential length
H = hamiltonians.chain(L) # get the Hamiltonian

import time
ii = 2**(L-1) # bulk
jj = 0 # edge

# perform the SCF
SCF = H.get_SCF_Hubbard(U=3.0,U_profile=None)
SCF.solve(use_qtci=False,use_kpm = False)


import matplotlib.pyplot as plt

(es,ds) = SCF.get_dos(i=[ii],kpm_scale=4,delta=1e-1)
plt.plot(es,ds,label="Bulk")
(es,ds) = SCF.get_dos(i=[jj],kpm_scale=4,delta=1e-1)
plt.plot(es,ds,label="Edge")

plt.legend()

plt.xlabel("Energy")
plt.ylabel("DOS")

plt.show()

