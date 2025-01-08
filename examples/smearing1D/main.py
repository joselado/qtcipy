import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 8 # exponential length
H = hamiltonians.chain(L) # get the Hamiltonian

import time
#(es,ds) = H.get_dos_i(i=2**(L-1))
ii = 2**(L-1)
numps = [1,4,8]
import matplotlib.pyplot as plt

for nump in numps:
  (es,ds) = H.get_dos_i(i=ii,kpm_scale=4,delta=1e-1,npol_scale=nump)
  plt.plot(es,ds,label="N = "+str(nump)+"$N_0$")

plt.legend()

plt.xlabel("Energy")
plt.ylabel("DOS")

plt.show()

