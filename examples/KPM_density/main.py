import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 4 # exponential length
H = hamiltonians.chain(L) # get the Hamiltonian

H.add_onsite(lambda r: np.random.random()) # random onsites

fermi = 0.5

den0 = H.get_density(use_kpm=False,fermi=fermi)
den1 = H.get_density(use_kpm=True,fermi=fermi)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6,4))
plt.scatter(range(len(den0)),den0,s=120,c="red",label="ED")
plt.scatter(range(len(den0)),den0,s=40,c="blue",label="KPM")
plt.xlabel("Site")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.show()

