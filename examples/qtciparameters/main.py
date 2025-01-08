import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 4 # exponential length
H = hamiltonians.chain(L) # get the Hamiltonian

def f(r):
    omega = np.pi*2.*np.sqrt(2.)/20
    return 1. + 0.2*np.cos(omega*r[0]) #+0.2*np.cos(np.pi*2.*np.sqrt(3.)*r[0])
H.modify_hopping(f)

# get the SCF object
SCF = H.get_SCF_Hubbard(U=3.0) # generate a selfconsistent object

SCF.solve(info=True,
        use_qtci=True,use_kpm = True,
        info_qtci = True,
        maxite = 1,
        delta= 1e-3,
        use_dynamical_qtci = True,
        qtci_maxfrac = 0.1,
        qtci_maxm = 40,
        qtci_norb = 2,
        backend = "C++",
        chiral_AF = True, # use symmetry for chiral models
        ) # solve the SCF

print(SCF.qtci_kwargs)

