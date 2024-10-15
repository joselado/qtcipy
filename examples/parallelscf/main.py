import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 12 # exponential length
H = hamiltonians.chain(L) # get the Hamiltonian

def f(r):
    omega = np.pi*2.*np.sqrt(2.)/20
    return 1. + 0.2*np.cos(omega*r[0]) #+0.2*np.cos(np.pi*2.*np.sqrt(3.)*r[0])
H.modify_hopping(f)

# get the SCF object
SCF = H.get_SCF_Hubbard(U=3.0) # generate a selfconsistent object

#m = SCF.estimate_qtci_maxm(f)

#SCF = SCF.load() ; MF = SCF.MF

#m = 10
#print(m) ; exit()


#SCF.solve(info=True) # solve the SCF
SCF.solve(info=True,
        use_qtci=True,use_kpm = True,
        info_qtci = True,
#        maxite = 7,
        delta= 1e-3,
        use_dynamical_qtci = True,
        qtci_norb = 1,
        qtci_maxfrac = 0.1,
        backend = "C++",
        chiral_AF = True, # use symmetry for chiral models
        ) # solve the SCF

#print(SCF.qtci_kwargs)
SCF.save()
SCF = SCF.load()
print(SCF.log["QTCI_eval"])
print(SCF.qtci_kwargs)

#print("Loading from file")
#SCF.solve(info=True,use_qtci=True,use_kpm=False) # solve the SCF

