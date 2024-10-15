
import numpy as np

from ..interpolate import Interpolator as Interpolator_single

def Interpolator(f,qtci_norb=1,**kwargs):
    return interpolate_norb(f,qtci_norb=qtci_norb,**kwargs) # conventional case
    if norb==1: # one orbital
        return Interpolator_single(f,**kwargs) # conventional case
    else: # several orbitals
        return interpolate_norb(f,qtci_norb=qtci_norb,**kwargs) # conventional case




class Discrete_Interpolator():
    def __init__(self,IP):
        """Dummy class for a discrete interpolator, purely with Python
        objects"""
        self.frac = IP.frac
        self.nb = IP.nb
        x,y = IP.get_evaluated()
        self.x_ev = x.copy()
        self.y_ev = y.copy()
        self.error = IP.error
        self.qtci_maxm = IP.qtci_maxm # store
        self.qtci_accumulative = IP.qtci_accumulative # store
        self.qtci_kernel = IP.qtci_kernel # store
        self.qtci_fullPiv = IP.qtci_fullPiv # store 
        self.qtci_args = IP.qtci_args
        self.qtci_pivot1 = IP.qtci_pivot1
        self.qtci_tol = IP.qtci_tol
        self.out = [IP(i) for i in range(2**self.nb)]
    def __call__(self,i):
        return self.out[i]
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)
    def get_evaluated(self):
        return self.x_ev,self.y_ev



def interpolate_norb(f,dim=1,qtci_norb=1,info_qtci=False,**kwargs):
    """Obtain an interpolator, where the function f has a certain
    number of orbitals"""
    IPs = [] # empty list
    ev = [] # evaluated points
    def get_IP(iorb): # return the interpolator
        if dim==1: # one dimensional
            def fi(ii): return f(ii*qtci_norb + iorb) # redefine function
            IP = Interpolator_single(fi,dim=dim,
                    info_qtci = info_qtci,
                    **kwargs) # new interpolator
        else: raise # not implemented
        IP = Discrete_Interpolator(IP) # redefine
        return IP
    from .. import parallel
#    from pyqula import parallelmpi
#    print("Norb",qtci_norb)
    IPs = parallel.pcall(get_IP,range(qtci_norb)) # call all
    IP = Interpolator_norb(IPs) # full interpolator
    return IP


class Interpolator_norb():
    def __init__(self,IPs): 
        # do nothing
        self.nb = IPs[0].nb # number of bits
        self.frac = np.mean([IP.frac for IP in IPs])
        #self.qtci_args = [IP.qtci_args for IP in IPs] # store the list
#        if len(IPs)==1: # single one
        self.qtci_args = IPs[0].qtci_args # store the list
        # common for all #
        self.qtci_maxm = IPs[0].qtci_maxm # store the list
        self.qtci_accumulative = IPs[0].qtci_accumulative # store the list
        self.qtci_kernel = IPs[0].qtci_kernel # store the list
        self.qtci_fullPiv = IPs[0].qtci_fullPiv # store the list
        self.qtci_tol = IPs[0].qtci_tol # store the list
        self.qtci_pivot1 = IPs[0].qtci_pivot1 # store the list
        ##################
        self.error = np.mean([IP.error for IP in IPs])
        self.qtci_norb = len(IPs) # number of orbitals
        self.out = np.zeros(self.qtci_norb*(2**self.nb)) # initialize
        for iorb in range(self.qtci_norb): # loop
            for ii in range(2**self.nb): # loop over bits
                self.out[self.qtci_norb*ii + iorb] = IPs[iorb](ii) # call
        # store evaluated points
        nev = sum([len(IP.get_evaluated()[0]) for IP in IPs]) # number of evaluated points
        self.x_ev = np.zeros(nev) # initialize
        self.y_ev = np.zeros(nev) # initialize
        icount = 0
        for iorb in range(self.qtci_norb): # loop
            xs,ys = IPs[iorb].get_evaluated()
            for (x,y) in zip(xs,ys):
                self.x_ev[icount] = self.qtci_norb*x + iorb # store
                self.y_ev[icount] = y # store
    def __call__(self,i):
        return self.out[int(i)] # return result
    def get_evaluated(self):
        return self.x_ev,self.y_ev
    def get_kwargs(self):
        """Return a dictionary with the required kwargs"""
        out = dict()
        out["qtci_norb"] = self.qtci_norb
        out["qtci_maxm"] = self.qtci_maxm
        out["qtci_accumulative"] = self.qtci_accumulative
        out["qtci_kernel"] = self.qtci_kernel
        out["qtci_fullPiv"] = self.qtci_fullPiv
        out["qtci_tol"] = self.qtci_tol
        out["qtci_pivot1"] = self.qtci_pivot1
        out["qtci_args"] = self.qtci_args
        return out



