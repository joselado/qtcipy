
import numpy as np
import sys
import os
try:
    sys.path.append(os.environ["PYQULAROOT"]) # pyqula
    from pyqula import geometry
except:
    print("pyqula has to be installed")
    print(" 1) use 'pip install pyqula' in your terminal")
    print(" 2) OR create an environmental variable PYQULAROOT with the address of pyqula in your system")
    exit()

def rint(x): return int(np.round(x)) # round int


def get_density_i_from_dos(m,fermi=0.,**kwargs):
    """Return electronic density at site i"""
    (es,ds) = get_dos_i(m,**kwargs) # energies and DOS
    ds = ds.real # real part
    den = np.trapz(ds[es<fermi])/np.trapz(ds) # return filling of the site
    return den



def get_density_i_direct(m,
        kpm_prec="single",
        kpm_scale = None, # scale of KPm method
        kernel="jackson", # kernel
        **kwargs):
    from pyqula.kpmtk.density import get_density
    den = get_density(m,kpm_prec=kpm_prec,scale=kpm_scale,
            kernel=kernel,**kwargs)
#    den1 = get_density_i_from_dos(m,**kwargs)
#    print(den,den1)
    return den



#get_density_i = get_density_i_from_dos # use this method
get_density_i = get_density_i_direct # use this method



#
def memoize(func):
    """Decorator to use a cache"""
    cache = {}
    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return memoized_func
#
#




def get_den_kpm_qtci(h,**kwargs):
    """Return the full density using QTCI"""
    return get_den_kpm_qtci_general(h,**kwargs)



def get_den_kpm_qtci_general(h,log=None,**kwargs):
    """Return the electronic density of the system uisng KPM and QTCI"""
    f = get_function(h,**kwargs) # get the function to interpolate
    nb = get_nbits(h,**kwargs) # return the number of bits
    lim = get_lim(h,**kwargs) # get the limits
    IP = get_interpolator(h,f,nb,lim,
            **kwargs) # keyword arguments
    update_log(log,h,IP,**kwargs) # update the log
    out = evaluate_interpolator(h,IP,**kwargs) # evaluate the interpolator
    return out # return the output


def update_log(log,h,IP,info_qtci=False,**kwargs):
    """Update the log"""
    if log is not None: # make a log
        rse,zse = IP.get_evaluated()
        log["QTCI_eval"].append(IP.frac) # ratio of evaluations
        log["qtci_args"] = IP.qtci_args # store
        log["QTCI_error"].append(IP.error) # store
    if info_qtci:
         print(IP.frac,"ratio of QTCI evaluations in the SCF")




def get_mz_kpm_qtci(h,log=None,**kwargs):
    """Return the magnetization of the system uisng KPM and QTCI"""
    f0 = get_function(h,**kwargs) # get the density to interpolate
    def f(i): # new function
        return (f0(i)-0.5)*2. # magnetization
    nb = get_nbits(h,**kwargs) # return the number of bits
    lim = get_lim(h,**kwargs) # get the limits
    IP = get_interpolator(h,f,nb,lim,
            **kwargs) # keyword arguments
    update_log(log,h,IP,**kwargs) # update the log
    out = evaluate_interpolator(h,IP,**kwargs) # evaluate the interpolator
    return np.array(out) # return the output magnetization




def get_profile_qtci(h,f,log=None,**kwargs):
    """Return the magnetization of the system uisng KPM and QTCI"""
    nb = get_nbits(h,**kwargs) # return the number of bits
    lim = get_lim(h,**kwargs) # get the limits
    IP = get_interpolator(h,f,nb,lim,
            **kwargs) # keyword arguments
    update_log(log,h,IP,**kwargs) # update the log
    out = evaluate_interpolator(h,IP,**kwargs) # evaluate the interpolator
    return np.array(out) # return the output magnetization










def evaluate_interpolator(h,IP,dim=1,**kwargs):
    nat = h.shape[0]
    if dim==1: # 1D
        out = np.zeros(nat,dtype=np.float32) # initialize
        for i in range(nat): out[i] = IP(float(i)) # store result
        return out
    elif dim==2: # 2D
#        raise
        print("Using 2D")
        n = rint(np.sqrt(nat)) # lateral size of the system
        out = np.zeros(nat,dtype=np.float_) # initialize
        for i in range(n): 
            for j in range(n): 
                ii = n*i + j # index in real space
                out[ii] = IP(float(i),float(j)) # store result
        return out
    else: raise



def get_interpolator(h,f,nb,lim,dim=1,backend="C++",
        qtci_tol=1e-2,**kwargs):
    """Return the interpolator"""
    from . import discreteinterpolator as interpolate
    if dim==1: # one dimensional
        IP = interpolate.Interpolator(f,tol=qtci_tol,nb=nb,xlim=lim[0],
                dim=1,backend=backend,**kwargs)
    elif dim==2: # two dimensional
        IP = interpolate.Interpolator(f,tol=qtci_tol,nb=nb,xlim=lim[0],
                ylim=lim[1],dim=2,**kwargs)
    else: raise # error otherwise
    return IP # return the interpolator




def get_lim(h,dim=1,qtci_norb=1,**kwargs):
    """Return the limits"""
    if dim==1: # one dimensional
        n = h.shape[0] # number of sites
        if qtci_norb>1: n = n//qtci_norb # by the number of orbitals
        xlim = [0,n] # limits of the interpolation
        return xlim,None
    elif dim==2: # two dimensional
        n = h.shape[0] # number of sites
        if qtci_norb>1: n = n//qtci_norb # by the number of orbitals
        n = rint(np.sqrt(n)) # lateral size of the system
        xlim = [0,n] # limits of the interpolation
        return xlim,xlim # return the limits
    else: raise # not implemented




def get_nbits(h,qtci_norb=1,dim=1,**kwargs):
    """Get the number of required bits"""
    n = h.shape[0] # number of sites
    if qtci_norb>1: n = n//qtci_norb # number of cells
    if dim==1: pass # ignore for 1d
    elif dim==2: # 2d
      n = rint(np.sqrt(n)) # lateral size of the system
    else: raise
    nb = np.log(n)/np.log(2) # number of pseudospin sites
    if np.abs(int(nb)-nb)>1e-5:
        print("Number of points must be a power of 2")
        raise
    return rint(nb) # return number of bits 



def get_function(h,dim=1,**kwargs):
    """Return the function to interpolate"""
    ### norb only implemented for 2d ###
    kpm_scale = estimate_bandwidth(h) # compute the scale just once
    kwargs["kpm_scale"] = kpm_scale # overwrite
    def f1d(i): # function to interpolate
        ii = rint(i) # round the value
        if not 0<=ii<h.shape[0]: # fix and say
            print("WARNING, you are out of bounds!!!!")
            if ii<0: ii = 0 # fix
            else: ii = h.shape[0]-1 # last one
        out = get_density_i(h,i=ii,**kwargs) # get the density
        return out
    def f2d(i,j): # function to interpolate
        n = h.shape[0] # number of sites
        n = rint(np.sqrt(n)) # lateral size of the system
        ii = n*i + j # index in real space
        return get_density_i(h,i=rint(ii),**kwargs)
    if dim==1: return f1d
    elif dim==2: return f2d
    else: raise










import os ; import sys
sys.path.append(os.environ["PYQULAROOT"]) # pyqula

from pyqula.kpmtk.bandwidth import estimate_bandwidth
from pyqula import kpm


def get_dos_i(m,i=0,
        delta=1e-1, # effective smearing
        kpm_prec="single",
        kpm_scale = None, # scale of KPm method
        kernel="jackson", # kernel 
        npol_scale=4, # rescale number of polynomials
        **kwargs):
    """Return electronic density at site i"""
    if kpm_scale is None: # if none provided
        scale = estimate_bandwidth(m) # estimate the bandwidth
    else: scale = kpm_scale # given from input
    npol = rint(npol_scale*scale/delta) # number of polynomials
    ne = npol*10 # scale the number of energies accordingly
    (es,ds) = kpm.ldos(m,i=i,ne=ne,kernel=kernel,kpm_prec=kpm_prec,
            scale=scale,
            npol=npol,**kwargs) # compute the LDOS with KPM
    return es,ds.real




def estimate_qtci_maxm(h,R,f,qtci_tol=1e-2,**kwargs):
    """Estimate a good guess for the bond dimension"""
    from . import discreteinterpolator as interpolate
    nb = get_nbits(h,**kwargs) # return the number of bits
    lim = get_lim(h,**kwargs) # get the limits
    if callable(f): # if function provided
        fo = lambda i: f(R[rint(i),:]) # assume it is a function
    else:
        fo = lambda i: f[rint(i)] # assume it is an array/list
    IP = interpolate.Interpolator(fo,tol=qtci_tol,nb=nb,xlim=lim[0],
                qtci_recursive = True,
                **kwargs,
                dim=1)
    return IP.opt_qtci_maxm,IP.frac 








