# routines to perform selfconsistent calculations
# of a tight binding model

import numpy as np
import sys
#import os
#sys.path.append(os.environ["PYQULAROOT"]) # pyqula

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


def get_density_i(m,fermi=0.,**kwargs):
    """Return electronic density at site i"""
    from .kpmrho import get_dos_i
    (es,ds) = get_dos_i(m,**kwargs) # energies and DOS
    ds = ds.real # real part
    return np.trapz(ds[es<fermi])/np.trapz(ds) # return filling of the site


def get_den_ed(h,fermi=0.,**kwargs):
    """Return the total electronic density using exact diagonalization"""
    from scipy.linalg import eigh
    if h.shape[0]>20000: raise # sanity check
    (es,ws) = eigh(h.todense()) # diagonalize
    ws = ws.T # wavefucntions as rows
    out = 0. # initialize
    for i in range(len(es)):
        if es[i]<fermi: out += np.abs(ws[i])**2 # add contribution
    return out






def get_mz_ed(h,**kwargs):
    """Magnetization for chiral systems"""
    return (get_den_ed(h,**kwargs) - 0.5)*2 # magnetization







def get_den_kpm(h,use_qtci=True,**kwargs):
    """Return the electronic density of the system uisng KPM and QTCI"""
    if use_qtci: # use quantics tensor cross interpolation
        from .kpmrho import get_den_kpm_qtci
        return get_den_kpm_qtci(h,**kwargs)
    else: # brute force
        def f(i): # function to interpolate
            return get_density_i(h,i=int(i),**kwargs)
        return np.array([f(i) for i in range(0,h.shape[0])])



def get_mz_kpm(h,use_qtci=True,**kwargs):
    """Return the electronic density of the system uisng KPM and QTCI"""
    if use_qtci: # use quantics tensor cross interpolation
        from .kpmrho import get_mz_kpm_qtci
        return get_mz_kpm_qtci(h,**kwargs)
    else: # brute force
        def f(i): # function to interpolate
            dup = get_density_i(h,i=int(i),**kwargs)
            return dup - 0.5 # magnetization
        return np.array([f(i) for i in range(0,h.shape[0])])







def get_den(h,use_kpm=False,**kwargs):
    """Get the electronic density of a matrix"""
    if use_kpm: return get_den_kpm(h,**kwargs) # compute using the KPM
    else: return get_den_ed(h,**kwargs) # compute using ED



def get_mz(h,use_kpm=False,**kwargs):
    """Get the electronic density of a matrix"""
    if use_kpm: return get_mz_kpm(h,**kwargs) # compute using the KPM
    else: return get_mz_ed(h,**kwargs) # compute using ED



def SCF_Hubbard(scf,maxerror=1e-3,maxite=None,
                log=None, # dictionary for logs
                use_dynamical_qtci = True, # update the QTCI on the fly
                chiral_AF = False, # flag to enforce chiral AF
                mix=0.3,info=False,**kwargs):
    """
    Perform a selfconsistent Hubbard calculation
       - h0 is the single particle Hamiltonian
       - U is the Hubbard interaction
       - dup is the initial guess for the up density
       - ddn is the initial guess for the dn density
       - maxerror is the maximum error of the selfconsistent loop
       - mix mixes the mean field, for stability
       """
    ### extract the required objects
    h0 =  scf.H0.H
    U =   scf.U # Hubbard
    dup = scf.MF[0].copy() # first mean-field
    ddn = scf.MF[1].copy() # second mean-field
    log = scf.log # the log
    kwargs["AB"]  = scf.H0.AB # the sublattice
    kwargs["dim"]  = scf.H0.dim # the dimensionality
    # initialize a local log
    if log is not None:
        log0 = dict()
        log0["QTCI_eval"] = []
        log0["QTCI_error"] = []
        log0["opt_qtci_maxm"] = log["opt_qtci_maxm"]
    else: log0 = None # default
    ddn_old = ddn.copy() # make a copy
    dup_old = dup.copy() # make a copy
    from scipy.sparse import diags
    ite = 0
    import time
    t0 = time.time() # get the time
    while True: # infinite loop
        if info: print("Starting SCF loop")
        ite += 1 # iteration
        hup = h0 + diags(U*(ddn_old-0.5),shape=h0.shape) # up Hamiltonian
        hdn = h0 + diags(U*(dup_old-0.5),shape=h0.shape) # down Hamiltonian
        if chiral_AF: # by symmetry for chiral AF systems
            mz = get_mz(hup,log=log0,**kwargs)
            dup = 0.5 + mz/2. # up density
            ddn = 0.5 - mz/2. # down density
        else: # compute explicitly
            ddn = get_den(hdn,log=log0,**kwargs) # generate down density
            dup = get_den(hup,log=log0,**kwargs) # generate up density
        error = np.mean(np.abs(ddn-ddn_old) + np.abs(dup-dup_old)) # error
        if log is not None: # do the logs
            log["opt_qtci_maxm"] = log0["opt_qtci_maxm"]
            log["SCF_time"].append(time.time() - t0) # store time
            log["SCF_error"].append(error) # store time
        if info: 
            print("SCF Error",error,"iteration",ite)
            print("Average magnetization",np.mean(np.abs(ddn-dup)))
            print("Max magnetization",np.max(np.abs(ddn-dup)))
        mz = dup - ddn # magnetization
        if use_dynamical_qtci: # update the QTCI options
            from .dynamicalqtci import overwrite_qtci_kwargs
            from .dynamicalqtci import get_qtci_kwargs
            qtci_kwargs = get_qtci_kwargs(kwargs,mz,scf_error=error) # get the new one
            overwrite_qtci_kwargs(kwargs,qtci_kwargs) # overwrite
            scf.qtci_kwargs = qtci_kwargs # overwrite the options
        dup_old = mix*dup_old + (1.-mix)*dup # update
        ddn_old = mix*ddn_old + (1.-mix)*ddn # update
        # stopping criteria
        if error<maxerror: break # stop loop
        if maxite is not None:
            if ite>=maxite: break
    if log is not None: # up down logs
        ev = log0["QTCI_eval"] 
        qterr = log0["QTCI_error"] 
        if not chiral_AF: # resum if needed
            ev = [(ev[2*i] + ev[2*i+1])/2. for i in range(len(ev)//2)] # resum
            qterr = [(qterr[2*i] + qterr[2*i+1])/2. for i in range(len(ev)//2)] # resum
        log["QTCI_eval"] += ev # store
        log["QTCI_error"] += qterr # store
    # convert to single (real) precision
    hup = hup.astype(np.float32)
    hdn = hdn.astype(np.float32)
    dup = dup.astype(np.float32)
    ddn = ddn.astype(np.float32)
    # update the SCF object
    scf.H[0] = hup.copy() # replace matrix
    scf.H[1] = hdn.copy() # replace matrix
    scf.H = [hup,hdn] # store the resulting Hamiltonian
    scf.MF = [dup,ddn] # store the densities
    scf.Mz = dup - ddn # magnetization
    return scf


