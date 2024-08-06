import numpy as np

from ..qtcirecipes import optimal_qtci


def get_qtci_kwargs(kwargs,v,scf_error=None,**kw):
    """Overwrite the QTCI optional arguments according
    to how the mean field is evolving"""
    from ..qtcidistance import qtci_error # the default error
    tol = qtci_error # default tol
    tol = 1e-2 # start with this
    if "qtci_tol" in kwargs: # overwrite if it is given
        tol = kwargs["qtci_tol"] # target tolerance
    else: # not given
        if scf_error is not None: # if given
            tol = min([tol,scf_error/2.]) # overwrite
    # obtain the optimal QTCI for this data
    frac,qtci_kwargs = optimal_qtci(v,qtci_error=tol,kwargs0=kwargs,
            qtci_error_factor = 1.01, # call again if needed with the same error
            recursive=True,**kw)
    if qtci_kwargs is None: # none succeded
        print("No fitting QTCI found, using default")
        return get_default(v)
    else: 
#        print("Next iteration uses new QTCI found with fraction",frac)
#        print(qtci_kwargs)
        return qtci_kwargs

def overwrite_qtci_kwargs(kwargs,qtci_kwargs):
    for key in qtci_kwargs: # loop over all the keys
        kwargs[key] = qtci_kwargs[key] # overwrite
   #     print("Updating",key)




def get_default(v):
    """Return a default set of parameters for the QTCI"""
    qtci_kwargs = {"qtci_maxm":400} # reasonable guess
    qtci_kwargs["qtci_accumulative"] = True # accumulative mode
    qtci_kwargs["qtci_tol"] = 1e-2 # initial tol
    return qtci_kwargs # return this




from copy import deepcopy as cp


def initial_qtci_kwargs(SCF,**kwargs):
    """Return a reasonable initial guess for the kwargs
    of a QTCI for an SCF object"""
    if "use_qtci" in kwargs:
        if not kwargs["use_qtci"]: return {}
    if SCF.qtci_kwargs is None: # first iteration
        qtci_kwargs = {"qtci_maxm":100} # reasonable guess
        qtci_kwargs["qtci_accumulative"] = True # accumulative mode
        qtci_kwargs["qtci_tol"] = 1e-1 # initial tol
        SCF0 = SCF.copy() # make a copy
        SCF0.qtci_kwargs = qtci_kwargs # overwrite
        kw = cp(kwargs) # make a copy
        kw["kpm_delta"] = 2.0
        kw["delta"] = 2.0
        kw["use_qtci"] = True
        kw["use_kpm"] = True
        kw["backend"] = "C++"
        kw["info"] = False
        kw["maxite"] = 1 # one iteration
        mz = SCF0.H0.get_moire()*SCF0.MF[0] # make a guess
        # get some kwargs
        frac,qtci_kwargs = optimal_qtci(mz,recursive=True) 
        SCF0.qtci_kwargs = qtci_kwargs # use these ones
        SCF0.solve(**kw) # one iteration without accuracy
        print("SCF Initialization DONE")
        return SCF0.qtci_kwargs
    else: return SCF.qtci_kwargs # return this choice

maxerror_dyn_qtci = 1e-2

def dynamical_update(scf,use_dynamical_qtci=True,
        maxerror_dyn_qtci=maxerror_dyn_qtci, # max error for dynamical QTCI 
        info = False,
        use_qtci=True,**kwargs):
    """Dynamically update the QTCI"""
    if not use_qtci: return # do nothing
    mz = scf.Mz # magnetization
    error = scf.scf_error
    qtci_kwargs = scf.qtci_kwargs
    if error<maxerror_dyn_qtci: return # return if the error is small
    from ..qtcirecipestk.refine import refine_qtci_kwargs
    if use_dynamical_qtci: # update the QTCI options
        if info: print("Dynamical update of the QTCI")
        frac,qtci_kwargs = refine_qtci_kwargs(mz,qtci_kwargs)
#        qtci_kwargs = get_qtci_kwargs(kwargs,mz,scf_error=error) # get the new one
#        overwrite_qtci_kwargs(kwargs,qtci_kwargs) # overwrite
        scf.qtci_kwargs = qtci_kwargs # overwrite the options
    else: return # do nothing












