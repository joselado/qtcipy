from ..qtcirecipes import get_frac_args

from copy import deepcopy as cp
import numpy as np

optimizations = ["maxm","global_pivots","pivot1","tol","norb"]


def refine_qtci_kwargs(v,kw,qtci_refine_ntries=5,**kwargs):
    for i in range(qtci_refine_ntries): # these many tries
        frac,kw = refine_qtci_kwargs_single(v,kw,**kwargs)
    return frac,kw




def refine_qtci_kwargs_single(v,kw,**kwargs):
    """Do small changes to the QTCI to see if it gets better"""
    frac0,kw0 = get_frac_args(v,qtci_error=1e10,**kw)
    if "global_pivots" in optimizations:
        frac,kw = refine_global_pivots(v,kw,**kwargs) # add global pivots
    if "pivot1" in optimizations:
        frac,kw = refine_pivot1(v,kw,**kwargs) # refine the pivot
    if "maxm" in optimizations:
        frac,kw = refine_maxm(v,kw,**kwargs) # refine the bond dimension
    if "tol" in optimizations:
        frac,kw = refine_tol(v,kw,**kwargs) # refine the tolerance
    if "kernel" in optimizations:
        frac,kw = refine_kernel(v,kw,failsafe,**kwargs) # optimize the kernel
    if "norb" in optimizations:
        frac,kw = refine_norb(v,kw,**kwargs) # optimize the kernel
    return frac,kw # return the optimized result



def refine_maxm(v,kw,**kwargs):
    """Refine the bond dimension"""
    ps = [1.0,1.0,1.0]
    ps[1] = 1.0 - 0.3*np.random.random() # random decrease
    ps[2] = 1.0 + 0.3*np.random.random() # random increase
    def convert(p,f):
        if p*f<3.: return 3 # minimum bond dim
        elif p*f>400: return 800 # maximum bond dim
        else: return int(np.round(p*f)) # the closest integer
    return refine_parameter(v,kw,name="qtci_maxm",p0=10,convert=convert,
            scales=ps,**kwargs)



def refine_norb(v,kw,**kwargs):
    """Refine the number of orbitals"""
    ps = [1,2,4]
    def convert(p,f):
        return f # return the Norb to try
    return refine_parameter(v,kw,name="qtci_norb",p0=1,convert=convert,
            scales=ps,**kwargs)




def refine_pivot1(v,kw,**kwargs):
    """Refine the bond dimension"""
    ps = [1.0,None,None] # original and two news
    def convert(p,f):
        if f is None: return None # force finding a new one
        else: return p # old one
    return refine_parameter(v,kw,name="qtci_pivot1",p0=None,
            convert=convert,**kwargs)



def refine_tol(v,kw,**kwargs):
    """Refine the bond dimension"""
    ps = [1.0,1.0,1.0]
    ps[1] = 1.0 - 0.3*np.random.random() # random decrease
    ps[2] = 1.0 + 0.3*np.random.random() # random increase
    return refine_parameter(v,kw,name="qtci_tol",p0=0.01,
            scales=ps,**kwargs)






def refine_global_pivots(v,kw,**kwargs):
    """Refine the global pivots"""
    if "qtci_accumulative" in kw:
        if kw["qtci_accumulative"]: 
            ## accumulative mode does not have pivots
            return 1.0,kw # do nothing
    ps = ["nothing","add","remove"]
    def convert(p0,f):
        if f=="nothing": return p0 # do nothing
        elif f=="add": 
            return [ip for ip in p0] + [None]
        elif f=="remove":
            if len(p0)>0:
                pi = [ip for ip in p0] # make a copy
                del pi[np.random.randint(len(pi))] # remove one
                return pi
            else: return [] # empty list
        else: raise
    return refine_parameter(v,kw,name="qtci_global_pivots",p0=[],
            convert = convert,
            in_qtci_args=True,
            scales=ps,**kwargs)






#
#def refine_kernel(v,kw,**kwargs):
#    """Refine the kernel"""
#    kps = [kp,kp,kp]
#    kps[1] = kp*(1. + 0.3*np.random.random()) # increase randomly
#    kps[2] = kp*(1. - 0.3*np.random.random()) # reduce randomly
#    return refine_parameter(v,kw,name="qtci_kernel",p0=0.01,
#            convert = convert,
#            scales=kps,**kwargs)
#
#



def refine_parameter(v,kw,name=None,p0=None,convert=lambda x,y: x*y,
        failsafe=True,in_qtci_args=False,
        scales = [1.],**kwargs):
    """Change a parameter to optimize the QTCI"""
    kw0 = cp(kw) # make a copy, just in case this fails
    if kw is None: return 1.0,None
    # input is all the kwargs of QTCI, the variable is in qtci_kernel
    p = p0
    if not in_qtci_args: # not in qtci_args
        if name in kw: # check if the parameter is present
            p = kw[name] # get the parameter
    else: # if in qtci args
        if "qtci_args" in kw: # if it has been set up
            if name in kw["qtci_args"]: # check if the parameter is present
                p = kw["qtci_args"][name] # get the parameter
        else: # dictionary does not exist, create it
            kw["qtci_args"] = {name : p0} # create the dictionary
    ps = scales # parameters redefinitions to try
    kwlist = [] # empty list
    for pi in ps: # loop
        pi = convert(p,pi) # make a conversion, if needed
        kwi = cp(kw) # make a copy
        if not in_qtci_args: # not in qtci_args
            kwi[name] = pi # overwrite
        else: # in QTCI args
            kwi["qtci_args"][name] = pi # overwrite
        kwlist.append(kwi) # store
    frac,kwbest = best_kwargs(v,kwlist,**kwargs) # return the best
    if kwbest is None: 
        if failsafe: return 1.0,kw0 # backup option
    return frac,kwbest # return the ones found





def refine_kernel(v,kw,**kwargs):
    """Change the kernel, to optimize the QTCI"""
    # input is all the kwargs of QTCI, the variable is in qtci_kernel
    if kw is None: return 1.0,None
    kp = 0.25
    if "qtci_kernel" in kw:
        if "qtci_power_kernel" in kw:
            kp = kw["qtci_power_kernel"] # get the previous kernel
    kps = [kp,kp,kp]
    kps[1] = kp*(1. + 0.3*np.random.random()) # increase randomly
    kps[2] = kp*(1. - 0.3*np.random.random()) # reduce randomly
    kwlist = [] # empty list
    for kpi in kps: # loop
        kwi = cp(kw) # make a copy
        kwi["qtci_kernel"] = {"qtci_power_kernel":kpi}
        kwlist.append(kwi) # store
    return best_kwargs(v,kwlist,**kwargs) # return the best













def bool_invert(kw1,kw2,key,default,in_qtci_args=True):
    """Set the opposite for a bool keyword, with a default"""
    if in_qtci_args: # argument is stored in qtci_args 
        if key in kw2["qtci_args"]: # if key is present
            kw1["qtci_args"][key] = not kw2["qtci_args"][key]
        else: kw1["qtci_args"][key] = default
    else:
        if key in kw2: # if key is present
            kw1[key] = not kw2[key]
        else: kw1[key] = default




def global_pivot_refinement(v,kws,**kwargs):
    """Hard coded ways of improving a QTCI"""
    acc = "qtci_accumulative"
    ### if non accumulative mode, try to add global pivots"""
    kwlist = [kws]
    if acc in kws: # this is the accumulative mode
        if not kws[acc]: # non accumulative mode 
            kwi = cp(kws) # make a copy
            # try the opposite option
#            bool_invert(kwi,kws,"qtci_use_global_pivots",True)
            kwi["qtci_use_global_pivots"] = True
            kwlist.append(kwi) # add this one
    # return the best one
    return best_kwargs(v,kwlist,**kwargs)






def best_kwargs(v,kwarg_list,**kwargs):
    """Given a list of QTCI kwargs, return the best one"""
    kwmin,fracmin = None,1.0 # initialize
    for kw in kwarg_list: # loop
        o = get_frac_args(v,**kw,**kwargs) # compute this one
        if o is not None: # ok to check
            if o[0]<fracmin: # if better than previous
                fracmin = o[0] # store
                kwmin = o[1] # store
    # if no kwargs satisfy the qtci_error criteria, the output will be None
    # this has to be fixed
    return fracmin,kwmin




