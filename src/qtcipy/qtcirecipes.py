import numpy as np

from .tbscftk import discreteinterpolator
from .qtcirecipestk.qtcikernels import random_kernel as random_qtci_kernel


def rint(x): return int(np.round(x)) # round int


methods = ["maxm","accumulative"]
#methods = ["accumulative"]


#### Acummulative mode has segfault randomly ####

def optimal_qtci(v,recursive=False,
        qtci_error_factor = 1.3,
        kwargs0=None,**kwargs):
    methods_f = [] # empty list
    if "maxm" in methods: methods_f.append(optimal_maxm) 
    if "accumulative" in methods: methods_f.append(optimal_accumulative) 
    outs = [] # storage
    if kwargs0 is not None: # if initial guess was given
        from .qtcirecipestk.refine import refine_qtci_kwargs
        o = refine_qtci_kwargs(v,kwargs0,**kwargs) # get best fraction
        if o is not None: # this is ok
            frac0,kw0 = o[0],o[1]
            outs.append([frac0,kw0])
    for method in methods_f:
        out = method(v,**kwargs) # compute this one
        if out is not None: # sucess
            outs.append(out) # store
    kwmin,fracmin = None,1.0 # initialize the optimal ones
    if len(outs)>0: # if any succeded
        for out in outs:
            if out[0]<fracmin:
                fracmin = out[0]
                kwmin = out[1]
    if kwmin is not None: # if an optimal one has been found
        return fracmin,kwmin
    else: # not a good one has been found
        from copy import deepcopy
        kw = deepcopy(kwargs)
        if recursive: # if you want to try again
            if "qtci_error" in kw:
                kw["qtci_error"] = qtci_error_factor*kw["qtci_error"]
            else:
                kw["qtci_error"] = 0.01
            if "qtci_opt_ntries" in kw:
                kw["qtci_refine_ntries"] = kw["qtci_opt_ntries"] + 1
            else:
                kw["qtci_refine_ntries"] = 5
            print("Recalling QTCI optimization with lower threshold",kw["qtci_error"])
            return optimal_qtci(v,recursive=True,
                    qtci_error_factor=qtci_error_factor,
                    kwargs0=kwargs0,**kw)
        else: # give up, return the input
            print("Nothing found")
            return None,kwargs0 # none succeded



def get_qtci_flags(kwargs):
    """Retain only the QTCI flags"""
    flags = ["qtci_maxm","qtci_accumulative","qtci_pivot1",
            "norb","qtci_tol","qtci_pivots","qtci_args"]
    out = dict()
    for key in kwargs: # loop
        if key in flags:
            out[key] = kwargs[key]
    return out




def get_frac_args(v,qtci_error=1e-3,**kwargs):
    """Get the fraction with a certain set of parameters"""
    nb = get_nbits(v,**kwargs)
    lim = get_lim(v,**kwargs)
    if np.mean(np.abs(v))<qtci_error/100: return 1.,None # flat distribution
    vi = v + qtci_error/100*(np.random.random(len(v))-0.5) # add noise
    f = lambda i: vi[rint(i)] # function
    IP = get_interpolator(f,nb,lim,**kwargs) 
    from .qtcidistance import get_distance
    disf = get_distance() # get the distance function
    erri = disf([IP(i) for i in range(len(v))],v)
    if erri<qtci_error: # this is ok, return 
        kw = IP.get_kwargs()
        return IP.frac,IP.get_kwargs() # return the fraction
    else: 
        return None   # this did not work




def get_interpolator(f,nb,lim,dim=1,backend="C++",
        qtci_tol=1e-2,**kwargs):
    """Return the interpolator"""
    from .tbscftk import discreteinterpolator as interpolate
    if dim==1: # one dimensional
        IP = interpolate.Interpolator(f,tol=qtci_tol,nb=nb,xlim=lim,
                dim=1,backend=backend,**kwargs)
    elif dim==2: # two dimensional
        IP = interpolate.Interpolator(f,tol=qtci_tol,nb=nb,xlim=lim[0],
                ylim=lim[1],dim=2)
    else: raise # error otherwise
    return IP # return the interpolator




def get_nbits(v,norb=1,dim=1,**kwargs):
    """Get the number of required bits"""
    n = len(v) # number of sites
    if norb>1: n = n//norb # number of cells
    if dim==1: pass # ignore for 1d
    elif dim==2: # 2d
      n = rint(np.sqrt(n)) # lateral size of the system
    else: raise
    nb = np.log(n)/np.log(2) # number of pseudospin sites
    if np.abs(int(nb)-nb)>1e-5:
        print("Number of points must be a power of 2")
        raise
    return rint(nb) # return number of bits




def get_lim(v,dim=1,norb=1,**kwargs):
    """Return the limits"""
    if dim==1: # one dimensional
        n = len(v) # number of sites
        if norb>1: n = n//norb # by the number of orbitals
        xlim = [0,n] # limits of the interpolation
        return xlim
    elif dim==2: # two dimensional
        n = h.shape[0] # number of sites
        if norb>1: n = n//norb # by the number of orbitals
        n = rint(np.sqrt(n)) # lateral size of the system
        xlim = [0,n] # limits of the interpolation
        return xlim,xlim # return the limits
    else: raise # not implemented











def optimal_maxm(v,qtci_error=1e-2,**kwargs):
    """Find the optimal QTCI with the non accumulative method"""
    ntries = 10 # average over these many tries
    kwargs_opt = None # output arguments
    err,frac,pivot,maxm = 1e7,1.0,None,None # initialize
    if len(v)<10: norbs = [1]
    else: norbs = [1,2,4]
    weights = np.abs(v - np.mean(v)) + qtci_error # weight for pivots
    maxms = [2+int(1.2**s) for s in range(30)] # set of bond dimensions
    for itry in range(ntries): # try as many times
        norbi = norbs[np.random.randint(len(norbs))] # one choice at random
#        for norbi in norbs: # loop over norbs
        nb = np.log(len(v)/norbi)/np.log(2) ; nb = rint(nb) # integer value
        fullPivi = np.random.random()<0.5 # True or False randomly
        maxmi = pick_randomly(maxms) # randomly
        #### generate global pivots ####
        use_gp = pick_randomly([True,False]) # True or False randomly
        ################################
        vi = v + qtci_error/100*(np.random.random(len(v)) - 0.5) # slightly random
        f = lambda i: vi[rint(i)] # function to interpolate
        wi = [weights[norbi*i] for i in range(2**nb)] # redefine weight
        qtci_tol = qtci_error*pick_randomly([0.3,1.,3]) # pick one
        IP = discreteinterpolator.interpolate_norb(f,norb=norbi,xlim=[0,2**nb],
                                       nb=nb,backend="C++",
                                       qtci_tol = qtci_tol,
                                       qtci_kernel = random_qtci_kernel(), # random kernel
                                       qtci_accumulative = False, # non acc mode
                                       qtci_fullPiv = fullPivi, # pivotting mode
                                       qtci_maxm=maxmi) # maxm
        from .qtcidistance import get_distance
        disf = get_distance() # get the distance function
        erri = disf([IP(i) for i in range(len(v))],v)
        fraci = IP.frac # fraction of the space
        if erri<qtci_error: # desired error level has been reached
            if fraci<frac: # better fraction of space than the stored one
                err = erri # store the error
                frac = fraci # store the best fraction
                kwargs_opt = IP.get_kwargs() # get the optimal kwargs 
    if kwargs_opt is None: return None # this did not work
#    print("Rook QTCI, optimal has fraction",frac,"error",err)
    # do an optimization of the chosen one
    from .qtcirecipestk.refine import refine_qtci_kwargs
    # get best fraction
    frac,kwargs_opt = refine_qtci_kwargs(v,kwargs_opt,qtci_error=qtci_error,**kwargs) 
    return frac,kwargs_opt # return optimal parameters



#
#def random_global_pivots(v):
#    """Generte random global pivots"""
#    use_gp = np.random.random()<0.5 # True or False randomly
#    qtci_args = {"qtci_use_global_pivots":use_gp} # store
#    ntries = 5
#    if use_gp: # use pivots
#        n = len(v) # number of elements
#        out = []
#        for i in range(ntries): # one every 100
#            out.append(pick_index(v)) # store
#        out = np.unique(out) # get unique ones
#        qtci_args["qtci_global_pivots_real"] = out
#    return qtci_args # return
#


def pick_randomly(v):
    """Given a list, return an element randomly"""
    return v[np.random.randint(len(v))] # return 




def optimal_accumulative(v,qtci_error=1e-2,**kwargs):
    """Find the optimal QTCI with the non accumulative method"""
    ntries = 10 # best of these many tries
    maxmi = 3 # start with 10
    kwargs_opt = None # none
    err,frac,pivot,maxm = 1e7,1.0,None,None # initialize
    if len(v)<10: norbs = [1]
    else: norbs = [1,2,4]
    maxms = [2+int(1.2**s) for s in range(30)] # set of bond dimensions
    weights = np.abs(v - np.mean(v)) + qtci_error # weight for pivots
    for it in range(ntries): # these many tries, pick the best
        norbi = norbs[np.random.randint(len(norbs))] # one choice at random
        vi = v + qtci_error/100*(np.random.random(len(v)) - 0.5) # slightly random
        f = lambda i: vi[rint(i)] # function to interpolate
        nb = np.log(len(v)/norbi)/np.log(2) ; nb = rint(nb) # integer value
        tol_fac = pick_randomly([1.0,0.5,2.]) # factors to consider
        qtci_tol = qtci_error*tol_fac # check a potential refactoring
        fullPivi = pick_randomly([True,False]) # full pivots
        maxmi = pick_randomly(maxms) # bond dimension
        IP = discreteinterpolator.interpolate_norb(f,norb=norbi,xlim=[0,2**nb],
                                       nb=nb,backend="C++",
                                       qtci_tol = qtci_tol, # tolerance
                                       qtci_kernel = random_qtci_kernel(), # random kernel
                                       qtci_accumulative = True, # non acc mode
                                       qtci_fullPiv = fullPivi, # pivotting mode
                                       qtci_maxm=maxmi) # maxm
        from .qtcidistance import get_distance
        disf = get_distance() # get the distance function
        erri = disf([IP(i) for i in range(len(v))],v)
        fraci = IP.frac # fraction of the space
        if erri<qtci_error: # desired error level has been reached
            if fraci<frac: # better fraction of space than the stored one
                err = erri # store the error
                frac = fraci # store the best fraction
                kwargs_opt = IP.get_kwargs() # get the optimal kwargs 
        maxmi = int(1.1*maxmi) + 1 # next bond dimension
    if kwargs_opt is None: return None # this did not work
#    print("Accumulative QTCI, optimal has fraction",frac,"error",err)
    # do an optimization of the chosen one
    from .qtcirecipestk.refine import refine_qtci_kwargs
    frac,kwargs_opt = refine_qtci_kwargs(v,kwargs_opt,qtci_error=qtci_error,**kwargs) # get best fraction
    return frac,kwargs_opt # return optimal parameters

