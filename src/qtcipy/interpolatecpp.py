import os
import sys

path = os.path.dirname(os.path.realpath(__file__))

# add xfacpy library
sys.path.append(path+"/pylib/xfac/build/python")

from .qtcirecipestk.qtcikernels import get_kernel

import xfacpy

class Interpolator():
    def __init__(self,f,xlim=[0.,1.],nb=20,
        qtci_maxm=100, # initial bond dimension
        qtci_accumulative = False,
        max_pivot0_tries = 100,
        qtci_tol = 1e-3,
        qtci_kernel = {}, # no kernel provided
        qtci_pivot1 = None,
        qtci_fullPiv = True,
            **kwargs):
        """Initialize the interpolator object"""
        # build the quantics grid
        qgrid = xfacpy.QuanticsGrid(a=xlim[0],b=xlim[1], nBit=nb)  
        self.kernel = get_kernel(qtci_kernel) # get the kernel
        self.inv_kernel = get_kernel(qtci_kernel,inverse=True) # get the kernel
        ## apply kernel to the function and tol
        fk = lambda x: self.kernel(f(x)) # redefine
        self.qtci_tol = qtci_tol # store
        qtci_tol_kernel = self.kernel(qtci_tol) # tol after applying the kernel
        self.qtci_kernel = qtci_kernel # store the kernel
        ######
        self.f = memoize(fk) # memoize
        # initialize the pivot
        ip = 0
        while True: # select pivot
            if qtci_pivot1 is None: # if pivot is None
                r = np.random.random()*(xlim[1]-xlim[0]) + xlim[0] # random point
                qtci_pivot1 = qgrid.coord_to_id([r]) # random place
            r = qgrid.id_to_coord(qtci_pivot1) # to coordinate
            y = self.f(r[0]) # compute
            if np.abs(y)>1e-8: break
            ip+=1
            if ip>max_pivot0_tries: raise # error
            print("Choosing another pivot",r,y)
            qtci_pivot1 = None # set to None
        ###############
        # obtain the interpolator
        ci,qtci_args = get_ci(self.f,nb=nb,qgrid=qgrid,
                qtci_maxm = qtci_maxm,
                qtci_accumulative = qtci_accumulative,
                qtci_tol = qtci_tol_kernel,
                qtci_pivot1 = qtci_pivot1,
                qtci_fullPiv = qtci_fullPiv,
                **kwargs)
        # store the relevant parameters
        self.qtci_args = qtci_args
        self.ci = ci
        self.qtci_maxm = qtci_maxm
        self.qtci_accumulative = qtci_accumulative
        self.qtci_fullPiv = qtci_fullPiv
        self.qtci_pivot1 = qtci_pivot1
        self.nb = nb
        self.xlim = xlim
        self.qgrid = qgrid
        self.errors = ci.pivotError
        self.error = err = ci.pivotError[len(ci.pivotError)-1]
        self.R = nb
        rse,zse = self.get_evaluated()
        self.frac = len(rse)/(2**nb)
#        self.qtt = ci.get_qtt()  # the actual function approximating f
        self.qtt = ci.tt  # the actual function approximating f
    def __call__(self,xs):
        if is_iterable(xs):
            out = []
            for x in xs:
                xi = self.qgrid.coord_to_id([x])
                oi = self.qtt.eval(xi)
                out.append(oi)
        else:
            xi = self.qgrid.coord_to_id([xs])
            out = self.qtt.eval(xi)
        # apply the inverse kernel
        out = self.inv_kernel(out)
        return out
    def integrate(self,axis=None,**kwargs):
        raise
    def get_evaluated(self):
        return get_cache_info(self.f)
    def get_eval_frac(self):
        return (len(self.get_evaluated()[0]))/(2**self.nb)

# this is not used so far

class CI():
    """Class for an enriched CI object"""
    def __init__(self,ci,qgrid=None,nb=1):
        self.xfac_ci = ci # original ci from xfac
        self.qgrid = qgrid # qgrid
        self.nb = nb # number of bits
    def iterate(self):
        """Make an iteration"""
        self.xfac_ci.iterate() 
    def __call__(self,x):
        return eval_ci(self.xfac_ci,self.grid)



def eval_ci(ci,qgrid,x):
    """Evaluate a CI"""
    xi = qgrid.coord_to_id([x]) # to grid
    return ci.tt.eval(xi) # evaluate this point




def random_pivot(p):
    """Return a random pivot with the same shape as the input"""
    return [np.random.randint(0,2) for i in p] # return this one




import numpy as np


def get_ci(f, qgrid=None, nb=1,
        qtci_recursive = False,
        qtci_pivot1 = None, # no initial guess
        qtci_tol = 1e-3, # tolerance of quantics
        qtci_maxm=100, # initial bond dimension
        qtci_accumulative = False,
        qtci_fullPiv = True,
        tol=None,**kwargs):
    """Compute the CI, using an iterative procedure if needed"""
    if tol is not None:  qtci_tol = tol
    args = xfacpy.TensorCI2Param()  # fix the max bond dimension
    args.bondDim = qtci_maxm
    args.useCachedFunction = False # do not use C++ cache
    args.useCachedFunction = qtci_fullPiv # full pivot method
    args.fullPiv = qtci_fullPiv # search the full Pi matrix
    ci = xfacpy.QTensorCI(f1d=f, qgrid=qgrid, args=args)  # construct a tci
    # select the mode
#    print("Here")
#    print(qtci_accumulative)
    if qtci_accumulative: # accumulative mode
        ci,qtci_args = accumulative_train(ci,qtci_tol=qtci_tol,nb=nb,f=f,**kwargs)
    else: # conventional mode
        ci,qtci_args = rook_train(ci,qtci_tol=qtci_tol,nb=nb,
                qtci_pivot1=qtci_pivot1,
                qgrid=qgrid,f=f,**kwargs)
    return ci,qtci_args # return the optimal bond dimension, for next iteration




def accumulative_train(ci,qtci_tol=1e-3,qgrid=None,
        info_qtci=False,
        nb=1,f=None,**kwargs):
    """Train a QTCI using the accumulative mode"""
    if info_qtci:
        print("Accumulative mode")
    ci = xfacpy.to_tci1(ci) # to type one
    batch = 2 # two times
    while True: # infinite loop
        tols = []
        for j in range(batch): # do this once
            ci.iterate(1) # iterate
            toli = ci.pivotError[len(ci.pivotError)-1] # tolerance in evaluated points
            tols.append(toli) # store
        if np.max(tols)<qtci_tol: 
#            print("Accumulative reached error, stopping",np.min(tols))
            break # stop loop
        # other stopping criteria
        npoints = 2**nb
        nev = len(get_cache_info(f)[0])
        if nev/npoints>0.95: 
#            print("Almost all points evaluated, stopping")
            break
    ci = xfacpy.to_tci2(ci) # to type two
    qtci_args = {}
    return ci,qtci_args # return





def rook_train(ci,qtci_tol=1e-3,qgrid=None,
        info_qtci=False,
        qtci_pivot1=None,
        qtci_args = {}, # empty distionary
        nb=1,f=None,
        **kwargs):
    """Rook restart mode"""
    # set the pivot if available
    if info_qtci:
        print("Rook QTCI mode")
    ### Add global pivots ####
    if "qtci_global_pivots" in qtci_args: # if there are global pivots
        ### QTCI global pivots is a list with the pivots ###
        gp = qtci_args["qtci_global_pivots"] # get the list with pivots
        for i in range(len(gp)): # restart the None ones
            if gp[i] is None: # initialize None ones
                gp[i] = random_pivot(qtci_pivot1) # pick a random one
        qtci_args["qtci_global_pivots"] = gp # store again
        if len(gp)>0: # if there are pivots
            ci.addPivotsAllBonds(gp) # global pivots
            ci.makeCanonical() # for stability (?)
#    if "qtci_rook_pivots" in qtci_args: # pivots are given
#        print("#### Adding rook QTCI pivots")
#        qtci_pivots = qtci_args["qtci_rook_pivots"]
#        for i in range(len(qtci_pivots)): # loop
#            ci.addPivotsAt(qtci_pivots[i],i) # add pivots
#        ci.makeCanonical()
    while not ci.isDone(): # iterate until convergence
        ci.iterate()
        err = ci.pivotError[len(ci.pivotError)-1]
        if qtci_tol is not None: # if tol given, break when tol reached
            if err<qtci_tol: # tolerance reached, check a few random points
                err_est = estimate_error(ci,f,nb=nb,qgrid=qgrid)
                if err_est<qtci_tol: # error semms ok, stopping
#                    if info_qtci:
#                        print("QTCI pivot tol reached",err," stopping training")
                    break # stop loop
    # evaluate error #
    evf = len(get_cache_info(f)[0])/(2**nb) # percentage of evaluations
    err = ci.pivotError[len(ci.pivotError)-1] # error
    if info_qtci:
        print("Eval frac = ",evf,"error = ",err)
#    args = dict() # dictionary with the arguments
#    qtci_args["qtci_global_pivots"] = [ci.getPivotsAt(ii) for ii in range(ci.len()-1)]
    return ci,qtci_args



def estimate_error(ci,f,nb=1,qgrid=None):
    """Estimate the error between the function and the tensor train"""
    out = 0.
    ntries = int(2**nb/100 + 3) # number of tries
    out = np.zeros(ntries) # initialize
    outci = np.zeros(ntries) # initialize
    ### choose a few random points ###
    for i in range(ntries): # a few random points
        x = float(np.random.randint(0,2**nb)) # random point
        yci = eval_ci(ci,qgrid,x) # evaluate
        yreal = f(x)
        outci[i] = yci
        out[i] = yreal
    from .qtcidistance import get_distance
    disf = get_distance() # get the distance function
    err1 = disf(outci,out)
    ### now evaluate all the points that have been called ###
    x,y = get_cache_info(f)
    ytt = [eval_ci(ci,qgrid,xi) for xi in x]
    err2 = disf(y,ytt) # error of the evaluated points
    return err1 + err2







from collections.abc import Iterable
def is_iterable(e): return isinstance(e,Iterable)



import pickle
from functools import lru_cache, wraps

def memoize(f):
    @lru_cache(maxsize=None)
    @wraps(f)
    def memoized_func(*args, **kwargs):
        return f(*args, **kwargs)

    memoized_func._cache = {}

    original_func = memoized_func.__wrapped__

    @wraps(memoized_func)
    def wrapper(*args, **kwargs):
        result = memoized_func(*args, **kwargs)
        key = pickle.dumps((args, frozenset(kwargs.items())))
        memoized_func._cache[key] = result
        return result

    wrapper.cache_info = memoized_func.cache_info
    wrapper._cache = memoized_func._cache

    return wrapper

# Define the function to be memoized
def f(x):
    return x**2

# Memoize the function
#fo = memoize(f)

# Recover the list of evaluated points
def get_cache_info(func):
    cache_info = func.cache_info()
    cache_keys = list(func._cache.keys())
    xs = [pickle.loads(key) for key in cache_keys]
    xs = [x[0][0] for x in xs]
    ys = [func(x) for x in xs]
    return xs,ys






