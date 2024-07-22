import os
import sys

path = os.path.dirname(os.path.realpath(__file__))

# add xfacpy library
sys.path.append(path+"/pylib/xfac/build/python")

import xfacpy

class Interpolator():
    def __init__(self,f,xlim=[0.,1.],nb=20,
            **kwargs):
        """Initialize the interpolator object"""
        qgrid = xfacpy.QuanticsGrid(a=xlim[0],b=xlim[1], nBit=nb)  # build the quantics grid
        self.f = memoize(f)
        ci,args,opt_qtci_maxm = get_ci(self.f,qgrid=qgrid, **kwargs)
        self.ci = ci
        self.xlim = xlim
        self.qgrid = qgrid
        self.qtci_maxm = args.bondDim # store the bond dimension
        self.errors = ci.pivotError
        self.opt_qtci_maxm = opt_qtci_maxm
        self.R = nb
        self.qtt = ci.get_qtt()  # the actual function approximating f
    def __call__(self,xs):
        if is_iterable(xs):
            out = [self.qtt.eval([x]) for x in xs]
        else:
            out = self.qtt.eval([xs])
        return out
    def integrate(self,axis=None,**kwargs):
        raise
    def get_evaluated(self):
        return get_cache_info(self.f)




def get_ci(f, qgrid=None, 
        qtci_maxm=1, # initial bond dimension
        tol=None,**kwargs):
    """Compute the CI, using an iterative procedure if needed"""
    args = xfacpy.TensorCI2Param()                      # fix the max bond dimension
    args.bondDim = qtci_maxm
    ci = xfacpy.QTensorCI(f1d=f, qgrid=qgrid, args=args)  # construct a tci
    while not ci.isDone(): # iterate until convergence
        ci.iterate()
    return ci,args,qtci_maxm # return the optimal bond dimension
    if tol is not None: # if tolerance is enforced, do it iteratively
        errs = ci.pivotError
        if errs[len(errs)-1]>tol: # do it again
#            args.bondDim = int(int(args.bondDim)*1.5) # increase bond dimension
            m0 = qtci_maxm
            qtci_maxm = int(qtci_maxm) + 1 # redefine
            print("New quantics bond dimension",qtci_maxm)
            print("Original quantics bond dimension",m0)
            return get_ci(f, qgrid=qgrid, 
                    qtci_maxm=qtci_maxm,tol=tol,**kwargs) # do it again
    return ci,args,qtci_maxm # return the optimal bond dimension











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
    evaluated_points = [pickle.loads(key) for key in cache_keys]
    return evaluated_points, cache_info











