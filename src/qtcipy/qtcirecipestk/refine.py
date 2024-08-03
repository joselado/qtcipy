from ..qtcirecipes import get_frac_args

from copy import deepcopy as cp



def refine_qtci_kwargs(v,kw,**kwargs):
    """Do small changes to the QTCI to see if it gets better"""
    return refine_kernel(v,kw,**kwargs)


def refine_kernel(v,kw,**kwargs):
    """Change the kernel, to optimize the QTCI"""
    # input is all the kwargs of QTCI, the variable is in qtci_kernel
    kp = 0.25
    if "qtci_kernel" in kw:
        if "qtci_power_kernel" in kw:
            kp = kw["qtci_power_kernel"] # get the previous kernel
    kps = [kp,0.8*kp,kp/0.8] # try to refine it
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
    return fracmin,kwmin



