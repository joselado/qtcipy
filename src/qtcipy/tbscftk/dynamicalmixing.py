import numpy as np

from ..qtcidistance import get_distance 



def dynamical_mixing_hubbard(SCF,dnew,dold,mix=0.5,
        mixing_strategy="failsafe",**kwargs):
    """Do a dynamical update of the mean-field, depending on the
    error of the last iteration"""
    def mix_den(mix):
        """Wrapper for the mixing"""
        dup,ddn = dnew[0],dnew[1]
        dup_old,ddn_old = dold[0],dold[1]
        dup_out = mix*dup_old + (1.-mix)*dup # update
        ddn_out = mix*ddn_old + (1.-mix)*ddn # update
        return (dup_out,ddn_out) # return the output
    ### start with the mixing strategies
    # conventional one
    if mixing_strategy=="plain": 
        return mix_den(mix)
    # robust to fluctations
    elif mixing_strategy=="failsafe": 
        errors = SCF.log["SCF_error"] # get all the errors
        normal_mix = True # do not use the usual mixing
        if len(errors)>2: # more than 2
            if errors[-1]<errors[-2]: # if the error decreased, mix as usual
                return mix_den(mix)
            else: # use the backup mixing
                print("Error has increased, use conservative mixing")
                mix_c = 1e-2 # highly conservative mixing
                return mix_den(1.-mix_c) # normal mixing
        else: # for first iteations just mix normally 
            return mix_den(mix)
    else: raise # not implemented



