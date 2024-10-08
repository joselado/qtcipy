
# class to perform SCF hubbard calculations
import numpy as np

class SCF_Hubbard():
    """Class to perform selfconsistency"""
    def __init__(self,H,MF=None,U=0.,**kwargs):
        """Initialize the SCF object"""
        self.H0 = H # store initial Hamiltonian
        self.H = [H.H.copy(),H.H.copy()] # store initial Hamiltonian
        if MF is None:
            dup0 = 0.4*H.AB # initialize
            ddn0 = -dup0.copy() # initialize
            MF = [dup0,ddn0] # store
        self.MF = MF # store the mean-field guess
        self.qtci_kwargs = None # options for QTCI
        set_Hubbard(self,U,**kwargs) # set the Hubbard
#        self.U = U # Hubbard interaction 
        self.Mz = MF[0]*0. # magnetization
        log = dict() # dictionary
        log["SCF_error"] = [] # initialize
        log["QTCI_eval"] = [] # initialize
        log["QTCI_error"] = [] # initialize
        log["SCF_time"] = [] # initialize
        self.log = log # store the log
    def solve(self,**kwargs):
        """Perform the SCF loop"""
        from .hubbard import SCF_Hubbard
        from .dynamicalqtci import initial_qtci_kwargs
        self.qtci_kwargs = initial_qtci_kwargs(self,**kwargs)
        return SCF_Hubbard(self,**kwargs)
    def estimate_time(self,**kwargs):
        from .timeestimator import testimate
        return testimate(self.H0.H,dim=self.H0.dim,**kwargs)
    def get_dos_i(self,w=None,**kwargs):
        """Compute the DOS in site i"""
        from .kpmrho import get_dos_i
        if w is None: w = np.linspace(-5.,5.,1000)
        (es,dup) = get_dos_i(self.H[0].H,x=w,**kwargs)
        (es,ddn) = get_dos_i(self.H[1].H,x=w,**kwargs)
        return es,dup+ddn # return energy and DOS
    def get_dos(self,**kwargs):
        from .ldos import get_dos
        (es,dup) = get_dos(self.H[0],**kwargs)
        (es,ddn) = get_dos(self.H[1],**kwargs)
        return es,dup+ddn # return energy and DOS
    def get_ldos(self,**kwargs):
        from .ldos import get_ldos
        oup = get_ldos(self.H[0],**kwargs)
        odn = get_ldos(self.H[1],**kwargs)
        return (oup+odn)/2. # up plus down
    def get_spin_ldos(self,**kwargs):
        from .ldos import get_ldos
        oup = get_ldos(self.H[0],**kwargs)
        odn = get_ldos(self.H[1],**kwargs)
        return (oup-odn)/2. # up minus down
    def get_spin_dos_i(self,w=None,**kwargs):
        """Compute the DOS in site i"""
        from .kpmrho import get_dos_i
        if w is None: w = np.linspace(-5.,5.,1000)
        (es,dup) = get_dos_i(self.H[0].H,x=w,**kwargs)
        (es,ddn) = get_dos_i(self.H[1].H,x=w,**kwargs)
        return es,dup-ddn # return energy and DOS
    def estimate_qtci_maxm(self,**kwargs):
        from .kpmrho import estimate_qtci_maxm
        from .hamiltonians import matrix2array
        f = matrix2array(self.H0.H)*self.H0.AB
        return estimate_qtci_maxm(self.H0.H,self.H0.R,f,**kwargs)
    def save(self,**kwargs):
        from .saveload import save_SCF
        save_SCF(self,**kwargs)
    def load(self,**kwargs):
        from .saveload import load_SCF
        return load_SCF(**kwargs)
    def get_qtci_kwargs(self):
        """Return a set of good parameters"""
        from .dynamicalqtci import initial_qtci_kwargs
        if len(self.qtci_kwargs)==0: # first iteration
            return initial_qtci_kwargs(self)
        else: return self.qtci_kwargs # return the stored ones
    def get_bandwidth(self):
        """Return the bandwidth of the associated mean-field Hamiltonian"""
        from .kpmrho import estimate_bandwidth
        bup = estimate_bandwidth(self.H[0]) # up
        bdn = estimate_bandwidth(self.H[1]) # down
        return np.max([bup,bdn]) # maximum
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)




def set_Hubbard(self,U,U_profile="envelop"):
    """Process the Hubbard interaction"""
    if type(U)==np.array: # if array
        if len(self.H.H)==U.shape[0]:
            self.U = U # store as array
        else: raise
    elif callable(U):
        Ua = [U(ri) for ri in self.H0.R] # callable
        self.U = Ua
    else: # assume that it is a float
        self.U = U # store as float
    # add the profile
    from .profiles import get_profile
    self.U = self.U*get_profile(self.H0,U_profile)



