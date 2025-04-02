import numpy as np
from numba import jit

class Hamiltonian():
    """Class for a Hamiltonian"""
    def __init__(self,dim=1,H=None,R=None,AB=None):
        """Class for a Hamiltonian"""
        self.dim = 1 # dimensionality
        self.H = H # matrix
        self.R = R - np.mean(R,axis=0) # center in 0
        self.AB = AB # sublattice
        self.norb = 1 # single orbital
    def get_SCF_Hubbard(self,**kwargs):
        """Return the SCF object"""
        from .scf import SCF_Hubbard
        return SCF_Hubbard(self,**kwargs)
    def modify_hopping(self,*args,**kwargs):
        return modify_hopping(self,*args,**kwargs)
    def add_onsite(self,*args,**kwargs):
        return add_onsite(self,*args)
    def get_density_i(self,**kwargs):
        from .hubbard import get_density_i
        return get_density_i(self.H,**kwargs)
    def get_density(self,**kwargs):
        from .hubbard import get_den
        return get_den(self.H,**kwargs)
    def get_dos_i(self,w=None,**kwargs):
        from .kpmrho import get_dos_i
        if w is None: w = np.linspace(-5.,5.,1000)
        return get_dos_i(self.H,x=w,**kwargs)
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)
    def get_ldos(self,**kwargs):
        """Return the LDOS at a certain energy"""
        from .ldos import get_ldos
        return get_ldos(self.H,**kwargs)
    def index_around_r(self,**kwargs):
        return index_around_r(self.R,**kwargs)
    def get_dos(self,**kwargs):
        from .ldos import get_dos
        return get_dos(self.H,**kwargs)
    def get_moire(self):
        """Extract the approximate moire pattern of the system"""
        out = matrix2array(self.H)
        return out



def matrix2array(H):
    """Convert a matrix into an array"""
    from scipy.sparse import coo_matrix
    mi = coo_matrix(H)
    row = mi.row
    col = mi.col
    data = mi.data.real
    out = np.zeros(mi.shape[0]) # output
    @jit(nopython=True)
    def f(data,row,col,out):
        for i in range(len(data)): # loop over data
            r,c = row[i],col[i]
            out[r] = out[r] + data[i]
        return out
    out = f(data,row,col,out)
    return out




def chain(L):
    """Hamiltonian of a chain"""
    # define a first neighbor tight binding model
    n = 2**L # number of sites
    rows,cols = np.array(np.arange(0,n-1)),np.array(np.arange(1,n)) # indexes
    data = np.zeros(n-1,dtype=np.float32) # hopping 1 for all
    data[:] = 1.0 # initialize
    from scipy.sparse import csc_matrix
    h0 = csc_matrix((data,(rows,cols)),shape=(n,n),dtype=np.float32) # create single particle hopping
    h0 = h0 + h0.T # add the transpose
    r = np.zeros((n,2),dtype=np.float32) # initialize
    r[:,0] = np.arange(n) # locations
    AB = np.array(np.arange(0,n))%2 # parity
    AB = AB*2 - 1. # +- 1
    H = Hamiltonian(dim=1,H=h0,R=r,AB=AB) # create Hamiltonian
    return H # return the Hamiltonian



def square(L,periodic=False):
    """Hamiltonian of a square lattice"""
    # define a first neighbor tight binding model
    n = 2**L # number of sites
    row,col,data = hopping_square(n,periodic=periodic) # return the hoppings
    from scipy.sparse import csc_matrix
    h0 = csc_matrix((data,(row,col)),shape=(n**2,n**2),dtype=np.float32) # create single particle hopping
    r,AB = position_square(n) # return the positions for square lattice
    H = Hamiltonian(dim=1,H=h0,R=r,AB=AB) # create Hamiltonian
    return H # return the Hamiltonian



# Hopping for the square lattice
@jit(nopython=True)
def hopping_square(N,periodic=False):
    """Return hopping for the square lattice"""
    count = 0
    row = np.zeros(4*N**2,dtype=np.int_) # index
    col = np.zeros(4*N**2,dtype=np.int_) # index
    data = np.zeros(4*N**2,dtype=np.float32) # index
    for i1 in range(N):
      for j1 in range(N):
          ind1 = i1*N + j1 # index for the first site 
          for di,dj in [[-1,0],[1,0],[0,1],[0,-1]]:
                i2 = i1 + di
                j2 = j1 + dj
                if periodic: # periodic boundary conditions
                    i2 = i2%N
                    j2 = j2%N
                ind2 = i2*N + j2 # index for the second side
                if 0<=i2<N and 0<=j2<N: # no overflow
                    row[count] = ind1 # store index
                    col[count] = ind2 # store index
                    data[count] = 1.0 # store hopping
                    count +=1 # increase counter
    row = row[0:count] # only those stored
    col = col[0:count] # only those stored
    data = data[0:count] # only those stored
    return row,col,data # return rows,cols and data



# position for the square lattice
@jit(nopython=True)
def position_square(N):
    """Return hopping for the square lattice"""
    count = 0
    r = np.zeros((N**2,2),dtype=np.float32) # index
    AB = np.zeros(N**2,dtype=np.float32) # index
    for i1 in range(N): # first loop 
      for j1 in range(N): # second loop 
          r[count,0] = float(i1)
          r[count,1] = float(j1)
          AB[count] =  (-1)**i1*(-1)**j1 # sublattice
          count += 1 # increase counter
    return r,AB











def modify_hopping(self,f,use_dr=False,**kwargs):
    """Modify the hoppings of a Hamiltonian"""
    from scipy.sparse import coo_matrix
    m = self.H.copy() # get the matrix
    R = self.R # positions
    mo = coo_matrix(m) # to coo_matrix
    for i in range(len(mo.data)): # loop over non-zero elements
        ii = mo.row[i] # get this row
        jj = mo.col[i] # get this column
        d = mo.data[i] # get this data
        r = (R[ii] + R[jj])/2. # average location
        if use_dr:
            dr = R[ii] - R[jj] # distance
            d1 = d + f(r,dr) # add a contribution
        else:
            d1 = d + f(r) # add a contribution
        m[ii,jj] = d1 # store the new hopping
    self.H = m # store the new Hamiltonian



def add_onsite(self,f):
    """Add an onsite energy tot he Hamiltonian"""
    n = self.H.shape[0]
    rows,cols = np.array(range(n)),np.array(range(n)) # indexes
    data = np.array([f(ri) for ri in self.R],dtype=np.float32) 
    from scipy.sparse import csc_matrix
    h0 = csc_matrix((data,(rows,cols)),shape=(n,n),dtype=np.float32)
    self.H = self.H + h0 # add the onsite energy




#########################
### Honeycomb lattice ###
#########################


def honeycomb(L,periodic=False):
    """Hamiltonian of a honeycomb lattice"""
    # define a first neighbor tight binding model
    n = 2**L # number of unit cells
    row,col,data = hopping_honeycomb(n,periodic=periodic) # return the hoppings
    from scipy.sparse import csc_matrix
    h0 = csc_matrix((data,(row,col)),shape=(4*n**2,4*n**2),dtype=np.float32) # create single particle hopping
    r,AB = position_honeycomb(n) # return the positions for square lattice
    H = Hamiltonian(dim=1,H=h0,R=r,AB=AB) # create Hamiltonian
    return H # return the Hamiltonian



# Hopping for the square lattice
@jit(nopython=True)
def hopping_honeycomb(N,periodic=False):
    """Return hopping for the honeycomb lattice"""
    count = 0 # counter
    row = np.zeros(4*3*N**2,dtype=np.int_) # index
    col = np.zeros(4*3*N**2,dtype=np.int_) # index
    data = np.zeros(4*3*N**2,dtype=np.float32) # index
    for i1 in range(N): # loop over x unit cell index
      for j1 in range(N): # loop over y unit cell index
          ind1 = i1*N + j1 # index for the first UC 
          for di,dj in [[0,0],[-1,0],[1,0],[0,1],[0,-1]]: # loop over neigh unit cells
                i2 = i1 + di
                j2 = j1 + dj
                if periodic: # periodic boundary conditions
                    i2 = i2%N
                    j2 = j2%N
                ind2 = i2*N + j2 # index for the second UC
                if 0<=i2<N and 0<=j2<N: # no overflow of unit cells
                    # now loop over the orbitals in the unit cell
                    # do it case by case
                    ## in the (0,0) direction
                    if di==0 and dj==0:
                        pairs = [[0,1],[1,0],[1,2],[2,1],[3,2],[2,3]] # onsite
                    if di==1 and dj==0:  pairs = [[3,0]]
                    if di==-1 and dj==0:  pairs = [[0,3]]
                    if di==0 and dj==1:  pairs = [[1,0],[2,3]]
                    if di==0 and dj==-1:  pairs = [[0,1],[3,2]]
                    for o1,o2 in pairs: # store the pairs
                        row[count] = 4*ind1 + o1 # store index
                        col[count] = 4*ind2 + o2 # store index
                        data[count] = 1.0 # store hopping
                        count +=1 # increase counter
    row = row[0:count] # only those stored
    col = col[0:count] # only those stored
    data = data[0:count] # only those stored
    return row,col,data # return rows,cols and data


# position for the square lattice
@jit(nopython=True)
def position_honeycomb(N):
    """Return hopping for the square lattice"""
    count = 0
    r = np.zeros((4*N**2,2),dtype=np.float32) # index
    AB = np.zeros(4*N**2,dtype=np.float32) # index
    a1 = np.array([3.,0.]) # shift in x
    a2 = np.array([0.,np.sqrt(3.)]) # shift in y
    for i1 in range(N): # first loop 
      for j1 in range(N): # second loop 
          dr = float(i1)*a1 + float(j1)*a2 # shift
          r[4*count] = dr # shift
          r[4*count+1] = dr + a2/2. + np.array([.5,.0]) # shift
          r[4*count+2] = dr + a2/2. + np.array([1.5,.0]) # shift
          r[4*count+3] = dr + np.array([2.,.0]) # shift
          for io in range(4):
              AB[4*count+io] = (-1)**io # sublattice
          count += 1 # increase counter
    return r,AB

def index_around_r(R0,r=[0.,0.],dr=1.0):
    R = R0.copy() # make a copy
    R[:,0] -= r[0] # center
    R[:,1] -= r[1] # center
    R2 = np.sum(R*R,axis=1) # square distance
    inds = np.where(R2 < dr*dr)[0]
    return inds # return indexes





