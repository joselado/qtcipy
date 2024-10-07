#  qtcipy
Python library to perform quantics tensor cross interpolation. The library
is built on top of [QuanticsTCI.jl](https://github.com/tensor4all/QuanticsTCI.jl?tab=readme-ov-file) and [xfacpy](https://github.com/tensor4all/xfac)

# Examples

The folder examples contains several use cases of the library

Some of the examples use the library for electronic structure [pyqula](https://github.com/joselado/pyqula)

## Mean field with quantics tensor cross interpolation

Below you can see a minimal example of a mean field calculation using
the quantics tensor cross interpolation algorithm combined with
the kernel polynomial method for an interacting Hubbard Hamiltonian.

### Moire in an interacting one-dimensional model

```python
from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 12 # exponential length, leads to 2**L sites
H = hamiltonians.chain(L) # get the Hamiltonian

def f(r):
    """Modulation of the hopping"""
    omega = np.pi*2.*np.sqrt(2.)/(2**L/10) # frequency of the modulation
    return 1. + 0.2*np.cos(omega*r[0]) # return correction to the hopping
H.modify_hopping(f) # modify the hopping 

SCF = H.get_SCF_Hubbard(U=3.0) # generate a selfconsistent object

SCF.solve(use_qtci=True,use_kpm=True)
Mz = SCF.Mz # selfconsistent magnetization
```

The moire Hamiltonian and resulting selfconsistent electronic order are
![Alt text](images/1dmoire.png?raw=true "Interaction-driven order with KPQTC")



### Moire domain wall in an interacting one-dimensional model

We will now see how an interface between two moire patterns can be computed

```python
from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 14 # exponential length, leads to 2**L sites
H = hamiltonians.chain(L) # get the Hamiltonian

def f(r):
    """Modulation of the hopping"""
    omega1 = np.pi*2.*np.sqrt(2.)/(2**L/10) # left frequency
    omega2 = np.sqrt(3)*omega1 # right frequency
    if r[0]<0.: return 0.2*np.sin(omega1*r[0]) # return correction to the hopping
    else: return 0.1*np.sin(omega2*r[0]) # return correction to the hopping

H.modify_hopping(f) # modify the hopping 

SCF = H.get_SCF_Hubbard(U=3.0) # generate a selfconsistent object

SCF.solve(use_qtci=True,use_kpm=True)
Mz = SCF.Mz # selfconsistent magnetization
```

The moire Hamiltonian and resulting selfconsistent electronic order are
![Alt text](images/1dinterface.png?raw=true "Interaction-driven order with KPQTC in a moire interface")



### Non-uniform strained moire in an interacting one-dimensional model

We will now compute a moire pattern with non-uniform strain

```python
from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 14 # exponential length, leads to 2**L sites
H = hamiltonians.chain(L) # get the Hamiltonian

def f(r):
    """Modulation of the hopping"""
    length = 2**L # total length
    omega0 = np.pi*2.*np.sqrt(2.)/(length/20) # base frequency
    omega = omega0*(1 + r[0]/length) # position-dependent frequency
    return 0.2*np.sin(omega*r[0]) # return correction to the hopping


H.modify_hopping(f) # modify the hopping 

SCF = H.get_SCF_Hubbard(U=3.0) # generate a selfconsistent object

SCF.solve(use_qtci=True,use_kpm=True)
Mz = SCF.Mz # selfconsistent magnetization
```

The moire Hamiltonian and resulting selfconsistent electronic order are
![Alt text](images/1dstrainedmoire.png?raw=true "Interaction-driven order with KPQTC in a moire interface")


# Documentation

Documentation about the kernel polynomial tensor cross interpolation algorithm for interacting tight binding models can be found [here](https://github.com/joselado/qtcipy/blob/main/doc/user_guide.md)

# Installation

You need to have Julia installed in your computer.
Afterwards, execute "python install.py" in the current directory.
Julia needs to be in your PATH, as the code will use the output of "which julia"

If you want to install the C++ version, use
```bash
python install_cpp.py
```

For the kernel polynomial quantics tensor cross algorithm for interacting tight binding models, the library pyqula has to be installed
```bash
pip install pyqula
```
