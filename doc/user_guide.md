
# Selfconsistent tight binding calculations

## Creating the Hamiltonian

The Hamiltonian can be created from one of the sublibraries as

```python
from qtcipy.tbscftk import hamiltonians
H = hamiltonians.chain(8) # get the Hamiltonian
H = hamiltonians.honeycomb(5,periodic=True) # get the Hamiltonian
```

where the number denotes the lateral size of the system in log scale.
For example, in the chain "8" leads to a system fo 2^8 sitees, whereas
in the honeycomb lattice "5" leads to a system with $$4*2^(5*2)$$ sites

### Modifying the Hamiltonian
By default the Hamiltonians are uniform in space. To add modulations in the
hopping, you can modify the hoppings as

```python
H.modify_hopping(F)
```

where "F" is a function that is call with input the center of each bond, and returns
a correction to that hopping that it is added to the Hamiltonian.

## Creating the selfconsistent object

The object to perform selfconsistent calculations can be created as

```python
SCF = H.get_SCF_Hubbard(U=3.0) # generate a selfconsistent object
```


where "U" is the value of the Hubbard interaction. By default, the interaction
is taken as zero at the edge of the system to avoid instabilities stemming from edge effects.


## Performing a selfconsistent calculation

The selfconsistent mean field can be computed as 

```python
SCF.solve(use_qtci=True,use_kpm = True,chiral_AF = True)
```

The quantics tensor corss interpolation method is activated by setting
```python
use_qtci=True,use_kpm = True.
```

The conventional exact diagonalization mode is set by using as keyword arguments

```python
use_qtci=False,use_kpm = False
```

For systems with chiral symmetry at half filling, acceleration can be activated with

```python
chiral_AF = True
```

This makes the calculation twice as fast.

### Mixing in a selfconsistent calculation

The mixing between iterations can be controlled with

```python
SCF.solve(mix=0.5)
```

The default value of mix is 0.5 (50% of the old and 50% of the new). A mix = 1.0 would correspond to not updating at all, whereas mix = 0 would correspond to using only the new iteration. If you see that the calculation is very unstable, use mix=0.9, whereas if you want to accelerate the convergence as much as possible use mix=0.1.


Sometimes the quantics tensor cross may have instabilities in one
iteration. For this reason, the mixing is by default on the
mixing_strategy="failsafe" mode, meaning that if it detects that one iteration fluctuates
it goes back close to the previous solution. If you want to use
a conventional mixing, you can activate the option with

```python
SCF.solve(mixing_strategy="plain")
```

## Selfconsistent magnetization

The selfconsistent magnetization can be extracted as 
```python
Mz = SCF.Mz
```

with the positions of sites obtained from
```python
H.R
```


## Computing the non interacting density of states

The density of states in site index can be computed as
```python
(energies,dos) = H.get_dos(i=index,delta=1e-2)
```

where delta is the smearing. The computational cost of this calculation scales as 1/delta.

If you want to compute the DOS in a list of sites indexes = [i1,i2,..,iN]
using the stochastic trace method, you can do it as
```python
(energies,dos) = H.get_dos(i=indexes,delta=1e-2,ntries=5)
```

with "ntries" the number of stochastic vectors tried. In practice,
10 vectors is enough for large enough systems. The computational cost
of this calculation scales linearly with the number of ntries,
but it does not depend on the length of "indexes". This allows
computing local DOS in large regions of a system at a low cost.
The computational cost of this calculation scales as 1/delta.

## Computing the interacting and interacting density of states

The density of states in site "index" can be computed as
```python
(energies,dos) = SCF.get_dos(i=index,delta=1e-2)
```

where delta is the smearing. The computational cost of this calculation scales as 1/delta.

If you want to compute the DOS in a list of sites indexes = [i1,i2,..,iN]
using the stochastic trace method, you can do it as
```python
(energies,dos) = SCF.get_dos(i=indexes,delta=1e-2,ntries=5)
```

with "ntries" the number of stochastic vectors tried. In practice,
10 vectors is enough for large enough systems. The computational cost
of this calculation scales linearly with the number of ntries,
but it does not depend on the length of "indexes". This allows
computing local DOS in large regions of a system at a low cost.
The computational cost of this calculation scales as 1/delta.


## The quantics tensor cross matrix product state in the mean field

The mean field is parametericed by quantics tensor cross at each step.
By default, the QTCI architecture
is optimized until a certain threshold in the SCF is reached,
and afterwards the same architecture is used to finish the 
selfconsistent loop. After a selfconsistent calculation, the architecture
of the optimized QTCI is stored in 

```python
SCF.qtci_kwargs
```

If you want to switch off the dynamical optimization of the QTCI architecture,
and rather provide fixed values at the beginning, you can do so as

```python
SCF.solve(use_qtci=True,use_kpm = True,mix=0.5,use_dynamical_qtci=False,
                        **qtci_kwargs)
```


The architecture of the QTCI is optimized at each selfconsistent step,
as long as the error is above a threshold. If you want to change the threshold
when the architecture is frozen, you can do it with

```python
from qtcipy.tbscf dynamicalqtci 
dynamicalqtci.maxerror_dyn_qtci = 5e-2 # this is the default
```

The architecture is optimized to minimize a certain error function between the mean field and the QTCI. You can choose
the error function to minimize with

```python
from qtcipy import qtcidistance
qtcidistance.default = "mean" # this is the default, alternative is "max" 
```


If you want to restrict the optimization method of the QTCI, you can redefine the list of methods used as

```python
from qtcipy import qtcirecipes
qtcirecipes.methods = ["maxm","accumulative"] # you can remove any if you wish
```

### Enforcing a fraction in the SCF

If you want to enforce that only up to a specific fraction of the sites
are evaluated, you can use the "qtci_macfrac" argument as

```python
SCF.solve(use_qtci=True,use_kpm = True,qtci_maxfrac=0.1)
```

the statement above will enforce that only 10% of the sites are evaluated.
This option is useful if you want to precisely control the time spent in the
calculation, at the expense of accuracy.

## Log with the performance of the mean field
After performing a mean-field calculation you may want to see how the
selfconsisstent error, the performance of the QTCI architecture evolved.
You can access those varaibles from the SCF.log dictionary, in particular

```python
SCF.log["SCF_error"] # error in the SCF steps
SCF.log["QTCI_eval"] # fraction of space evaluated by the QTCI
```

## Loading and saving

If you want to save a selfconsistent calculation for later use, you do it as

```python
SCF.save() # save the SCF object
```

If you want to load a selfconsistent calculation, you can do it as

```python
SCF = SCF.load() # load the SCF object
```

The mean field is stored in 

```python
MF = SCF.MF # mean field
```

If you wish to give a mean-field as initial guess for the selfconsistent loop, you can do it as

```python
SCF.solve(MF=MF) # give MF as initial guess
```

## GPU acceleration

If you want to use GPU acceleration in any part of the code that has it,
you have to give as optional argument to the relevant function

```python
h.get_dos(kpm_cpugpu="GPU")
```

