# FAQ

- What are the units being used in `kinisi`?

    > After `kinisi` reads in a file, the units are modified such that distances are in **angstrom** and time 
    > in **picoseconds** (these are the standard units for length and time in 
    > [MDAnalysis objects](https://docs.mdanalysis.org/1.1.1/documentation_pages/units.html), while for VASP 
    > we internally convert from femtoseconds to picoseconds on parsing), this means that time objects in 
    > the `parser_params` should use in the input unit (i.e. femtoseconds VASP objects or picoseconds for MDAnalysis objects). 
    > The `msd` attribute are in units of **squared-angstrom** and the `dt` attribute are in units of **picoseconds**. 
    > The diffusion or jump-diffusion coefficient has units of **squared-centimetre per second** and the conductivity is 
    > **millisiemens per centimetre** (these were chosen as they are common units for these parameters).

- I have been using `kinisi` in my research and would like to cite the package, how should I do this?

    > Thanks for using `kinisi`, we are working on a paper that you can cite in the future, but for now please use the 
    > following [citation found on Github](https://github.com/bjmorgan/kinisi).
    
- How does `kinisi` work?

    > Please have a look at our [methodology](./methodology) to learn about how `kinisi` works. 
    
- How does `kinisi` compare to the similar functionality in `pymatgen`?

    > The `kinisi` API is based on the `pymatgen` equivalent. 
    > However, `kinisi` offers insight that is not possible with `pymatgen`. 
    > We investigate this in [this Jupyter Notebook](./pymatgen). 

- I got a strange `memory_limit` related error, what's happening?

    > Check out the [specific page](./memory_limit) that we have related to this error. 

```{toctree}
---
:hidden: true
---

memory_limit
```
