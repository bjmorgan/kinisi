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

    > Thanks for using `kinisi`, we recommend that you cite the [arXiv preprint](https://arxiv.org/abs/2305.18244) which 
    > discusses the methodology along with a specific reference to the version of `kinisi` that has been used. 
    
- How does `kinisi` work?

    > Please have a look at our [arXiv preprint](https://arxiv.org/abs/2305.18244) to learn about how `kinisi` works. 
    
- How does `kinisi` compare to the similar functionality in `pymatgen`?

    > The `kinisi` API is based on the `pymatgen` equivalent. 
    > However, `kinisi` offers insight that is not possible with `pymatgen`. 
    > We investigate this in [this Jupyter Notebook](./pymatgen). 

- I got a strange `memory_limit` related error, what's happening?

    > Check out the [specific page](./memory_limit) that we have related to this error. 

- Running the documentation locally gave me different numbers, how come?

    > `kinisi` aims to be reproducible on a per-environment basis. Therefore, we do not pin versions in 
    > the `pyproject.toml` hence, when you run `pip install '.[docs]'` you might get different package 
    > versions and due to the stochastic nature of the sampling in `kinisi`, this leads to *slightly* 
    > different values in the results. `kinisi` allows a `random_state` to be passed to many methods, 
    > however, this will only ensure reproducibility when the same enviroment is present. Consider using 
    > pinned versions in a conda/mamba environment if you want to enable *true* reproducibility.
    
- How are trajectories unwrapped?

  > When calculating displacements `kinisi` uses a simple heuristic to unwrap the trajectory. 
  > If the displacement, between two steps, is greater than half the simulation cell length, `kinisi` wraps that
  > displacement. This scheme assumes that no particle moves more than one cell between steps. Therefore, it requires that
  > enough simulation data is provided to the programme. This process is performed for each dimension of the trajectory,
  > allowing for any orthorhombic cell. However, this heuristic does not support simulation cells that change size or shape.
  > This is the reason for not supporting NPT simulations, althought this is being investigated.