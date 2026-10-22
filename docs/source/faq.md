# FAQ

- I have been using `kinisi` in my research and would like to cite the package, how should I do this?

    > Thanks for using `kinisi`.
    > `kinisi` supports [the `duecredit` framework](https://github.com/duecredit/duecredit) for 
    > generating the relevant citations to go with your analysis. 
    > If you cannot use this framework, please cite the methodological 
    > [Journal of Chemical Theory and Computation paper](https://doi.org/10.1021/acs.jctc.4c01249), the 
    > [JOSS software paper](https://doi.org/10.21105/joss.05984), and specifically refer to the version of 
    > `kinisi` that has been used. If you have used the centre of mass functionality, please also cite [our
    > publication on that](https://doi.org/10.1063/5.0260928).
    
- How does `kinisi` work?

    > Please have a look at our [methodology paper](https://doi.org/10.1021/acs.jctc.4c01249) to learn about how `kinisi` works. 

- Running the documentation locally gave me different numbers, how come?

    > `kinisi` aims to be reproducible on a per-environment basis. Therefore, we do not pin versions in 
    > the `pyproject.toml` hence, when you run `pip install '.[docs]'` you might get different package 
    > versions and due to the stochastic nature of the sampling in `kinisi`, this leads to *slightly* 
    > different values in the results. `kinisi` allows a `random_state` to be passed to many methods, 
    > however, this will only ensure reproducibility when the same environment is present. Consider using 
    > pinned versions in a conda/mamba environment if you want to enable *true* reproducibility.
    
- How are trajectories unwrapped?

    > When calculating displacements, `kinisi` aims to find the minimum displacement between trajectory steps.
    > This can be done for orthorhombic cells with a simple heuristic: if the displacement is greater than one half
    > the simulation cell length, `kinisi` wraps that displacement.
    > For the case of non-orthorhombic simulation cells, the displacements to all periodic images are calculated 
    > and the minimum used. This scheme assumes that no particle moves more than one cell between steps.
    > Therefore, it requires that enough simulation data is provided to `kinisi`,
    > in other words, that there is a small enough time skip between trajectory steps. 
    > A small enhancement is made to these methods, in order to account for changes in cell volume during 
    > NPT simulations. This enhancement is the 
    > [TOR scheme, developed by Bullerjahn and co-workers](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00308).
    > If you use `kinisi` for an NPT simulation, please cite their work.

- I don't want to sample the MSD at all possible time intervals, how can I achieve this?

    > This can be achieved through the use of a custom time interval input. 
    > You can see how to do this in the [MDAnalysis comparison notebook](./mdanalysis). 

- My analysis is giving very weird numbers for the diffusion coefficient, and my trendline appears very wrong, what's happening?

    > You may be encountering the joys of a covariance matrix with a high condition number. 
    > This leads to issues with numerical precision in some of the operations kinisi performs. 
    > We have a short [documentation page](./condition_number) that describes this issue and provides a potential mitigation strategy. 