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

- I ran `kinisi` on my system and the diffusion coefficient value was really very unrealistic, i.e. 
    like 1e+128 cm<sup>2</sup>/s. What happened?

    > In short, the way to fix this is to decrease the value of the `cond_max` parameter in the `diffusion` method 
    > until the value is realistic (the best way to check this is with a plot of the model and the data). 
    > Try to find the highest value that gives a realistic result, as going too low can affect accuracy. 
    > This problem is because the covariance matrix used by `kinisi` is an estimate of the true covariance matrix 
    > for the mean-squared displacement. 
    > This estimation can mean that the matrix is ill-conditioned, which is where the ratio between the largest 
    > and smallest eigenvalues of the matrix is very large (you can read more about condition numbers of matrix 
    > on [Wikipedia](https://en.wikipedia.org/wiki/Condition_number#Matrices)). When the matrix is 
    > ill-conditioned, linear algebra can have numerical precision issues. 
    > To solve this, `kinisi` uses the [minimum eigenvalue method](https://doi.org/10.1080/16000870.2019.1696646) to 
    > recondition the matrix, and the `cond_max` parameter is the condition number of the reconditioned covariance 
    > matrix. 
    > So, decreasing the value of `cond_max` will reduce the condition number of the covariance matrix, but if you 
    > decrease it too much, necessary information will be lost from the covariance matrix, leading to a loss of accuracy.

- Running the documentation locally gave me different numbers, how come?

    > `kinisi` aims to be reproducible on a per-environment basis. Therefore, we do not pin versions in 
    > the `pyproject.toml` hence, when you run `pip install '.[docs]'` you might get different package 
    > versions and due to the stochastic nature of the sampling in `kinisi`, this leads to *slightly* 
    > different values in the results. `kinisi` allows a `random_state` to be passed to many methods, 
    > however, this will only ensure reproducibility when the same enviroment is present. Consider using 
    > pinned versions in a conda/mamba environment if you want to enable *true* reproducibility.
    
- How are trajectories unwrapped?

  > When calculating displacements, `kinisi` uses a simple heuristic to unwrap trajectories. 
  > If the displacement between two steps, is greater than half the simulation cell length, `kinisi` wraps that
  > displacement. This scheme assumes that no particle moves more than one cell between steps. Therefore, it requires that
  > enough simulation data is provided to `kinisi`. 
  > This is the reason for not supporting NPT simulations, although this is being investigated.
