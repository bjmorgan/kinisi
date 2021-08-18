FAQ
===

- What are the units being used in :py:mod:`kinisi`?

    When :py:mod:`kinisi` reads in a file, the units are modified such that distances are in angstrom and time in picoseconds, this means that values for the :py:attr:`msd` attribute are in units of squared-angstrom and the :py:attr:`dt` attribute are in units of picoseconds. However, the diffusion or jump-diffusion coefficient have units of centimetre per second and the conductivity is millisiemens per centimetre (these were chosen as they are common units for these parameters). 

- I have been using :py:mod:`kinisi` in my research and would like to cite the package, how should I do this?

    Thanks for using :py:mod:`kinisi`, we are working on a paper that you can cite in the future, but for now please use the following citation: “A. R. McCluskey & B. J. Morgan. kinisi: Uncertainty quantification in diffusion. https://github.com/bjmorgan/kinisi.”
    
- How does :py:mod:`kinisi` work?

    :py:mod:`kinisi` uses a custom bootstrap resampling approach to determine accurate uncertainties and covariances in MSD and leverages likelihood sampling of a covariant multidimensional Gaussian distribution to determine the diffusion coefficient and associated uncertainty. A paper discussing this in detail is in preparation.


