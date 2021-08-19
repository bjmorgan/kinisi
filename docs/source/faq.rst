FAQ
===

- What are the units being used in :py:mod:`kinisi`?

    When :py:mod:`kinisi` reads in a file, the units are modified such that distances are in **angstrom** and time in **picoseconds** (these are the standard units for length and time in `MDAnalysis objects`_, while for VASP we internally convert from femtoseconds to picoseconds on parsing), this means that values for the :py:attr:`msd` attribute are in units of **squared-angstrom** and the :py:attr:`dt` attribute are in units of **picoseconds**. However, the diffusion or jump-diffusion coefficient have units of **squared-centimetre per second** and the conductivity is **millisiemens per centimetre** (these were chosen as they are common units for these parameters). 

- I have been using :py:mod:`kinisi` in my research and would like to cite the package, how should I do this?

    Thanks for using :py:mod:`kinisi`, we are working on a paper that you can cite in the future, but for now please use the following citation: “McCluskey, A. R., & Morgan, B. J. (2021). kinisi (Version 0.1.0) [Computer software]. https://github.com/bjmorgan/kinisi”
    
- How does :py:mod:`kinisi` work?

    :py:mod:`kinisi` uses a custom bootstrap resampling approach to determine accurate uncertainties and covariances in MSD and leverages likelihood sampling of a covariant multidimensional Gaussian distribution to determine the diffusion coefficient and associated uncertainty. A paper discussing this in detail is in preparation.


.. _MDAnalysis objects: https://docs.mdanalysis.org/1.1.1/documentation_pages/units.html
