FAQ
===

- What are the units being used in :py:mod:`kinisi`?

    After :py:mod:`kinisi` reads in a file, the units are modified such that distances are in **angstrom** and time in **picoseconds** 
    (these are the standard units for length and time in `MDAnalysis objects`_, while for VASP we internally convert from femtoseconds 
    to picoseconds on parsing), this means that time objects in the :py:attr:`parser_params` should use in the input unit 
    (i.e. femtoseconds VASP objects or picoseconds for MDAnalysis objects). 
    The :py:attr:`msd` attribute are in units of **squared-angstrom** and the :py:attr:`dt` attribute are in units of **picoseconds**. 
    The diffusion or jump-diffusion coefficient has units of **squared-centimetre per second** and the conductivity is 
    **millisiemens per centimetre** (these were chosen as they are common units for these parameters).

- :py:mod:`kinisi` has given me a really weird value for the diffusion coefficient, how come?

    The best way to check the diffusion coefficient is to compare with the mean-squared displacement as a function of timestep as shown 
    in the `diffusion coefficient tutorial`_. If this "looks" wrong (i.e. really wrong, like a gradient of near 0 when the data clearly 
    has some large gradient) then it is possible that there has been a numerical precision error. Unfortunately, this is related to the 
    nature of the covariance matrix in the analysis (and specifically the inverse of this matrix), the easiest way to fix this is to 
    increase the :code:`rtol` value for the `bootstrap_GLS`_ method, start at some very small value (like :code:`1e-10`) and gradually 
    increase this until you get a reasonable value. **Be aware** that increasing the :code:`rtol` too much may reduce the accuracy 
    of the estimation of the diffusion coefficient. 

- I have been using :py:mod:`kinisi` in my research and would like to cite the package, how should I do this?

    Thanks for using :py:mod:`kinisi`, we are working on a paper that you can cite in the future, but for now please use the 
    following `citation found on Github`_.
    
- How does :py:mod:`kinisi` work?

    Please have a look at our `methodology`_ to understand how :py:mod:`kinisi` works. 


.. _MDAnalysis objects: https://docs.mdanalysis.org/1.1.1/documentation_pages/units.html
.. _diffusion coefficient tutorial: ./vasp_d.html
.. _bootstrap: ./diffusion.html#kinisi.diffusion.Bootstrap.bootstrap_GLS
.. _citation found on Github: https://github.com/bjmorgan/kinisi
.. _methodology: ./methodology.html
