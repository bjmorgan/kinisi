FAQ
===

- What are the units being used in :py:mod:`kinisi`?

    When :py:mod:`kinisi` reads in a file, the units are modified such that distances are in **angstrom** and time in **picoseconds** 
    (these are the standard units for length and time in `MDAnalysis objects`_, while for VASP we internally convert from femtoseconds 
    to picoseconds on parsing), this means that time objects in the :py:attr:`parser_params` should use in the input unit 
    (i.e. femtoseconds VASP objects or picoseconds for MDAnalysis objects). 
    The :py:attr:`msd` attribute are in units of **squared-angstrom** and the :py:attr:`dt` attribute are in units of **picoseconds**. 
    The diffusion or jump-diffusion coefficient has units of **squared-centimetre per second** and the conductivity is 
    **millisiemens per centimetre** (these were chosen as they are common units for these parameters). 

- I have been using :py:mod:`kinisi` in my research and would like to cite the package, how should I do this?

    Thanks for using :py:mod:`kinisi`, we are working on a paper that you can cite in the future, but for now please use the 
    following `citation found on Github`_.
    
- How does :py:mod:`kinisi` work?

    Please have a look at our `methodology`_ to understand how :py:mod:`kinisi` works. 


.. _MDAnalysis objects: https://docs.mdanalysis.org/1.1.1/documentation_pages/units.html
.. _citation found on Github: https://github.com/bjmorgan/kinisi
.. _methodology: ./methodology.html