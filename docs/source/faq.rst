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

- :py:mod:`kinisi` crashes when I have a very long simulation, how come?

    This is a known issue with :py:mod:`kinisi` that we are working on a solution.
    The reason for this is that for each timestep investigated, :py:mod:`kinisi` will produce an array of displacements, the shape of which is given by :code:`[atom, displacement observation, dimension]`, which are stored in a list.
    Therefore, with a simulation of say 1 000 atoms for 1 000 picoseconds which is analysed with a minimum timestep of 10 picosecond, the first, and largest, item in the list of arrays of will have a size of 3 000 000 floating point numbers (specifically :code:`float64`), each of which is 8 bytes in size.
    This means that if a users has a very long simulation, the size of this list miight end up much larger than the available RAM on the system, causing a crash. 
    The current work around is to limit either the number of atoms in the analysis, using the :py:attr:`sub_sample_atoms` keyword arguement, or the number of timesteps that are read in, using the :py:attr:`sub_sample_traj` keyword arguement.
    For a long simulation, start these both with large numbers and gradually decrease them until the resulting uncertainty in the mean-squared displacements or diffusion coefficient is acceptable. 

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
