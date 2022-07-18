Memory overhead
===============

The file parsing functionality of :py:mod:`kinisi` includes a user-definable memory ceiling, which may result in you seeing the following error:

.. code-block:: python

    >>> diff = DiffusionAnalyzer.from_Universe(my_universe, parser_params=p_params)
    MemoryError: The memory limit of this job is 8.0e0 GB but the displacement values will use 9.0e0 GB. 
    Please either increase the memory_limit or decrease the sampling rate (see https://kinisi.readthedocs.io/en/latest/memory_limit.html).

This reason for this ceiling is that for each timestep investigated, :py:mod:`kinisi` will produce an array of displacements, the shape of which is given by :code:`[atom, displacement observation, dimension]`, which are stored in a list.
This means that for a simulation of say 1 000 atoms for 1 000 picoseconds which is analysed with a minimum timestep of 10 picoseconds, the first, and largest, item in the list of arrays will have a size of 3 000 000 floating-point numbers (specifically :code:`float64`), each of which is 8 bytes in size.
Therefore, if a user has a very long simulation, the size of this list might end up much larger than the available RAM on the system, causing a crash. 

The default ceiling is 8 gigabytes, but this can be changed by adding a :py:attr:`memory_limit` item to the :py:attr:`parser_params` dictionary, for example: 

.. code-block:: python

    >>> p_params = {'specie': 'Li',
                    'time_step': 2.0,
                    'step_skip': 50,
                    'min_obs': 50, 
                    'memory_limit': 16.}

Alternatively, for example, if you reach the maximum memory limit of your machine, you can use sub-sampling approaches to reduce the number of observations. 
For example, you can limit either the number of atoms used in the analysis, using the :py:attr:`sub_sample_atoms` keyword argument, or the number of timesteps that are read in, using the :py:attr:`sub_sample_traj` keyword argument.
For a long simulation, start these both with large numbers and gradually decrease them until the resulting uncertainty in the mean-squared displacements or diffusion coefficient is acceptable. 