.. kinisi documentation master file, created by
   sphinx-quickstart on Fri Jan 17 18:56:48 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Uncertainty quantification in diffusion
=======================================

:py:mod:`kinisi` is an open-source package focussed on accurately quantifying the uncertainty in atomic and molecular displacements, and using this to more completely understanding diffusion in materials.

**Bootstrapping**

:py:mod:`kinisi` uses a custom bootstrapping method to evaluate distribution of the mean-squared displacement at a particular timestep length.
This resampling is performed until the distribution is found to be normal, or a user-controlled threshold is reached.

**Diffusion estimation**

A diffusion distribution is evaluated using a generalised least squares approach to modelling the Einstein relation to the data.
This uses a covariance matrix defined based on the bootstrapped uncertainties for each MSD. 
This approach allows an estimate of the true displacement to be found from an infinitely long simulation.
**Note, this methodology is unpublished and results from it should be considered with great caution**.

**Uncertainty propagation**

The :py:class:`uravu.relationship.Relationship` class is leveraged to propagate the uncertainty in the diffusion coefficient using Bayesian inference, allowing the determination of the uncertainty in the activation energy from either an Arrhenius or a `super-Arrhenius relationship`_.
Finally, it is possible to use :py:mod:`uravu` to perform Bayesian model selection between the different temperature dependent relationships.

Brief tutorials showing how :py:mod:`kinisi` may be used in the study of an `VASP Xdatcar`_ file can be found in the `tutorials`_.


Contributors
------------

`Andrew R. McCluskey`_ | `Benjamin J. Morgan`_

.. _Andrew R. McCluskey: https://www.mccluskey.scot
.. _Benjamin J. Morgan: http://analysisandsynthesis.com
.. _super-Arrhenius relationship: https://doi.org/10.1103/PhysRevB.74.134202
.. _VASP Xdatcar: https://www.vasp.at/wiki/index.php/XDATCAR
.. _tutorials: ./tutorials.html


.. toctree::
   :hidden:
   :maxdepth: 2

   installation
   tutorials
   faq
   modules

Searching
=========

:ref:`genindex` | :ref:`modindex` | :ref:`search`
