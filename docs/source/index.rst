.. kinisi documentation master file, created by
   sphinx-quickstart on Fri Jan 17 18:56:48 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Uncertainty quantification in diffusion
=======================================

``kinisi`` is an open-source package focussed on accurately quantifying the uncertainty in atomic and molecular displacements, and using this to more completely understanding diffusion in materials.

``kinisi`` has the following functionality:

- a custom bootstraping method to evaluate the mean-squared displacment (and the associated standard deviation) for the motions of atomic or molecular species in a material,
- the `uravu.relationship.Relationship`_ class is leveraged to propagate these uncertainties using Bayesian inference and enable the determination of the uncertainty in the diffusion coefficient and activation energy,
- finally, Bayesian model selection can be used to determine between an Arrhenius and a `super-Arrhenius relationship`_ in the temperature-dependent behaviour.

Brief tutorials showing how ``kinisi`` may be used in the study of an `VASP Xdatcar`_ file and a `MDAnalysis.core.Universe`_ object can be found in the `tutorials`_, along with an example of `kinisi` being used to distinguish between Arrhenius and super-Arrhenius behaviour.


Contributors
------------

- `Andrew R. McCluskey`_
- `Benjamin J. Morgan`_

.. _Andrew R. McCluskey: https://www.armccluskey.com
.. _Benjamin J. Morgan: http://analysisandsynthesis.com
.. _uravu.relationship.Relationship: https://uravu.readthedocs.io/en/latest/relationship.html#uravu.relationship.Relationship
.. _super-Arrhenius relationship: https://doi.org/10.1103/PhysRevB.74.134202
.. _VASP Xdatcar: https://www.vasp.at/wiki/index.php/XDATCAR
.. _MDAnalysis.core.Universe: https://www.mdanalysis.org/docs/documentation_pages/core/universe.html
.. _tutorials: ./tutorials.html


.. toctree::
   :hidden:
   :maxdepth: 2

   installation
   tutorials
   modules

Searching
=========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
