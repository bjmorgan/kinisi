.. kinisi documentation master file, created by
   sphinx-quickstart on Fri Jan 17 18:56:48 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Uncertainty quantification in diffusion
=======================================

:py:mod:`kinisi` is an open-source Python package focussed on accurately quantifying the uncertainty in diffusion processes in atomic and molecular systems.

The approach used by :py:mod:`kinisi` ensured an accurate and `statistically efficient`_ estimation of the diffusion coefficient and ordinate offset. 
More information about *how* :py:mod:`kinisi` determines the diffusion coefficient can be found in the detailed `methodology`_.

:py:mod:`kinisi` can handle simulation trajectories from many common molecular dynamics packages, including `VASP`_ and those that can be read by `MDAnalysis`_.
Examples of some of these analyses are shown in the `tutorials`_, also given there is an example of using :py:mod:`kinisi` to investigate the Arrhenius relationship of diffusion as a function of temperature.

.. figure:: _static/example.pdf
  :width: 450
  :align: center
  :alt: An example of the kinisi analysis for the diffusion of lithium in a superionic material. 

  An example of the output from a :py:mod:`kinisi` analysis; showing the determined mean-squared displacements (solid black line), 
  the estimated Einstein diffusion relationship (blue regions representing descreasing credible intervals), 
  and the estimate of the start of the diffusive regime using the maximum of the non-Gaussian parameter (green vertical line).

Contributors
============

`Andrew R. McCluskey`_ | `Benjamin J. Morgan`_

.. _Andrew R. McCluskey: https://www.mccluskey.scot
.. _Benjamin J. Morgan: http://analysisandsynthesis.com
.. _VASP: https://www.vasp.at/wiki/index.php/XDATCAR
.. _MDAnalysis: https://userguide.mdanalysis.org/stable/reading_and_writing.html
.. _tutorials: ./tutorials.html
.. _statistically efficient: https://en.wikipedia.org/wiki/Efficiency_(statistics)
.. _methodology: ./methodology.html


.. toctree::
   :hidden:
   :maxdepth: 2

   installation
   methodology
   tutorials
   faq
   modules
   papers

Searching
=========

:ref:`genindex` | :ref:`modindex` | :ref:`search`

