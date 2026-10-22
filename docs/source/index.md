# Accurate Estimation of Diffusion Coefficients and their Uncertainties

[![JCTC Status](https://img.shields.io/badge/JCTC-10.1021/acs.jctc.4c01249-blue.svg)](https://doi.org/10.1021/acs.jctc.4c01249)
[![JOSS Status](https://joss.theoj.org/papers/1ae102ffb6b3c63b04c002976440815d/status.svg)](https://joss.theoj.org/papers/1ae102ffb6b3c63b04c002976440815d)
[![PyPI version](https://badge.fury.io/py/kinisi.svg)](https://badge.fury.io/py/kinisi)

`kinisi` is an open-source Python package that can accurately estimate diffusion processes in atomic and molecular systems and determine an accurate estimate of the uncertainty in these processes.

This is achieved by modelling the diffusion process as a [multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution), based on that for a random walker. 
This gives an accurate estimate of the diffusion coefficient and its uncertainty.
More information about the approach `kinisi` uses can be found in the [methodology article](https://doi.org/10.1021/acs.jctc.4c01249), which is also introduced in this [poster](./_static/poster.pdf).

```{image} ./_static/example.png
  :width: 450
  :align: center
  :alt: An example of the kinisi analysis for the diffusion of lithium in a superionic material. 
```
<center>
<small>
An example of the output from <code>kinisi</code>; showing the determined mean-squared displacements (solid line),<br>
the estimated Einstein diffusion relationship (blue regions representing decreasing credible intervals).
</small>
</center>
<br>

`kinisi` is built using the `scipp` library; we recommend [familiarising yourself](https://scipp.github.io/getting-started/index.html) with some of the `scipp` data structures if you want to start doing more complex things with `kinisi`. 
`kinisi` can handle simulation trajectories from many common molecular dynamics packages; if your trajectory can be read by [MDAnalysis](https://userguide.mdanalysis.org/stable/reading_and_writing.html), [ASE](https://wiki.fysik.dtu.dk/ase/ase/io/io.html), or [pymatgen](https://pymatgen.org), then you can use `kinisi`.
The Markov chain Monte Carlo algorithm uses [the `emcee` package](https://emcee.readthedocs.io/en/stable/).
Examples of some of the analyses `kinisi` can perform are shown in the [notebooks](./notebooks).
<center>
<a href="https://scipy.org"><img src="https://raw.githubusercontent.com/scipy/scipy.org/refs/heads/main/static/images/logo.svg" width="10%"></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://numpy.org"><img src="https://raw.githubusercontent.com/numpy/numpy/87e208c900ef0ac46bb3ea861c811f94f478698f/branding/logo/primary/numpylogo.png" width="25%"></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a href="http://scipp.github.io"><img src="https://scipp.github.io/_static/logo-2022.svg" width="25%"></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://emcee.readthedocs.io/"><img src="https://raw.githubusercontent.com/dfm/emcee/c75406b1a6bf197f71f9e56068e9ea57af08be54/docs/_static/logo-sidebar.png" width="20%"></a>&nbsp;&nbsp;&nbsp;&nbsp;
<br>
<a href="http://mdanalysis.org"><img src="https://www.mdanalysis.org/public/mdanalysis-logo_square.png" width="10%"></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a href="http://pymatgen.org"><img src="https://raw.githubusercontent.com/materialsproject/pymatgen/master/docs/assets/pymatgen.svg" width="20%"></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a href="http://ase-lib.org"><img src="https://ase-lib.org/_static/ase256.png" width="10%"></a>&nbsp;&nbsp;&nbsp;&nbsp;
</center>

## [Contributors](https://github.com/kinisi-dev/kinisi/graphs/contributors)

```{toctree}
---
hidden: True
---

installation
methodology
notebooks
faq
papers
modules
```
