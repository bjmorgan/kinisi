---
title: 'kinisi: Bayesian analysis of mass transport from molecular dynamics simulations'
tags:
  - Python
  - molecular dynamics
  - diffusion
  - covariance matrix
  - Bayesian regression
authors:
  - name: Andrew R. McCluskey
    orcid: 0000-0003-3381-5911
    affiliation: "1,2"
  - name: Alexander G. Squires
    orcid: 0000-0001-6967-3690
    affiliation: "3"
  - name: Josh Dunn
    orcid: 0000-0003-2659-0806
    affiliation: "1"
  - name: Samuel W. Coles
    orcid: 0000-0001-9722-5676
    affiliation: "4,5"
  - name: Benjamin J. Morgan
    orcid: 0000-0002-3056-8233
    affiliation: "4,5"
affiliations:
 - name: School of Chemistry, University of Bristol, Cantock's Close, Bristol, BS8 1TS, United Kingdom
   index: 1
 - name: European Spallation Source ERIC, Ole Maaløes vej 3, 2200 København N, Denmark
   index: 2
 - name: School of Chemistry, University of Birmingham, Edgbaston, Birmingham, B15 2TT, United Kingdom
   index: 3
 - name: Department of Chemistry, University of Bath, Claverton Down, Bath, BA2 7AY, United Kingdom
   index: 4
 - name: The Faraday Institution, Quad One, Harwell Science and Innovation Campus, Didcot, OX11 0RA, United Kingdom
   index: 5
date: 16 August 2023
bibliography: paper.bib
---

# Summary
`kinisi` is a Python package for estimating transport coefficients&mdash;e.g., self-diffusion coefficients, $D^*$&mdash;and their corresponding uncertainties from molecular dynamics simulation data: it includes an implementation of the approximate Bayesian regression scheme described in @mccluskey_arxiv_2023, wherein the mean-squared displacement (MSD) of mobile atoms is modelled as a multivariate normal distribution that is parametrised from the input simulation data.
`kinisi` uses Markov-chain Monte Carlo [@Goodman2010;@foreman_emcee_2019] to sample this model multivariate normal distribution to give a posterior distribution of linear model ensemble MSDs that are compatible with the observed simulation data.
For each linear ensemble MSD, $\mathbf{x}(t)$, a corresponding estimate of the diffusion coefficient, $\widehat{D}^*$ is given via the Einstein relation,
$$\widehat{D}^* = \frac{1}{6}\frac{\mathrm{d}\,\mathbf{x}(t)}{\mathrm{d}\,t},$$
where $t$ is time.
The posterior distribution of compatible model ensemble MSDs calculated by `kinisi` gives a point estimate for the most probable value of $D^*$, given the observed simulation data, and an estimate of the corresponding uncertainty in $\widehat{D}^*$.
`kinisi` also provides equivalent functionality for estimating collective transport coefficients, i.e., jump-diffusion coefficients and ionic conductivities.

# Statement of Need

Molecular dynamics simulations are widely used to calculate transport coefficients such as self-diffusion coefficients and ionic conductivities [@morgan_relationships_2014;@morgan_mechanistic_2021;@poletayev_defect_2022;@klepis_long_2009;@wang_application_2011;@zelovich_hydroxide_2019;@sendner_interfacial_2009;@shimizu_structural_2015].
Because molecular dynamics simulations are limited in size and timescale, ensemble parameters, such as transport coefficients, that are calculated from simulation trajectories are estimates of the corresponding true (unknown) parameter of interest and suffer from statistical uncertainty.
The statistical properties of any calculated ensemble parameters depend on the details of the input molecular dynamics simulation&mdash;e.g., the choice of interaction potential, system size, and simulation timescale&mdash;and the choice of estimator for the target parameter to be calculated.
An optimal estimation method should minimise the statistical uncertainty in the derived parameter of interest&mdash;the method should be statistically efficient&mdash;and should provide an accurate estimate of this uncertainty, so that calculated values can be used in downstream statistical analyses.

One widely-used approach to estimating the self-diffusion coefficient, $D^*$, from molecular dynamics simulation is to fit a linear model to the observed mean-square displacement, $\mathbf{x}t$ [@allen2017], where the slope of this &ldquo;best fit&rdquo; linear relationship gives a point-estimate for $D^*$ via the corresponding Einstein relation.
The simplest approach to fitting a linear model to observed MSD data is ordinary least squares (OLS).
OLS, however, is statistically inefficient and gives a large uncertainty in the resulting estimate of $D^*$, while also significantly underestimating this uncertainty [@mccluskey_arxiv_2023].
`kinisi` implements the alternative approximate Bayesian regression scheme described in @mccluskey_arxiv_2023, which gives a statistically efficient estimate for $D^*$ and an accurate estimate for the associated uncertainty $\sigma^2[\widehat{D}^*]$.
This approach gives more accurate estimates of $D^*$ from a given size of simulation data (number of atoms and simulation timescale) than ordinary least-squares or weighted least-squares, while the calculated uncertainties allow robust downstream analysis, such as estimating activation energies by fitting an Arrhenius model to $D^*(T)$.

`kinisi` supports simulation output from a variety of common simulation software packages, including VASP [@kresse_ab_1993;@kresse_ab_1994;@kresse_efficiency_1996;@kresse_efficient_1996] and those compatible with Pymatgen [@ong_python_2013], atomic simulation environment (ASE) [@larsen_atomic_2017], and MDAnalysis [@michaud_mdanalysis_2011;@gowers_python_2016]. 
Tutorials and API-level documentation are provided online at [kinisi.rtfd.io](https://kinisi.rtfd.io). 

Full details of the approximate Bayesian regression method implemented in `kinisi` are provided in @mccluskey_arxiv_2023.
A list of publications where `kinisi` has been used in the analysis of simulation data is available at [kinisi.readthedocs.io/en/latest/papers.html](https://kinisi.readthedocs.io/en/latest/papers.html).

# Acknowledgements

The authors thank all of the users of `kinisi` for contributing feedback, suggesting new features and filing bug reports. 
S.W.C., A.G.S. and B.J.M. acknowledge the support of the Faraday Institution (grant numbers FIRG016, FIG017).
B.J.M. acknowledges support from the Royal Society (UF130329 and URF\textbackslash R\textbackslash 191006). 
