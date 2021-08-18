---
title: 'kinisi: Uncertainty quantification in diffusion'
tags:
  - Python
  - atomistic simulation
  - diffusion
  - battery materials
  - bootstrap
  - generalised least squares
  - Bayesian
authors:
  - name: Andrew R. McCluskey
    orcid: 0000-0003-3381-5911
    affiliation: "1, 2, 3"
  - name: Samuel W. Coles
    orcid: 0000-0001-9722-5676
    affiliation: 3
  - name: Benjamin J. Morgan
    orcid: 0000-0002-3056-8233
    affiliation: 3
affiliations:
 - name: Data Management and Software Centre, European Spallation Source ERIC, Ole Maaløes vej 3, 2200 København, Denmark
   index: 1
 - name: European Spallation Source ERIC, SE-21100 Lund, Sweden
   index: 2
 - name: Department of Chemistry, University of Bath, Claverton Down, Bath, BA2 7AY, UK
   index: 3
date: 18 July 2021
bibliography: paper.bib
---

# Summary

Molecular dynamics simulation are a popular tool for the study of diffusion in materials. 
However, the computational cost of these simulations limit the ``real-world'' timescale and system size that can the investigated. 
`kinisi` provides a straightforward interface, and compatibility with a variety of common simulation software packages, including VASP [@kresse_ab_1993,kresse_ab_1994,kresse_efficiency_1996,kresse_efficient_1996] and those compatible with MDAnalysis [@michaud_mdanalysis_2011,gowers_python_2016], for the accurate and statistically rigorous calculation of the mean-squared displacement and the associated uncertainty of atoms in a material.
The use of a custom blockstrapping method allows the mean-squared displacement, uncertainty, and covariance may be obtained.
This enables a generalised least squares likelihood sampling approach [@foreman_emcee_2019] to be applied for the estimation of the diffusion coefficient and intercept of the Einstein diffusion relationship, accurately determining the long timescale values and rational uncertainty.
Temperature-dependent relationships can then be studied with custom `uravu.relationship.Relationship` [@mccluskey_uravu_2020] objects.

The availability of this software will offer users the ability to accurately quantify the uncertainties in atomic displacement and introduce this into downstream modelling in a quantitative manner. 

# Statement of Need

Currently `kinisi` is the only software (to the authors' knowledge), that accurately quantifies the diffusion of atoms in materials and includes substantial compatibility with simulation software. 
While `pymatgen` [@ong_python_2013] and others are capable of estimating displacement and uncertainty, the methods applied lack the statistical rigour available in `kinisi`. 
Tutorials, and API level documentation is available online (kinisi.rtfd.io). 

# Acknowledgements

S.W.C. and B.J.M. acknowledge the support of the Faraday Institution through the CATMAT project (grant number FIRG016). B.J.M. acknowledges support from the Royal Society (UF130329 & URF\R\191006).

# References
