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
    affiliation: "1, 2"
  - name: Benjamin J. Morgan
    orcid: 0000-0002-3056-8233
    affiliation: 2
affiliations:
 - name: European Spallation Source ERIC, SE-21100 Lund, Sweden
   index: 1
 - name: Department of Chemistry, University of Bath, Claverton Down, Bath, BA2 7AY, UK
   index: 2
date: 18 July 2021
bibliography: paper.bib
---

# Summary

Molecular dynamics simulation are a popular tool for the study of diffusion and conductivity in materials. 
However, the computational cost of these simulations limit the ``real-world'' timescale and system size that can the investigated. 
`kinisi` provides a straightforward interfaces for the accurate and statistically rigorous estimation of the mean-squared displacement and associated uncertainty of atoms and molecules in a material. 
The use of a custom bootstrapping method enables the determination of the distribution of mean-squared displacement and covariance matrix. 
This enables a generalised least squares likelihood sampling approach [@foreman_emcee_2019] to be applied for the estimation of the diffusion coefficient and intercept of the Einstein diffusion relation, accurately determining long timescale values and rational uncertainties. 
In addition to the diffusion coefficient, `kinisi` can also determine the jump diffusion coefficient and material conductivity directly from a simulation.
Finally, temperature-dependent relationships can be studied with custom `uravu.relationship.Relationship` objects [@mccluskey_uravu_2020]. 

`kinisi` supports simulation output from a variety of common simulation software packages, including VASP [@kresse_ab_1993,kresse_ab_1994,kresse_efficiency_1996,kresse_efficient_1996] and those compatible with MDAnalysis [@michaud_mdanalysis_2011,gowers_python_2016]. 
The availability of this software will offer users the ability to accurately quantify the uncertainties in atomic displacement and introduce this into downstream modelling in a quantitative manner. 
Tutorials, and API-level documentation is available online (kinisi.rtfd.io). 

# Statement of Need

Currently `kinisi` is the only software (to the authors' knowledge), that accurately quantifies the diffusion of atoms in materials and includes substantial compatibility with simulation software. 
While `pymatgen` [@ong_python_2013] and others are capable of estimating displacement and uncertainty, the methods applied lack the statistical rigour available from the bootstrap-GLS approach in `kinisi`. 

# Acknowledgements

B.J.M. acknowledges the support of the Faraday Institution through the CATMAT project (grant number FIRG016) and the Royal Society (UF130329 & URF\R\191006).

# References
