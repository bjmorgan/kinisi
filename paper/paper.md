---
title: 'kinisi: Uncertainty quantification in diffusion'
tags:
  - Python
  - atomistic simulation
  - diffusion
  - battery materials
  - bootstrap
  - Bayesian
authors:
  - name: Andrew R. McCluskey
    orcid: 0000-0003-3381-5911
    affiliation: "1, 2" 
  - name: Benjamin J. Morgan
    orcid: 0000-0002-3056-8233
    affiliation: 1
affiliations:
 - name: Diamond Light Source, Rutherford Appleton Laboratory, Harwell Science and Innovation Campus, Didcot, OX11 0DE, UK
   index: 1
 - name: Department of Chemistry, University of Bath, Claverton Down, Bath, BA2 7AY, UK
   index: 2
date: 04 May 2020
bibliography: paper.bib
---

# Summary

Molecular dynamics simulation are a popular tool for the study of diffusion in materials. 
However, the computational cost of these simulations limit the ``real-world'' timescale and system size that can the investigated. 
`kinisi` provides a straightforward interface, and compatibility with a variety of common simulation software packages, including VASP [@kresse_ab_1993,kresse_ab_1994,kresse_efficiency_1996,kresse_efficient_1996] and those compatible with MDAnalysis [@michaud_mdanalysis_2011,gowers_python_2016], for the accurate and statistically rigorous calculation of the mean-squared displacement and the associated uncertainty of atoms in a material.
The use of a custom blockstrapping method allows the mean-squared displacement and uncertainty may be obtained, and the time- and temperature-dependent relationships can then be studied with custom `uravu.relationship.Relationship`[@mccluskey_uravu_2020] objects.

The availability of this software will offer users the ability to accurately quantify the uncertainties in atomic displacement and introduce this into downstream modelling in a quantitative manner. 

# Statement of Need

Currently `kinisi` is the only software (to the authors' knowledge), that accurately quantifies the diffusion of atoms in materials and includes substantial compatibility with simulation software. 
the `pymatgen` [@ong_python_2013] and others are capable of estimating displacement and uncertainty, however the method applied is not statistically rigorous. 
Tutorials, and API level documentation is available online (kinisi.rtfd.io). 

# Acknowledgements

This work is supported by the Ada Lovelace Centre – a joint initiative between the Science and Technology Facilities Council (as part of UK Research and Innovation), Diamond Light Source, and the UK Atomic Energy Authority.

# References