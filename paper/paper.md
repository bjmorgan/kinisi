---
title: 'kinisi: Accuracy and uncertainty quantification in diffusion'
tags:
  - Python
  - molecular dynamics
  - diffusion
  - covariance matrix
  - Bayesian regression
authors:
  - name: Andrew R. McCluskey
    orcid: 0000-0003-3381-5911
    affiliation: 1
  - name: Samuel W. Coles
    orcid: 0000-0001-9722-5676
    affiliation: "2,4"
  - name: Alex G. Squires
    orcid: 0000-0001-6967-3690
    affiliation: "3,4"
  - name: Benjamin J. Morgan
    orcid: 0000-0002-3056-8233
    affiliation: "2,4"
affiliations:
 - name: European Spallation Source ERIC, Ole Maaløes vej 3, 2200 København N, DK
   index: 1
 - name: Department of Chemistry, University of Bath, Claverton Down, Bath, BA2 7AY, UK
   index: 2
 - name: The Faraday Institution, Quad One, Harwell Science and Innovation Campus, Didcot, OX11 0RA, UK
   index: 3
 - name: Department of Chemistry, University College London, 20 Gordon Street, London WC1H 0AJ, UK
   index: 4
date: 15 June 2023
bibliography: paper.bib
---

# Summary

Molecular dynamics simulations are a popular tool for the study of diffusion and conductivity in materials. 
However, the computational cost of these simulations limit the "real-world" timescale and system size that can the investigated. 
`kinisi` provides a straightforward interface for the accurate and statistically efficient estimation of the mean-squared displacement and associated uncertainty of atoms and molecules in a material [@mccluskey_arxiv_2023]. 
A model covariance matrix, defined for freely diffusing atoms, is parameterised from the available simulation data and Bayesian regression by Markov chain Monte Carlo [@foreman_emcee_2019] is then used to quantify the posterior probability for the linear Einstein relation.  
In addition to the diffusion coefficient, `kinisi` can also determine the jump-diffusion coefficient and material conductivity directly from a simulation.
Finally, temperature-dependent relationships can be studied with custom `uravu.relationship.Relationship` objects [@mccluskey_uravu_2020]. 

`kinisi` supports simulation output from a variety of common simulation software packages, including VASP [@kresse_ab_1993,@kresse_ab_1994,@kresse_efficiency_1996,@kresse_efficient_1996] and those compatible with Pymatgen [@ong_python_2013], atomic simulation environment [@larsen_atomic_2017], and MDAnalysis [@michaud_mdanalysis_2011,@gowers_python_2016]. 
The availability of this software will offer users the ability to accurately quantify the uncertainties in atomic displacement and introduce this into downstream modelling in a quantitative manner. 
Tutorials and API-level documentation are available online (kinisi.rtfd.io). 

# Statement of Need

Currently `kinisi` is the only software (to the authors' knowledge), that implements the use of the model covariance matrix approach described in [@mccluskey_arxiv_2023], which accurately quantifies the diffusion of atoms in materials. 
While Pymatgen [@ong_python_2013] and others are capable of estimating displacement and uncertainty, the methods applied lack the accuracy and statistical efficiency available from the model covariance approach available in `kinisi`. 

# Acknowledgements

The authors thank all of the users of `kinisi` for contributing feedback, suggesting new features and filing bug reports. 
S.W.C., A.G.S. and B.J.M. acknowledge the support of the Faraday Institution (grant numbers FIRG016, FIG017) 
B.J.M. acknowledges support from the Royal Society (UF130329 and URF\textbackslash R\textbackslash 191006). 

# References
