__version__ = '2.1.0'

from .due import BibTeX, due

due.cite(
    BibTeX("""@article{mccluskey2024kinisi,
      author = {McCluskey, Andrew R. and Squires, Alexander G. and Dunn, Josh and Coles, Samuel W. and Morgan, Benjamin J.},
      title = {kinisi: Bayesian analysis of mass transport from molecular dynamics simulations},
      journal = {Journal of Open Source Software},
      volume = {9},
      number = {94},
      pages = {5984},
      year = {2024},
      month = feb,
      publisher = {The Open Journal},
      doi = {10.21105/joss.05984}
    }"""),
    description='kinisi: Bayesian analysis of mass transport from molecular dynamics simulations',
    path='kinisi',
    version=__version__,
    cite_module=True,
)
