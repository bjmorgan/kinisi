<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/kinisi-dev/kinisi/blob/main/docs/source/_static/logo_light.png?raw=true">
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/kinisi-dev/kinisi/blob/main/docs/source/_static/logo_dark.png?raw=true">
  <img alt="The kinisi logo" src="https://github.com/kinisi-dev/kinisi/blob/main/docs/source/_static/logo_light.png?raw=true">
</picture>

Pronunciation: *kee-nee-see*

[![JCTC Status](https://img.shields.io/badge/JCTC-10.1021/acs.jctc.4c01249-blue.svg)](https://doi.org/10.1021/acs.jctc.4c01249)
[![JOSS Status](https://joss.theoj.org/papers/1ae102ffb6b3c63b04c002976440815d/status.svg)](https://joss.theoj.org/papers/1ae102ffb6b3c63b04c002976440815d)
[![PyPI version](https://badge.fury.io/py/kinisi.svg)](https://badge.fury.io/py/kinisi)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kinisi-dev/kinisi)

`kinisi` is an open-source package focussed on accurately quantifying the uncertainty in atomic and molecular displacements, and using this to more completely understand diffusion in materials.

## Installation

`kinisi` can be installed from [PyPI](https://pypi.org/project/kinisi/) with `pip`:

```
pip install kinisi
```

or from [conda-forge](https://anaconda.org/conda-forge/kinisi):

```
conda install -c conda-forge kinisi
```

To install the latest development version, clone [this repository](https://github.com/kinisi-dev/kinisi) and run `pip install .` from its root. See the [installation guide](https://kinisi.rtfd.io/en/latest/installation.html) for development and documentation builds.

## Quick start

Estimate the lithium self-diffusion coefficient, with its uncertainty, from a VASP XDATCAR (reading the file needs [`pymatgen`](https://pymatgen.org); ASE and MDAnalysis trajectories are also supported):

```python
import scipp as sc
from pymatgen.io.vasp import Xdatcar
from kinisi.analyze import DiffusionAnalyzer

xd = Xdatcar('XDATCAR')
diff = DiffusionAnalyzer.from_xdatcar(
    xd, specie='Li', time_step=2.0 * sc.Unit('fs'), step_skip=50 * sc.Unit('dimensionless')
)

# Fit from the time at which the mean-squared displacement becomes linear (diffusive).
diff.diffusion(start_dt=3000 * sc.Unit('fs'))

print(sc.mean(diff.D), sc.std(diff.D, ddof=1))  # diff.D holds samples of D in cm^2/s
```

The [documentation](https://kinisi.rtfd.io) has worked examples for each trajectory format, jump diffusion, conductivity, Arrhenius fitting, and finite-size corrections, as well as a description of the [methodology](https://kinisi.rtfd.io/en/latest/methodology.html).

## Citing

If you use `kinisi` in your research, please cite the [methodology paper](https://doi.org/10.1021/acs.jctc.4c01249) and the [JOSS software paper](https://doi.org/10.21105/joss.05984), and give the version of `kinisi` you used. See the [FAQ](https://kinisi.rtfd.io/en/latest/faq.html) for the other papers to cite for specific features.

## Contributing

If you would like to contribute to `kinisi`, have a look at [CONTRIBUTING.md](https://github.com/kinisi-dev/kinisi/blob/main/CONTRIBUTING.md), which outlines the different ways to help out.
