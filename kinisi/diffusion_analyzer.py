"""
The :py:class:`kinisi.analyze.DiffusionAnalyser` class enable the evaluation of tracer mean-squared
displacment and the self-diffusion coefficient.
"""

# Copyright (c) kinisi developers. 
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61)

from typing import Union, List
import scipp as sc
from kinisi.displacement import calculate_msd
from kinisi.diffusion import Diffusion
from kinisi.parser import Parser, PymatgenParser


class DiffusionAnalyzer:
    def __init__(self, trajectory: Parser) -> None:
        self.trajectory = trajectory
        self.msd = calculate_msd(self.trajectory)

    @classmethod
    def from_Xdatcar(cls,
                     trajectory: Union['pymatgen.io.vasp.outputs.Xdatcar', List['pymatgen.io.vasp.outputs.Xdatcar']],
                     parser_params: dict):
        p = PymatgenParser(trajectory, **parser_params)
        return cls(p)
    
    def diffusion(self, start_dt: sc.Variable, diffusion_params: Union[dict, None] = None) -> None:
        if diffusion_params is None:
            diffusion_params = {}
        self.diff = Diffusion(self.msd)
        self.diff.diffusion(start_dt, **diffusion_params)