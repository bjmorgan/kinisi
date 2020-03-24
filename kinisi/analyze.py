"""
Analyze class. This is a API class that it is anticipated that most will 
make use of.
"""

# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import MDAnalysis as mda
from pymatgen.io.vasp import Xdatcar
from kinisi import diffusion
from kinisi.parser import MDAnalysisParser, PymatgenParser

class DiffAnalyzer:
    """
    Attributes:

    Args:
    """
    def __init__(self, file, params, format='Xdatcar', uncertainty='con_int', chg_diff=False):
        if format is 'Xdatcar':
            xd = Xdatcar(file)
            u = PymatgenParser(xd.structures, **params)
        else:
            universe = mda.Universe(*file, format=format)
            u = MDAnalysisParser(universe, **params)

        self.delta_t = u.delta_t
        self.disp_3d = u.disp_3d
        self.indices = u.indices

        diff_data = diffusion.msd_bootstrap(self.delta_t, self.disp_3d)

        self.delta_t = diff_data[0]
        self.msd = diff_data[1]
        self.msd_err = diff_data[2]
        self.msd_lb = diff_data[3]
        self.msd_ub = diff_data[4]

        if uncertainty is 'con_int':
            diff = diffusion.Diffusion(self.delta_t, self.msd, self.msd - self.msd_lb)
        elif uncertainty is 'std_dev':
            diff = diffusion.Diffusion(self.delta_t, self.msd, self.msd_err) 
        else:
            raise ValueError('Only `con_int` and `std_dev` are accepted `uncertainty` options.')

        diff.max_likelihood()
        diff.sample()

        self.D = diff.diffusion_coefficient

        if chg_diff:
            diff_data = diffusion.mscd_bootstrap(self.delta_t, self.disp_3d, self.indices)
            self.delta_t_mscd = diff_data[0]
            self.mscd = diff_data[1]
            self.mscd_err = diff_data[2]
            self.mscd_lb = diff_data[3]
            self.mscd_ub = diff_data[4] 


        


