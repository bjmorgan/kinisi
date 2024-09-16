"""
Functions for the calculation of different types of displacement. 
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61)

import scipp as sc
from tqdm import tqdm
from kinisi import parser


def calculate_msd(p: parser.Parser, progress: bool = True) -> sc.Variable:
    """
    Calculate the mean-squared displacement.
    
    :param p: The parser object containing the the relevant simulation trajectory data.
    :param progress: Whether to show the progress bar. Optional, default is :py:attr:`True`.

    :return: A :py:class:`scipp.DataArray` object containing the relevant mean-squared displacement
        data and number of independent samples. 
    """
    msd = []
    msd_var = []
    n_samples = []
    iterator = p.dt_int.values
    if progress:
        iterator = tqdm(p.dt_int.values, desc='Finding Means and Variances')
    for di in iterator:
        disp = sc.concat([p.displacements['obs', di - 1], p.displacements['obs', di:] - p.displacements['obs', :-di]],
                         'obs')
        n = (p.displacements.sizes['atom'] * p.dt_int['timestep', -1] / di).value
        s = sc.sum(disp**2, 'dimension')
        m = sc.mean(s).value
        v = (sc.var(s, ddof=1) / n).value
        msd.append(m)
        msd_var.append(v)
        n_samples.append(n)
    return sc.DataArray(data=sc.Variable(dims=['timestep'], values=msd, variances=msd_var, unit=s.unit),
                        coords={
                            'timestep': p.dt,
                            'n_samples': sc.array(dims=['timestep'], values=n_samples),
                            'dimensionality': p.dimensionality
                        })


def calculate_mstd(p: parser.Parser, progress: bool = True) -> sc.Variable:
    """
    Calculate the mean-squared total displacement, i.e., the displacement of the centre-of-mass of all particles.
    
    :param p: The parser object containing the the relevant simulation trajectory data.
    :param progress: Whether to show the progress bar. Optional, default is :py:attr:`True`.

    :return: A :py:class:`scipp.DataArray` object containing the relevant mean-squared total displacement
        data and number of independent samples. 
    """
    mstd = []
    mstd_var = []
    n_samples = []
    if progress:
        iterator = tqdm(p.dt_int.values, desc='Finding Means and Variances')
    for di in iterator:
        disp = sc.concat([p.displacements['obs', di - 1], p.displacements['obs', di:] - p.displacements['obs', :-di]],
                         'obs')
        n = (p.displacements.sizes['atom'] * p.dt_int['timestep', -1] / di).value / disp.sizes['atom']
        s = sc.sum(sc.sum(disp, 'atom')**2, 'dimension')
        if s.size <= 1:
            continue
        m = sc.mean(s).value
        v = (sc.var(s, ddof=1) / n).value
        mstd.append(m)
        mstd_var.append(v)
        n_samples.append(n)
    return sc.DataArray(data=sc.Variable(dims=['timestep'], values=mstd, variances=mstd_var, unit=s.unit),
                        coords={
                            'timestep': p.dt['timestep', :len(mstd)],
                            'n_samples': sc.array(dims=['timestep'], values=n_samples),
                            'dimensionality': p.dimensionality
                        })


def calculate_mscd(p: parser.Parser, ionic_charge: sc.Variable, progress: bool = True) -> sc.Variable:
    """
    Calculate the mean-squared charge displacement, i.e., the displacement of the centre-of-mass of all particles
    multiplied by the ionic charge.
    
    :param p: The parser object containing the the relevant simulation trajectory data.
    :param progress: Whether to show the progress bar. Optional, default is :py:attr:`True`.

    :return: A :py:class:`scipp.DataArray` object containing the relevant mean-squared charge displacement
        data and number of independent samples. 
    """
    mscd = []
    mscd_var = []
    n_samples = []
    if progress:
        iterator = tqdm(p.dt_int.values, desc='Finding Means and Variances')
    for di in iterator:
        disp = sc.concat([p.displacements['obs', di - 1], p.displacements['obs', di:] - p.displacements['obs', :-di]],
                         'obs')
        n = (p.displacements.sizes['atom'] * p.dt_int['timestep', -1] / di).value / disp.sizes['atom']
        s = sc.sum(sc.sum(ionic_charge * disp, 'atom')**2, 'dimension')
        if s.size <= 1:
            continue
        m = sc.mean(s).value
        v = (sc.var(s, ddof=1) / n).value
        mscd.append(m)
        mscd_var.append(v)
        n_samples.append(n)
    return sc.DataArray(data=sc.Variable(dims=['timestep'], values=mscd, variances=mscd_var, unit=s.unit),
                        coords={
                            'timestep': p.dt['timestep', :len(mscd)],
                            'n_samples': sc.array(dims=['timestep'], values=n_samples),
                            'dimensionality': p.dimensionality
                        })
