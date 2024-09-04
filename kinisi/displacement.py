import scipp as sc

from kinisi import parser

def calculate_msd(p: parser.Parser) -> sc.Variable:
    msd = []
    msd_var = []
    n_samples = []
    for di in p.dt_int.values:
        disp = sc.concat([p.displacements['obs', di - 1], p.displacements['obs', di:] - p.displacements['obs', :-di]], 'obs')
        n = (p.displacements.sizes['atom'] * p.dt_int['timestep', -1] / di).value
        s = sc.sum(disp ** 2, 'dimension')
        m = sc.mean(s).value
        v = (sc.var(s, ddof=1) / n).value
        msd.append(m)
        msd_var.append(v)
        n_samples.append(n)
    return sc.DataArray(data=sc.Variable(dims=['timestep'], 
                                values=msd, 
                                variances=msd_var, 
                                unit=s.unit), 
                coords={'timestep': p.dt, 'n_samples': sc.array(dims=['timestep'], values=n_samples), 'dimensionality': p.dimensionality})

def calculate_mstd(p: parser.Parser) -> sc.Variable:
    mstd = []
    mstd_var = []
    n_samples = []
    for di in p.dt_int.values:
        disp = sc.concat([p.displacements['obs', di - 1], p.displacements['obs', di:] - p.displacements['obs', :-di]], 'obs')
        n = (p.displacements.sizes['atom'] * p.dt_int['timestep', -1] / di).value / disp.sizes['atom']
        s = sc.sum(sc.sum(disp, 'atom') ** 2, 'dimension')
        if s.size <= 1:
            continue
        m = sc.mean(s).value
        v = (sc.var(s, ddof=1) / n).value
        mstd.append(m)
        mstd_var.append(v)
        n_samples.append(n)
    return sc.DataArray(data=sc.Variable(dims=['timestep'], 
                                values=mstd, 
                                variances=mstd_var, 
                                unit=s.unit), 
                coords={'timestep': p.dt['timestep', :len(mstd)], 'n_samples': sc.array(dims=['timestep'], values=n_samples), 'dimensionality': p.dimensionality})


def calculate_mscd(p: parser.Parser, ionic_charge: sc.Variable) -> sc.Variable:
    mscd = []
    mscd_var = []
    n_samples = []
    for di in p.dt_int.values:
        disp = sc.concat([p.displacements['obs', di - 1], p.displacements['obs', di:] - p.displacements['obs', :-di]], 'obs')
        n = (p.displacements.sizes['atom'] * p.dt_int['timestep', -1] / di).value / disp.sizes['atom']
        s = sc.sum(sc.sum(ionic_charge * disp, 'atom') ** 2, 'dimension')
        if s.size <= 1:
            continue
        m = sc.mean(s).value
        v = (sc.var(s, ddof=1) / n).value
        mscd.append(m)
        mscd_var.append(v)
        n_samples.append(n)
    return sc.DataArray(data=sc.Variable(dims=['timestep'], 
                                values=mscd, 
                                variances=mscd_var, 
                                unit=s.unit), 
                coords={'timestep': p.dt['timestep', :len(mscd)], 'n_samples': sc.array(dims=['timestep'], values=n_samples), 'dimensionality': p.dimensionality})