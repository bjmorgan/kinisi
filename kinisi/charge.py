"""
Investigation of the conductivity of ions in a material. This class takes
the displacements of atoms as a series of timesteps.
"""
# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

# pylint: disable=R0913

import numpy as np
from pint import UnitRegistry
from kinisi.distribution import Distribution
from kinisi.motion import Motion

UREG = UnitRegistry()


class Charge(Motion):
    """
    Methods for the determination of the squared mean displacements and
    charge diffusion from a set of displacements.

    Attributes:
        conductivity (kinisi.distribution.Distribution): The
            distribution of the charge diffusion from the straight
            line plot.
    """
    def __init__(self, displacements, abscissa, conversion_factor,
                 ordinate_units=UREG.angstrom**2,
                 abscissa_units=UREG.femtosecond, step_freq=1):
        """
        Args:
            displacements (list of array_like): A list of arrays, where
                each array has the axes [atom, displacement observation]
                and describes the displacements. There is one array in the
                list for each delta_t value. Note: this can't be a 3D array
                because we have a different number of displacements at each
                dt value.
            abscissa (array_like): The abscissa values that match with the
                delta_t values.
            conversion_factor (float): The value to convert from diffusion to
                conductivity.
            ordinate_units (pint.UnitRegistry(), optional) The units of the
                displacement data. Default is square ångström.
            abscissa_units (pint.UnitRegistry(), optional) The units of the
                delta_t data. Default is picosecond.
            step_freq (int, optional): The frequency in delta_t to be
                sampled. Default is `1`, sampling every step.
        """
        super().__init__(
            displacements,
            abscissa,
            conversion_factor,
            ordinate_units,
            abscissa_units,
            step_freq,
        )
        for i in range(len(self.displacements)):
            self.ordinate[i] = np.mean(self.displacements[i]) ** 2
            self.num_part[i] = self.displacements[i].size
        # Errors from random walk
        # https://pdfs.semanticscholar.org/
        # 5249/8c4c355c13b19093d897a78b11a44be4211d.pdf
        self.ordinate_error = 2 * np.sqrt(
            self.ordinate) * np.sqrt(6 / self.num_part)
        self.equ_straight_line()
        self.chg_diffusion = self.gradient / 60
        self.chg_conductivity = (self.chg_diffusion * self.conversion_factor)

    def resample_smd(self, **kwargs):
        """
        Resample the square-displacement data to obtain a description of
        the distribution as a function of the number of timesteps. Then
        take the mean of this.

        Args:
            n_resamples (int, optional): The initial number of resamples to
                be performed. Default is `1000`.
            samples_freq (int. optional): The frequency in observations to be
                sampled. Default is `1`.
            confidence_interval (array_like): The percentile points of the
                distribution that should be stored. Default is `[2.5, 97.5]`
                which is a 95 % confidence interval.
            progress (bool, optional): Show tqdm progress for sampling.
                Default is `True`.
        """
        self.resample(**kwargs)
        self.ordinate = self.ordinate ** 2
        self.ordinate_error = 2 * self.ordinate * self.ordinate_error

    def sample_chg_diffusion(self, **kwargs):
        """
        Use MCMC sampling to evaluate the charge diffusion coefficient.

        Args:
            walkers (int, optional): Number of MCMC walkers. Default is `100`.
            n_samples (int, optional): Number of sample points. Default is
                `500`.
            n_burn (int, optional): Number of burn in samples. Default is
                `500`.
            progress (bool, optional): Show tqdm progress for sampling.
                Default is `True`.
        """
        self.sample(**kwargs)
        self.chg_diffusion = Distribution(
            name=r'$D_c$',
            units=self.ordinate_units / self.abscissa_units
        )
        # The factor of 10 converts from angstrom^2 / fs to cm^2/s.
        # The factor of 6 is for dimensionality.
        self.chg_diffusion.add_samples(self.gradient.samples / 60)
        self.chg_diffusion.units = UREG.centimeters**2 / UREG.seconds
        self.chg_conductivity = Distribution(
            name=r'$\sigma_c$',
            units=UREG.millisieverts / UREG.centimeters,
        )
        self.chg_conductivity.add_samples(
            self.chg_diffusion.samples * self.conversion_factor
        )

    def plot_smd(self, figsize=(10, 6)):
        """
        Plot the mean squared displacements against the timesteps. Additional
        plots will be included on this if the data has been resampled or the
        MCMC sampling has been used to find the gradient and intercept
        distributions.

        Args:
            fig_size (tuple, optional): Horizontal and veritcal size for figure
                (in inches). Default is `(10, 6)`.

        Returns:
            (matplotlib.figure.Figure)
            (matplotlib.axes.Axes)
        """
        fig, axes = self.plot(figsize=figsize)
        axes.set_ylabel(
            r'$\langle \delta \mathbf{r} \rangle ^ 2$/' + '${:~L}$'.format(
                self.ordinate_units,
            )
        )
        return fig, axes
