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


class Conductivity(Motion):
    """
    Methods for the determination of the squared mean displacements and
    conductivity from a set of displacements.

    This subclasses the `Motion` class

    Attributes:
        conductivity (kinisi.distribution.Distribution): The
            distribution of the conductivity from the straight
            line plot.
    """
    def __init__(self, displacements, abscissa,
                 ordinate_units=UREG.angstrom**2,
                 abscissa_units=UREG.picosecond, step_freq=1):
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
            self.ordinate) * np.sqrt(6 / self.num_part) * np.sqrt(
                self.ordinate_error)
        self.equ_straight_line()
        self.conductivity = self.gradient / 6

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

    def sample_conductivty(self, **kwargs):
        """
        Use MCMC sampling to evaluate conductivity.

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
        self.conductivity = Distribution(
            name=r'$\sigma$',
            units=self.ordinate_units / self.abscissa_units
        )
        self.conductivity.add_samples(self.gradient.samples / 6)

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
