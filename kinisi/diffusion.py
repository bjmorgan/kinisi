"""
Investigation of the diffuion of atoms in a material. This class takes
the square displacements of atoms as a series of timesteps.
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


class Diffusion(Motion):
    """
    Methods for the determination of the mean squared displacements and
    diffusion coefficient from a set of squared displacements.

    This subclasses the `Motion` class

    Attributes:
        diffusion_coefficient (kinisi.distribution.Distribution): The
            distribution of the diffusion coefficient from the straight
            line plot.
    """
    def __init__(self, sq_displacements, abscissa,
                 ordinate_units=UREG.angstrom**2,
                 abscissa_units=UREG.picosecond, step_freq=1):
        """
        Args:
            sq_displacements (list of array_like): A list of arrays, where
                each array has the axes [atom, squared displacement
                observation] and describes the squared displacements.
                There is one array in the list for each delta_t value.
                Note: this can't be a 3D array because we have a different
                number of displacements at each dt value.
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
            sq_displacements,
            abscissa,
            ordinate_units,
            abscissa_units,
            step_freq,
        )
        for i in range(len(self.displacements)):
            self.ordinate[i] = np.mean(self.displacements[i])
            self.num_part[i] = self.displacements[i].size
        # Errors from random walk
        # https://pdfs.semanticscholar.org/
        # 5249/8c4c355c13b19093d897a78b11a44be4211d.pdf
        self.ordinate_error = np.sqrt(6 / self.num_part) * self.ordinate
        self.equ_straight_line()
        self.diffusion_coefficient = self.gradient / 6

    def resample_msd(self, **kwargs):
        """
        Resample the squared displacement data to obtain a description of
        the distribution as a function of the number of timesteps.

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

    def sample_diffusion(self, **kwargs):
        """
        Use MCMC sampling to evaluate diffusion coefficient.

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
        self.diffusion_coefficient = Distribution(
            name='$D$',
            units=self.ordinate_units / self.abscissa_units
        )
        self.diffusion_coefficient.add_samples(self.gradient.samples / 6)

    def plot_msd(self, figsize=(10, 6)):
        """
        Plot the MSD against the timesteps. Additional plots will be included
        on this if the data has been resampled or the MCMC sampling has been
        used to find the gradient and intercept distributions.

        Args:
            fig_size (tuple, optional): Horizontal and veritcal size for figure
                (in inches). Default is `(10, 6)`.

        Returns:
            (matplotlib.figure.Figure)
            (matplotlib.axes.Axes)
        """
        fig, axes = self.plot(figsize=figsize)
        axes.set_ylabel(
            r'$\langle \delta \mathbf{r} ^ 2 \rangle$/' + '${:~L}$'.format(
                self.ordinate_units,
            )
        )
        return fig, axes
