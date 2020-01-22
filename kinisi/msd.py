"""
Class for MSD investigation.

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0902,R0913

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pint import UnitRegistry
from kinisi import utils, straight_line, _fig_params
from kinisi.distribution import Distribution


UREG = UnitRegistry()


class MSD:
    """
    Mean-squared displacements, with uncertainties.

    Attributes:
        sq_displacements (list of array_like): A list of arrays, where
                each array has the axes [atom, squared displacement
                observation] and describes the squared displacements.
                The dimension of the list is down sampled by the
                `step_freq`.
        mean (array_like): The mean values for the ordinate axis.
        err (array_like): The uncertainty in the mean values for the
            ordinate axis.
        abscissa (array_like): The abscissa values that match with the
            delta_t values.
        gradient (kinisi.distribution.Distribution): The distribution of the
            gradient for the straight line plot.
        intercept (kinisi.distribution.Distribution): The distribution of the
            intercept for the straight line plot.
        diffusion_coefficient (kinisi.distribution.Distribution): The
            distribution of the diffusion coefficient from the straight
            line plot.
        ordinate_units (pint.UnitRegistry(), optional) The units of the
            displacement data. Default is square ångström.
        abscissa_units (pint.UnitRegistry(), optional) The units of the
                delta_t data. Default is picosecond.
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
        self.sq_displacements = sq_displacements[::step_freq]
        self.mean = np.zeros((len(self.sq_displacements)))
        for i in range(len(self.sq_displacements)):
            self.mean[i] = np.mean(self.sq_displacements[i])
        self.err = None
        self.abscissa = abscissa[::step_freq]
        self.gradient = None
        self.intercept = None
        self.diffusion_coefficient = None
        self.ordinate_units = ordinate_units
        self.abscissa_units = abscissa_units

    def resample(self, **kwargs):
        """
        Resample the square-displacement data to obtain a description of
        the distribution as a function of delta_t.

        Args:
            n_resamples (int, optional): The number of resamples to be
                performed.
            samples_freq (int. optional): The frequency in observations to be
                sampled.
            confidence_interval (array_like): The percentile points of the
                distribution that should be stored.
        """
        self.mean, self.err = utils.bootstrap(self.sq_displacements, **kwargs)

    def estimate_straight_line(self):
        """
        Estimate the straight line gradient and intercept

        Returns:
            (tuple) Length of 2, gradient and intercept.
        """
        result = linregress(self.abscissa, self.mean)
        return result.slope, result.intercept

    def sample_diffusion(self):
        """
        Use MCMC sampling to evaluate diffusion coefficient.
        """
        samples = straight_line.run_sampling(
            self.estimate_straight_line(),
            self.mean,
            self.err,
            self.abscissa,
        )
        self.gradient = Distribution(name='gradient')
        self.gradient.add_samples(samples[:, 0])
        self.intercept = Distribution(name='intercept')
        self.intercept.add_samples(samples[:, 1])
        self.diffusion_coefficient = Distribution(name='diffusion coefficient')
        self.diffusion_coefficient.add_samples(self.gradient.samples / 6)

    def plot(self, figsize=(10, 6)):
        """
        Make a nice plot depending on what has been done.
        """
        fig, axes = plt.subplots(figsize=figsize)
        axes.plot(self.abscissa, self.mean)
        axes.set_xlabel(r'$\delta t$/${:~L}$'.format(self.abscissa_units))
        axes.set_ylabel(
            r'$\langle \delta \mathbf{r} ^ 2 \rangle$/' + '${:~L}$'.format(
                self.ordinate_units,
            )
        )
        if self.err is not None:
            axes.fill_between(
                self.abscissa,
                self.mean - self.err,
                self.mean + self.err,
                alpha=0.5,
            )
            if self.diffusion_coefficient is None:
                gradient, intercept = self.estimate_straight_line()
                axes.plot(
                    self.abscissa,
                    utils.straight_line(self.abscissa, gradient, intercept),
                )
            else:
                plot_samples = np.random.randint(
                    0,
                    self.gradient.samples.size,
                    size=100,
                )
                for i in plot_samples:
                    axes.plot(
                        self.abscissa,
                        utils.straight_line(
                            self.abscissa,
                            self.gradient.samples[i],
                            self.intercept.samples[i],
                        ),
                        c=list(_fig_params.TABLEAU)[2],
                        alpha=0.01,
                    )
        return fig, axes
