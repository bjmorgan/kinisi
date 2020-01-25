"""
The general class of motions of atoms/ions in a material. This is subclassed
by the `kinisi.conductivity.Conductivity` and `kinisi.diffusion.Diffusion`
classes.
"""
# Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

# pylint: disable=R0902,R0913

import numpy as np
import matplotlib.pyplot as plt
from pint import UnitRegistry
from kinisi import utils, straight_line, _fig_params
from kinisi.distribution import Distribution

UREG = UnitRegistry()


class Motion:
    """
    The is a general class for storing information about the motion
    of atoms/ions in some species.

    Attributes:
        displacements (list of array_like): The motions of atoms (either
            displacments or squared displacements). This is a list of
            arrays, where the list index is the timestep (the abscissa) and
            the arrays have the axes [atom, motion observation]. Note: this
            can't be a 3D array because we have a different number of
            displacements at each dt value.
        abscissa (array_like): The abscissa values that match with the
            delta_t values.
        ordinate_units (pint.UnitRegistry(), optional) The units of the
            displacement data. Default is square ångström.
        abscissa_units (pint.UnitRegistry(), optional) The units of the
                delta_t data. Default is picosecond.
        step_freq (int, optional): The frequency in delta_t to be
                sampled. Default is `1`, sampling every step.
        ordinate (array_like): The ordinate of the motion measurement.
            This is either the mean-squared displacement (for diffusion) or
            the squared mean-displacement (for conductivity).
        ordinate_error (array_like): The uncertainty of the motion
            measurement.
        num_part (array_like): The number of particle observations at each
            timestep.
        gradient (uncertaintiese.ufloat or
            kinisi.distribution.Distribution): The ufloat or distribution of
            the gradient for the straight line plot.
        intercept (uncertaintiese.ufloat or
            kinisi.distribution.Distribution): The ufloat or distribution of
            the intercept for the straight line plot.
        resampled (bool): `True` if the resampling has been performed.
        mcmced (bool): `True` if the MCMC sampling has been performed,
    """
    def __init__(self, displacements, abscissa,
                 ordinate_units=UREG.angstrom**2,
                 abscissa_units=UREG.picosecond, step_freq=1):
        """
        """
        self.displacements = displacements[::step_freq]
        self.abscissa = abscissa[::step_freq]
        self.ordinate_units = ordinate_units
        self.abscissa_units = abscissa_units
        self.step_freq = step_freq
        self.ordinate = np.zeros((len(self.abscissa)))
        self.ordinate_error = np.zeros((len(self.abscissa)))
        self.num_part = np.zeros((len(self.abscissa)))
        self.gradient = None
        self.intercept = None
        self.resampled = False
        self.mcmced = False

    def equ_straight_line(self):
        """
        Get the equation of the straight line from the abscissa and ordinate,
        with associated ordinate error. This will assigned the `self.gradient`
        and `self.intercept` value to `uncertainties.ufloat`.
        """
        self.gradient, self.intercept = straight_line.equation(
            self.abscissa,
            self.ordinate,
            self.ordinate_error,
        )

    def resample(self, **kwargs):
        """
        Resample the displacement data to obtain a description of
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
        self.ordinate, self.ordinate_error = utils.bootstrap(
            self.displacements,
            **kwargs,
        )
        self.resampled = True

    def sample(self, **kwargs):
        """
        Perform the MCMC sampling to get the distributions of the gradient
        and intercept of the straight line.

        Args:
            walkers (int, optional): Number of MCMC walkers. Default is `100`.
            n_samples (int, optional): Number of sample points. Default is
                `500`.
            n_burn (int, optional): Number of burn in samples. Default is
                `500`.
            progress (bool, optional): Show tqdm progress for sampling.
                Default is `True`.
        """
        samples = straight_line.run_sampling(
            straight_line.equation(
                self.abscissa,
                self.ordinate,
                self.ordinate_error,
            ),
            self.ordinate,
            self.ordinate_error,
            self.abscissa,
            **kwargs,
        )
        self.mcmced = True
        self.gradient = Distribution(name='$m$')
        self.gradient.add_samples(samples[:, 0])
        self.intercept = Distribution(name='$c$')
        self.intercept.add_samples(samples[:, 1])

    def plot(self, figsize=(10, 6)):  # pragma: no cover
        """
        Make a nice plot of abscissa and ordinate, and additional plots
        depending on if the resampling or the MCMC sampling have been
        performed.

        Args:
            fig_size (tuple, optional): Horizontal and veritcal size for figure
                (in inches). Default is `(10, 6)`.

        Returns:
            (matplotlib.figure.Figure)
            (matplotlib.axes.Axes)
        """
        fig, axes = plt.subplots(figsize=figsize)
        axes.plot(self.abscissa, self.ordinate, c=list(_fig_params.TABLEAU)[0])
        axes.set_xlabel(r'$\delta t$/${:~L}$'.format(self.abscissa_units))
        if not self.resampled:
            axes.fill_between(
                self.abscissa,
                self.ordinate - self.ordinate_error,
                self.ordinate + self.ordinate_error,
                alpha=0.5,
                color=list(_fig_params.TABLEAU)[0]
            )
        else:
            axes.fill_between(
                self.abscissa,
                self.ordinate - self.ordinate_error,
                self.ordinate + self.ordinate_error,
                alpha=0.5,
                color=list(_fig_params.TABLEAU)[0]
            )
        if not self.mcmced:
            gradient, intercept = straight_line.equation(
                self.abscissa,
                self.ordinate,
                self.ordinate_error,
            )
            axes.plot(
                self.abscissa,
                straight_line.straight_line(
                    self.abscissa,
                    gradient.n,
                    intercept.n
                ),
                color=list(_fig_params.TABLEAU)[1]
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
                    straight_line.straight_line(
                        self.abscissa,
                        self.gradient.samples[i],
                        self.intercept.samples[i],
                    ),
                    color=list(_fig_params.TABLEAU)[1],
                    alpha=0.05,
                )
        return fig, axes
