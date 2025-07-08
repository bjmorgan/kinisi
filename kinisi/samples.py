"""
A class to represent samples of a physical quantity using scipp.
This class extends the scipp.Variable class to provide additional functionality
for handling samples, such as calculating the mean and standard deviation of the samples.
"""

# Copyright (c) kinisi developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61)

import scipp as sc
from uncertainties import ufloat


class Samples(sc.Variable):
    """
    A subclass of scipp.Variable that represents samples of a physical quantity.
    This class is designed to add some specific functionality for handling
    samples, such as calculating the mean and standard deviation of the samples.
    It also overrides the HTML representation to include these statistics.

    :param values: The values of the samples.
    :param unit: The unit of the samples, if applicable. Optional, defaults to dimensionless.
    """

    def __init__(self, values, unit=sc.units.dimensionless):
        super().__init__(values=values, unit=unit, dims=['samples'])

    def _repr_html_(self) -> str:
        """
        This function augments the default HTML representation of a scipp Variable
        to include the mean and standard deviation of the samples.
        """
        split_1 = sc.make_html(self).split('sc-value-preview sc-preview')
        split_2 = split_1[1].split('div')
        split_2[1] = '>' + ufloat(sc.mean(self).value, sc.std(self, ddof=1).value).__str__() + '</'
        split_1[1] = 'div'.join(split_2)
        split_3 = 'sc-value-preview sc-preview'.join(split_1).split('sc-obj-type')
        split_4 = split_3[3].split('div')
        split_5 = split_4[0].split()
        split_5[0] = "'>kinisi.Samples"
        split_4[0] = ' '.join(split_5)
        split_3[3] = 'div'.join(split_4)
        return 'sc-obj-type'.join(split_3)
