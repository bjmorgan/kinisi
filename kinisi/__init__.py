"""
Initialisation of kinisi.
"""


from pint import UnitRegistry, set_application_registry
UREG = UnitRegistry()
Q_ = UREG.Quantity
set_application_registry(UREG)
