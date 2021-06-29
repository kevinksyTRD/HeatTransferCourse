"""This module contains useful functions and classes for thermal system calculation"""

import numpy as np
from typing import Union


def get_heat_flux_by_conduction(
        #your arguments here
) -> Union[float, np.ndarray]:
    """Returns heat flux [W/m2] by conduction from a section of material to another"""
    # Your function implementation here


def get_heat_flux_by_convection(#Your arguments here
) -> float:
    """Returns heat flux [W/m2] by convection from a surface of a temperature (temp_from) to the
    fluid with another temperature (temp_to) for the given heat convection coefficient (alpha)"""
    # Your function implementation here