"""This module contains useful functions and classes for thermal system calculation"""

import numpy as np
from typing import Union


def get_heat_flux_by_conduction(
        temperature_from: Union[float, np.ndarray],
        temperature_to: Union[float, np.ndarray],
        thickness: float, 
        thermal_conductivity: float,
) -> Union[float, np.ndarray]:
    """Returns heat flux [W/m2] by conduction from a section of material to another"""
    return thermal_conductivity * (temperature_from - temperature_to) / thickness


def get_heat_flux_by_convection(alpha: float, temp_from: float, temp_to: float) -> float:
    """Returns heat flux [W/m2] by convection from a surface of a temperature (temp_from) to the
    fluid with another temperature (temp_to) for the given heat convection coefficient (alpha)"""
    return alpha * (temp_from - temp_to)


def get_thermal_resistance_cylinder(
        inner_radius: float,
        outer_radius: float,
        thermal_conductivity: float,
        cylinder_height: float
) -> float:
    """Returns thermal resistance [m2K/W] for a cylinder"""
    return np.log(outer_radius/inner_radius) / (2 * np.pi * cylinder_height * thermal_conductivity)
