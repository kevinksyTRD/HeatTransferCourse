"""This module contains useful functions and classes for thermal system calculation

Symbols:
    h: convection_heat_transfer_coeffient [W/m2K]
    k: thermal conductivity [W/mK]
    r: radius [m]
    cp: specific heat capacity at constant pressure [J/kgK]
    temp: temperature [degC or K]
"""
import logging
from abc import ABC, abstractmethod
from enum import Enum, auto, unique
from functools import cached_property
from typing import Union, Tuple, NamedTuple

import numpy as np
from CoolProp.CoolProp import PropsSI

# Define logger
logger = logging.getLogger('thermal_system_calculation')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)

Numeric = Union[float, np.ndarray]
GRAVITY = 9.81


def get_heat_flux_by_conduction(
        temperature_from: Numeric,
        temperature_to: Numeric,
        thickness: Numeric,
        thermal_conductivity: Numeric,
) -> Numeric:
    """Returns heat flux [W/m2] by conduction from a section of material to another"""
    return thermal_conductivity * (temperature_from - temperature_to) / thickness


def to_kelvin(temp_deg_c):
    """Converts degree celcius to kelvin"""
    return temp_deg_c + 273.15


def to_celcius(temp_k):
    """Converts kevin to degree culcius"""
    return temp_k - 273.15


def get_heat_flux_by_convection(h: Numeric, temp_from: Numeric, temp_to: Numeric) -> float:
    """Returns heat flux [W/m2] by convection from a surface of a temperature (temp_from) to the
    fluid with another temperature (temp_to) for the given heat convection coefficient (alpha)"""
    return h * (temp_from - temp_to)


###################################################################################################

# CoolProp related classes, functions

@unique
class Fluid(Enum):
    WATER = auto()
    AIR = auto()
    AMMONIA = auto()
    CO2 = auto()
    H2 = auto()
    NITROGEN = auto()
    OXYGEN = auto()
    R134A = auto()
    R143A = auto()
    R407C = auto()


@unique
class ThermodynamicState(Enum):
    PRESSURE = 'P'
    TEMPERATURE = 'T'
    SATURATION = 'Q'


@unique
class Properties(Enum):
    SPECIFIC_HEAT_CAPACITY_CONSTANT_PRESSURE = 'C'
    SPECIFIC_HEAT_CAPACITY_CONSTANT_VOLUME = 'CVMASS'
    GAS_CONSTANT_MOL = 'GAS_CONSTANT'
    DENSITY = 'D'
    CRITICAL_TEMPERATURE = 'Tcrit'
    RELATIVE_HUMIDITY = 'R'
    SPECIFIC_ENTHALPY = 'H'
    SPECIFIC_ENTROPY = 'S'
    SPECIFIC_INTERNAL_ENERGY = 'U'
    THERMAL_CONDUCTIVITY = 'CONDUCTIVITY'
    DYNAMIC_VISCOSITY = 'VISCOSITY'
    EXPANSION_COEFFICIENT_ISOBARIC = 'ISOBARIC_EXPANSION_COEFFICIENT'


class FluidState:
    """Class for fluid state"""

    def __init__(
            self,
            fluid: Fluid,
            pressure_pa: Numeric = 101325,
            temp_k: Numeric = 300,
            characteristic_length: Numeric = None
    ):
        """Constructor for the class"""
        self.fluid = fluid.name
        self.pressure_pa = pressure_pa
        self.temp_k = temp_k
        self.characteristic_length = characteristic_length

    @cached_property
    def k(self):
        """Returns thermal conductivity in W/mK"""
        return PropsSI(
            Properties.THERMAL_CONDUCTIVITY.value,
            ThermodynamicState.PRESSURE.value, self.pressure_pa,
            ThermodynamicState.TEMPERATURE.value, self.temp_k,
            self.fluid
        )

    @cached_property
    def dynamic_viscosity(self):
        """Returns dynamic viscosity in Pa-s"""
        return PropsSI(
            Properties.DYNAMIC_VISCOSITY.value,
            ThermodynamicState.PRESSURE.value, self.pressure_pa,
            ThermodynamicState.TEMPERATURE.value, self.temp_k,
            self.fluid
        )

    @cached_property
    def density(self):
        """Returns density kg/m3"""
        return PropsSI(
            Properties.DENSITY.value,
            ThermodynamicState.PRESSURE.value, self.pressure_pa,
            ThermodynamicState.TEMPERATURE.value, self.temp_k,
            self.fluid
        )

    @cached_property
    def cp(self):
        """Returns specific heat capacity at constant pressure"""
        return PropsSI(
            Properties.SPECIFIC_HEAT_CAPACITY_CONSTANT_PRESSURE.value,
            ThermodynamicState.PRESSURE.value, self.pressure_pa,
            ThermodynamicState.TEMPERATURE.value, self.temp_k,
            self.fluid
        )

    @cached_property
    def expansion_coeff_isobaric(self):
        raise NotImplementedError("The method has not been implemented yet.")

    @property
    def prantdl_number(self) -> Numeric:
        """Returns Prantdl number"""
        return self.cp * self.dynamic_viscosity / self.k

    def get_nusselt_number(self, h: Numeric) -> Numeric:
        """Returns Nusselt number for given characteristic length in m and convection heat transfer
        coefficient in W/m2K"""
        if self.characteristic_length is None:
            raise TypeError("Characteristic length is not set for the instance.")
        return h * self.characteristic_length / self.k

    def get_h_from_nusselt_number(self, nu: Numeric) -> Numeric:
        """Returns convection heat transfer coefficient in W/m2K for given Nusselt number"""
        if self.characteristic_length is None:
            raise TypeError("Characteristic length is not set for the instance.")
        return nu * self.k / self.characteristic_length

    def get_reynolds_number(self, velocity: Numeric) -> Numeric:
        """Returns reynolds number for given velocity"""
        if self.characteristic_length is None:
            raise TypeError("Characteristic length is not set for the instance")
        return self.density * velocity * self.characteristic_length / self.dynamic_viscosity
