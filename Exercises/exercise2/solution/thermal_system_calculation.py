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


def get_heat_flux_by_convection(h: Numeric, temp_from: Numeric, temp_to: Numeric) -> float:
    """Returns heat flux [W/m2] by convection from a surface of a temperature (temp_from) to the
    fluid with another temperature (temp_to) for the given heat convection coefficient (alpha)"""
    return h * (temp_from - temp_to)


###################################################################################################
# Exercise 2

def get_thermal_resistance_cylinder(
        radius: Numeric,
        wall_thickness: Numeric,
        thermal_conductivity: Numeric,
        length: Numeric
) -> Numeric:
    """Returns thermal resistance [m2K/W] for a cylinder"""
    outer_radius = radius + wall_thickness
    return np.log(outer_radius / radius) / (2 * np.pi * length * thermal_conductivity)


Numeric = Union[float, np.ndarray]


@unique
class TipBoundaryConditionForFin(Enum):
    Convection = auto()
    Adiabatic = auto()
    Temperature = auto()
    Long = auto()


class FinConfiguration(NamedTuple):
    perimeter: float
    area_cross_section: float
    length: float


def get_heat_transfer_1D_fin_with_uniform_cross_section(
        x: Numeric,
        temp_base: Numeric,
        temp_surr: Numeric,
        fin_configuration: FinConfiguration,
        h: Numeric,
        k: Numeric,
        boundary_condition: str,
        temp_tip: Numeric = None,
        length: float = 1
) -> Tuple[Numeric, Numeric]:
    m = np.sqrt(h * fin_configuration.perimeter / (k * fin_configuration.area_cross_section))
    h_mk = h / (m * k)
    m_l_x = m * (fin_configuration.length - x)
    m_l = m * fin_configuration.length
    theta_b = temp_base - temp_surr
    mm = np.sqrt(h * fin_configuration.perimeter * k * fin_configuration.area_cross_section) * \
        theta_b
    if boundary_condition == TipBoundaryConditionForFin.Convection:
        denominator = (np.cosh(m_l) + h_mk * np.sinh(m_l))
        theta = (np.cosh(m_l_x) + h_mk * np.sinh(m_l_x)) / denominator * theta_b
        heat_transfer_rate = mm * (np.sinh(m_l) + h_mk * np.cosh(m_l)) / denominator
    elif boundary_condition == TipBoundaryConditionForFin.Adiabatic:
        theta = np.cosh(m_l_x) / np.cosh(m_l) * theta_b
        heat_transfer_rate = mm * np.tanh(m_l)
    elif boundary_condition == TipBoundaryConditionForFin.Temperature:
        theta_l_b = (temp_tip - temp_surr) / theta_b
        theta = (theta_l_b * np.sinh(m * x) + np.sinh(m_l_x)) / np.sinh(m_l) * theta_b
        heat_transfer_rate = mm * (np.cosh(m_l) - theta_l_b) / np.sinh(m_l)
    elif boundary_condition == TipBoundaryConditionForFin.Long:
        theta = np.exp(-m * x) * theta_b
        heat_transfer_rate = mm
    else:
        raise TypeError(f'Invalid boundary condition is given: {boundary_condition}')
    temp = theta + temp_surr
    return heat_transfer_rate, temp


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


@unique
class NonCircularCylinderGeometry(Enum):
    SQUARE_FLAT = auto()
    SQUARE_OBLIQUE = auto()
    HEXAGON_FLAT = auto()
    HEXAGON_OBLIQUE = auto()
    VERTICAL_FLAT_FRONT = auto()
    VERTICAL_FLAT_BACK = auto()


@unique
class TubeBankArrangement(Enum):
    Aligned = auto()
    Staggered = auto()


class TubeBankConfiguration(NamedTuple):
    arrangement: TubeBankArrangement
    vertical_spacing: float
    horizontal_spacing: float
    number_rows: float
    number_tubes_each_row: float

    def get_maximum_velocity(self, velocity: float, diameter: float):
        if self.arrangement == TubeBankArrangement.Aligned:
            return self.vertical_spacing / (self.vertical_spacing - diameter) * velocity
        diagonal_spacing = np.sqrt(self.horizontal_spacing ** 2 + (self.vertical_spacing / 2) ** 2)
        if diagonal_spacing < np.mean([self.vertical_spacing, diameter]):
            return self.vertical_spacing / (2 * (diagonal_spacing - diameter)) * velocity
        else:
            return self.vertical_spacing / (self.vertical_spacing - diameter) * velocity

    @property
    def number_tubes_total(self):
        return self.number_rows * self.number_tubes_each_row


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
    

class Convection(ABC):
    """Class for convection phenomenum"""

    def __init__(
            self,
            temp_surface: Numeric,
            temp_infinity: Numeric,
            fluid: Fluid,
            characteristic_length,
            pressure_pa: Numeric = 101325
    ):
        """Constructor for the class"""
        self.temp_surface = temp_surface
        self.temp_infinity = temp_infinity
        self.pressure_pa = pressure_pa
        self.fluid_state = FluidState(
            fluid=fluid,
            pressure_pa=pressure_pa,
            temp_k=np.mean([self.temp_surface, self.temp_infinity]),
            characteristic_length=characteristic_length
        )

    @property
    def prantdl_number(self):
        return self.fluid_state.prantdl_number

    @property
    @abstractmethod
    def nusselt_number(self):
        """Returns the nusselt number"""
        pass

    @property
    def h(self):
        """Returns h from a nusselt number"""
        return self.nusselt_number * self.fluid_state.k / self.fluid_state.characteristic_length

    def get_heat_transfer_rate(self, area: Numeric):
        return self.h * area * (self.temp_surface - self.temp_infinity)


class ForcedConvection(Convection, ABC):
    """Abstract Class for forced convection"""

    def __init__(self, velocity: Numeric, *args, **kwargs):
        """Constructor for the class"""
        super().__init__(*args, **kwargs)
        self.velocity = velocity

    @property
    def reynolds_number(self):
        return self.fluid_state.get_reynolds_number(self.velocity)


class ForcedConvectionFlatPlateIsothermSurface(ForcedConvection):
    """Class for forced convection for flat plate """
    _REYNOLDS_NUMBER_AT_TRANSITION = 500000

    @property
    def critical_length(self) -> float:
        """returns the critical length at which transition from laminar to turbulent flow happens"""
        return self._REYNOLDS_NUMBER_AT_TRANSITION * self.fluid_state.dynamic_viscosity / \
            (self.fluid_state.density * self.velocity)

    @property
    def critical_length_ratio(self) -> float:
        """Returns the ratio of critical length to the characteristic length"""
        return self.critical_length / self.fluid_state.characteristic_length

    @property
    def nusselt_number(self):
        """Returns a Nusselt number"""
        # Calculate the characteristic length at transition
        if self.critical_length_ratio < 0.95:
            return (0.037 * self.reynolds_number ** 0.8 - 871) * self.prantdl_number ** (1 / 3)
        else:
            return 0.664 * self.reynolds_number ** 0.5 * self.prantdl_number ** (1 / 3)


class ForcedConvectionCylinderCrossFlow(ForcedConvection):
    """Class for forced convection for a circular cylinder in cross flow"""

    @property
    def nusselt_number(self):
        """Returns a Nusselt number by Churchill and Bernstein equation"""
        if self.reynolds_number * self.prantdl_number < 0.2:
            logger.warning("The method used here may not be valid because the "
                           "value of Re * Pr is less than 0.2")
        return 0.3 + (0.62 * self.reynolds_number ** 0.5 * self.prantdl_number ** (1/3)) / \
            (1 + (0.4 / self.prantdl_number) ** (2 / 3)) ** 0.25 * \
            (1 + (self.reynolds_number / 282000) ** (5 / 8)) ** (4 / 5)


class ForcedConvectionNonCircularCylinderCrossFlow(ForcedConvection):
    """Class for forced convection for a non-circular cylinder in cross flow"""

    def __init__(self, geometry: NonCircularCylinderGeometry, *args, **kwargs):
        """Constructor"""
        super().__init__(*args, **kwargs)
        self.geometry = geometry

    @property
    def nusselt_number(self):
        """Returns a Nusselt number"""
        error_msg_reynolds_number = f"The Nusselt number may not be valid as the Reynolds number" \
                                    f", {self.reynolds_number} is out of valid range"
        if self.geometry == NonCircularCylinderGeometry.SQUARE_FLAT:
            if self.reynolds_number < 6000 or self.reynolds_number > 60000:
                logger.warning(f"{error_msg_reynolds_number} (5000-60000).")
            c, m = 0.304, 0.59
        elif self.geometry == NonCircularCylinderGeometry.SQUARE_OBLIQUE:
            if self.reynolds_number < 5000 or self.reynolds_number > 60000:
                logger.warning(f"{error_msg_reynolds_number} (5000-60000).")
            c, m = 0.158, 0.66
        elif self.geometry == NonCircularCylinderGeometry.HEXAGON_FLAT:
            if self.reynolds_number < 5200:
                logger.warning(f"{error_msg_reynolds_number} (<5200).")
            if self.reynolds_number < 20400:
                c, m = 0.164, 0.638
            else:
                if self.reynolds_number > 105000:
                    logger.warning(f"{error_msg_reynolds_number} (>105000).")
                c, m = 0.039, 0.78
        elif self.geometry == NonCircularCylinderGeometry.HEXAGON_OBLIQUE:
            if self.reynolds_number < 4500 or self.reynolds_number > 90700:
                logger.warning(f"{error_msg_reynolds_number} (4500 - 90700).")
            c, m = 0.150, 0.638
        elif self.geometry == NonCircularCylinderGeometry.VERTICAL_FLAT_FRONT:
            if self.reynolds_number < 10000 or self.reynolds_number > 50000:
                logger.warning(f"{error_msg_reynolds_number} (10000 - 50000).")
            c, m = 0.667, 0.500
        elif self.geometry == NonCircularCylinderGeometry.VERTICAL_FLAT_BACK:
            if self.reynolds_number < 7000 or self.reynolds_number > 50000:
                logger.warning(f"{error_msg_reynolds_number} (7000 - 50000).")
            c, m = 0.191, 0.667
        else:
            raise TypeError("The type of geometry is not valid. Please use "
                            "NonCircularCylinderGeometry enumerator.")
        logger.info(f"{self.geometry}: C={c}, m={m}")
        return c * self.reynolds_number ** m * self.prantdl_number ** (1 / 3)


class ForcedConvectionTubeBanksCrossFlow(ForcedConvection):
    """Class for forced convection for banks of tubes in cross flow"""

    def __init__(self, configuration: TubeBankConfiguration, *args, **kwargs):
        """Constructor"""
        super().__init__(*args, **kwargs)
        self.configuration = configuration

    @property
    def correction_factor(self):
        if self.configuration.number_rows >= 20:
            return 1
        else:
            number_rows_ref = np.array([1, 2, 3, 4, 5, 7, 10, 13, 16])
            corr_factor_aligned = np.array([0.7, 0.8, 0.86, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99])
            corr_factor_staggered = np.array([0.64, 0.76, 0.84, 0.89, 0.92, 0.95, 0.97, 0.98, 0.99])
            if self.configuration.arrangement == TubeBankArrangement.Aligned:
                return np.interp(
                    self.configuration.number_rows, number_rows_ref, corr_factor_aligned)
            else:
                return np.interp(
                    self.configuration.number_rows, number_rows_ref, corr_factor_staggered)

    @property
    def reynolds_number(self):
        """Returns the Reynolds number depending the maximum velocity through the banks"""
        max_velocity = self.configuration.get_maximum_velocity(
            self.velocity, self.fluid_state.characteristic_length)
        return self.fluid_state.get_reynolds_number(max_velocity)

    @property
    def prantdl_number_surface(self):
        """Returns the prandtl number at the surface"""
        fluid_state_surface = FluidState(
            fluid=Fluid[self.fluid_state.fluid],
            pressure_pa=self.fluid_state.pressure_pa,
            temp_k=self.temp_surface,
            characteristic_length=self.fluid_state.characteristic_length
        )
        return fluid_state_surface.prantdl_number

    @property
    def nusselt_number(self):
        """Returns a Nusselt number"""
        if self.reynolds_number < 100:
            if self.configuration.arrangement == TubeBankArrangement.Aligned:
                c, m = 0.8, 0.4
            else:
                c, m = 0.9, 0.4
        elif self.reynolds_number < 1000:
            convection = ForcedConvectionCylinderCrossFlow(
                velocity=self.velocity,
                temp_surface=self.temp_surface,
                temp_infinity=self.temp_infinity,
                fluid=self.fluid_state.fluid,
                characteristic_length=self.fluid_state.characteristic_length,
                pressure_pa=self.fluid_state.pressure_pa
            )
            return self.correction_factor * convection.nusselt_number
        elif self.reynolds_number < 200000:
            if self.configuration.arrangement == TubeBankArrangement.Aligned:
                c, m = 0.27, 0.63
            else:
                ratio = self.configuration.vertical_spacing / self.configuration.horizontal_spacing
                if ratio < 2:
                    c, m = 0.35 * ratio ** 0.2, 0.60
                else:
                    c, m = 0.4, 0.60
        else:
            if self.configuration.arrangement == TubeBankArrangement.Aligned:
                c, m = 0.021, 0.84
            else:
                c, m = 0.022, 0.84
            if self.reynolds_number > 2 * 10 ** 6:
                logger.warning("The Nusselt number may not be valid because the Reynolds number "
                               "is out of the valid range (<2000000)")
        return self.correction_factor * c * self.reynolds_number ** m * \
            self.prantdl_number ** 0.36 * \
            (self.prantdl_number / self.prantdl_number_surface) ** 0.25

    @property
    def temp_out(self):
        """Returns the estimation of the outlet temperature of the external flow"""
        fluid_state_inlet = FluidState(
            fluid=Fluid[self.fluid_state.fluid],
            temp_k=self.temp_infinity,
            characteristic_length=self.fluid_state.characteristic_length
        )
        ratio = np.exp(
            -1 * np.pi * self.fluid_state.characteristic_length *
            self.configuration.number_tubes_total * self.h /
            (fluid_state_inlet.density * self.velocity * self.configuration.number_tubes_each_row
             * self.configuration.vertical_spacing * fluid_state_inlet.cp)
        )
        return self.temp_surface - ratio * (self.temp_surface - self.temp_infinity)

    @property
    def log_mean_temperature_difference(self):
        """Returns the logarithmic mean of temperature difference for the surface and external
        flow"""
        return (self.temp_out - self.temp_infinity) / \
               np.log((self.temp_surface - self.temp_infinity)/(self.temp_surface - self.temp_out))

    def get_heat_transfer_rate(self, length: Numeric):
        """Returns the heat transfer rate"""
        return self.configuration.number_tubes_total * \
            (self.h * np.pi * self.fluid_state.characteristic_length *
             self.log_mean_temperature_difference)
