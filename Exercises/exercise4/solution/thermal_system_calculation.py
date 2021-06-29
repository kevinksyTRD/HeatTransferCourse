"""This module contains useful functions and classes for thermal system calculation

Symbols:
    h: convection_heat_transfer_coeffient [W/m2K]
    k: thermal conductivity [W/mK]
    r: radius [m]
    cp: specific heat capacity at constant pressure [J/kgK]
    temp: temperature [degC or K]
"""
import logging
import warnings
from abc import ABC, abstractmethod
from enum import Enum, auto, unique
from functools import cached_property
from typing import Union, Tuple, NamedTuple

import numpy as np
from CoolProp.CoolProp import PropsSI
from CoolProp.HumidAirProp import HAPropsSI
import matplotlib.pyplot as plt
from scipy.optimize import root

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
        boundary_condition: TipBoundaryConditionForFin,
        temp_tip: Numeric = None
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
        return 0.3 + (0.62 * self.reynolds_number ** 0.5 * self.prantdl_number ** (1 / 3)) / \
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
        import pdb
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
               np.log(
                   (self.temp_surface - self.temp_infinity) / (self.temp_surface - self.temp_out))

    def get_heat_transfer_rate(self, length: Numeric):
        """Returns the heat transfer rate"""
        return self.configuration.number_tubes_total * \
               (self.h * np.pi * self.fluid_state.characteristic_length *
                self.log_mean_temperature_difference)


########################## Exercise 3 #############################################################

@unique
class FlowType(Enum):
    """Class for enumerator for the flow types"""
    Parallel = auto()
    Counter = auto()
    ShellAndTube = auto()
    Cross = auto()


class FlowHeatExchanger:
    """Class for flow through heat exchanger

    This is a class for a flow in a heat exchanger. There are at least two flows that are
    defined by its fluid, inlet temperature, mass flow and whether the flow is mixed or unmixed.
    If necessary, pressure and temperature output can be also defined.

    Example:
        flow = FlowHeatExchanger(
            fluid=Fluid.WATER,
            temp_in_k=300,
            mass_flow_kg_s=1.5
        )
        cp_fluid = flow.cp
        heat_capacity_rate = flow.heat_capacity_rate

        # If outlet temperature is defined, you can set the value in addtion or provide it to the
        # constructor.

        flow.temp_out = 350

        # If outlet temperature is defined, it is also possible to calculate the heat transfer rate.

        q = flow.heat_transfer_rate

        # if the instance is for a shell and tube type heat exchanger, you should define if the
        # flow is in the shell our outside.

        flow.is_in_shell = True

        # If there is heat loss to the envirionment, it can be set from 0 to 1 as a ratio to the
        # heat transfer rate

        flow.heat_loss = 0.1

    """

    def __init__(
            self,
            fluid: Fluid,
            temp_in_k: float,
            mass_flow_kg_s: float,
            pressure_pa: float = 101325,
            temp_out_k: float = None,
            is_mixed: bool = False,
            is_in_tube: bool = True,
            heat_loss: float = 0
    ):
        """Constructor for the class"""
        self._fluid_state = FluidState(fluid=fluid, temp_k=temp_in_k, pressure_pa=pressure_pa)
        self.temp_in = temp_in_k
        self.temp_out = temp_out_k
        self.mass_flow = mass_flow_kg_s
        self.is_mixed = is_mixed
        self.is_in_tube = is_in_tube
        self.heat_loss = heat_loss

    @property
    def cp(self):
        """Returns the specific heat capacity of the fluid"""
        return self._fluid_state.cp

    @property
    def heat_capacity_rate(self):
        """Returns the heat capacity rate (W/K) of the flow"""
        if self.mass_flow is None:
            raise TypeError("Mass flow is not defined yet.")
        return self.mass_flow * self.cp

    @property
    def heat_transfer_rate(self):
        """Returns the absolute value of the heat transfer rate"""
        if self.mass_flow is None:
            raise TypeError("Mass flow is not defined yet.")
        if self.temp_out is None:
            raise TypeError("Outlet temperature is not defined yet.")
        if (self.temp_out - self.temp_in) > 0:
            return abs(self.heat_capacity_rate * (self.temp_out - self.temp_in)) / \
                   (1 - self.heat_loss)
        else:
            return abs(self.heat_capacity_rate * (self.temp_out - self.temp_in)) / \
                   (1 + self.heat_loss)


class HeatExchanger(ABC):
    """Abstract class for the heat exchanger

    This class is an abstract class for the heat exchanger class. It defines the getter for heat
    transfer rate which can be used like its attribute. The constructor requires at least three
    arguments: flow_type, flow1, flow2. It provides a getter of heat transfer rate which from its
    flows.
    """

    def __init__(
            self,
            flow_type: FlowType,
            flow1: FlowHeatExchanger,
            flow2: FlowHeatExchanger,
            u_h: float = None,
            area: float = None
    ):
        """Constructor for the class

        Arguments:
            flow_type: Flow type defined in FlowType class
            flow1: Flow on one side of the heat exchanger
            flow2: Flow on the other side of the heat exchanger
            u_h: Overall heat transfer coefficient of the heat exchanger
            area: Area of the heat exchange
        """
        self.flow_type = flow_type
        self.flow1 = flow1
        self.flow2 = flow2
        self.u_h = u_h
        self.area = area

    @property
    def heat_transfer_rate(self):
        try:
            return self.flow1.heat_transfer_rate
        except TypeError:
            try:
                return self.flow2.heat_transfer_rate
            except TypeError:
                raise TypeError("None of the flow has both input and output temperature defined.")

    @property
    def flow_hot_side(self):
        return self.flow1 if self.flow1.temp_in > self.flow2.temp_in else self.flow2

    @property
    def flow_cold_side(self):
        return self.flow1 if self.flow1.temp_in < self.flow2.temp_in else self.flow2

    @abstractmethod
    def solve_temperature(self):
        pass

    @abstractmethod
    def get_heat_exchange_area(self, heat_transfer_coefficient: float):
        pass

    @abstractmethod
    def get_overall_heat_transfer_coefficient(self, heat_exchange_area: float):
        pass

    def draw_temperature_area_diagram(self):
        if self.flow1.temp_out is None or self.flow2.temp_out is None:
            self.solve_temperature()
        fig, ax = plt.subplots()
        ax.plot([0, 1], [self.flow_hot_side.temp_in])


class HeatExchangerLMTD(HeatExchanger):
    """Class for the heat changer analysis using LMTD

    The class is used to do the heat transfer analysis for a heat exchanger using logarithmic
    mean temperature difference. Here, it is assumed that at least one of the outlet temperature
    of the flows is defined.

    The class inherits HeatExchanger class. The class is initiated by passing the required
    arguments of flow_type, flow1 and flow2 and optional arguments of u_h, area,
    number_tube_passes and nuber_shells. The last two arguments are used to find a correction
    factor for a complex geometry like shell-and-tube type heat exchanger or cross-flow heat
    exchanger.

    Methods:
        solve_temperature: finds any missing outlet temperature by energy balance.
        logarithmic_mean_temperature_difference: getter that returns the logarithmic mean
            temperature difference after solving temperature if not solved already.
        get_heat_exchange_area: returns heat exchange area for the given overall heat transfer
            coefficient.
        get_overall_heat_transfer_coefficient: returns heat transfer coefficient for the given
            heat exchange area.
    """

    def __init__(
            self,
            flow_type: FlowType,
            flow1: FlowHeatExchanger,
            flow2: FlowHeatExchanger,
            u_h: float = None,
            area: float = None,
            number_tube_passes: int = 1,
            number_shells: int = 1,
            is_parallel_flow: bool = None
    ):
        """Constructor for the class

        Arguments:
            flow_type: Flow type defined in FlowType class
            flow1: Flow on one side of the heat exchanger
            flow2: Flow on the other side of the heat exchanger
            u_h: Overall heat transfer coefficient of the heat exchanger
            area: Area of the heat exchange
            number_tube_passes: number of tube passes
            number_shells: number of shells
        """
        # The following validation is required to make sure that one of the flow is defined as
        # in-shell flow.
        if flow_type == FlowType.ShellAndTube:
            if flow1.is_in_tube == flow2.is_in_tube:
                raise TypeError("Either of the flows should have `is_in_tube` attribute true "
                                "while the other is false.")
        self.is_parallel_flow = True if flow_type == FlowType.Parallel else is_parallel_flow
        super().__init__(flow_type=flow_type, flow1=flow1, flow2=flow2, u_h=u_h, area=area)
        self.number_tube_passes = number_tube_passes
        self.number_shells = number_shells

    def solve_temperature(self):
        """Calculate the missing outlet temperature"""

        # When both outlet temperatures are not defined, it has to be solved iteratively.
        if self.flow1.temp_out is None and self.flow2.temp_out is None:
            # Initial guess for the outlet temperature on the hot side
            heat_transfer_prev = (self.flow_hot_side.temp_in - self.flow_cold_side.temp_in) * \
                                 self.flow_hot_side.heat_capacity_rate
            self.flow_hot_side.temp_out = self.flow_hot_side.temp_in - 0.75 * \
                                          (self.flow_hot_side.temp_in - self.flow_cold_side.temp_in)
            heat_transfer_temp = self.flow_hot_side.heat_transfer_rate
            while (abs(heat_transfer_temp - heat_transfer_prev) /
                   heat_transfer_temp > 0.001):
                heat_transfer_prev = self.flow_hot_side.heat_transfer_rate
                self.solve_temperature()
                logger.info(f"Temperature: T1 = {self.flow_hot_side.temp_in:.2f}, "
                            f"T2 = {self.flow_hot_side.temp_out:.2f}, "
                            f"t1 = {self.flow_cold_side.temp_in:.2f}, "
                            f"t2 = {self.flow_cold_side.temp_out:.2f}")
                heat_transfer_temp = self.heat_transfer_rate_from_lmtd
                logger.info(f"LMTD: {self.logarithmic_mean_temperature_difference:.2f}")
                logger.info(f"q_energy: {self.heat_transfer_rate:.2f}")
                logger.info(f"q_lmtd: {self.heat_transfer_rate_from_lmtd:.2f}")
                self.flow_hot_side.temp_out = self.flow_hot_side.temp_in - \
                                              (
                                                          0.9 * self.heat_transfer_rate + 0.1 * self.heat_transfer_rate_from_lmtd) / \
                                              self.flow_hot_side.heat_capacity_rate

        # The case when the outlet temperature is known for the hot side
        if self.flow_hot_side.temp_out is not None:
            self.flow_cold_side.temp_out = self.flow_hot_side.heat_transfer_rate / \
                                           self.flow_cold_side.heat_capacity_rate + \
                                           self.flow_cold_side.temp_in


        # The case when the outlet temperature is known for the cold side
        else:
            self.flow_hot_side.temp_out = self.flow_hot_side.temp_in - \
                                          self.flow_cold_side.heat_transfer_rate / \
                                          self.flow_hot_side.heat_capacity_rate

    @property
    def logarithmic_mean_temperature_difference(self) -> float:
        """Returns logarithmic mean temperature difference when only outlet temperature of a one
        flow is missing while rest temperature and flow conditions are known."""
        if self.flow1.temp_out is None or self.flow2.temp_out is None:
            self.solve_temperature()
        if self.is_parallel_flow is None:
            raise TypeError("Whether it is is parallel or counter flow is not specified.")
        if self.is_parallel_flow:
            delta_t2 = self.flow1.temp_out - self.flow2.temp_out
            delta_t1 = self.flow1.temp_in - self.flow2.temp_in
        else:
            delta_t2 = self.flow1.temp_out - self.flow2.temp_in
            delta_t1 = self.flow1.temp_in - self.flow2.temp_out
        return (delta_t2 - delta_t1) / np.log(delta_t2 / delta_t1)

    @property
    def heat_capacity_rate_min(self):
        return min([self.flow1.heat_capacity_rate, self.flow2.heat_capacity_rate])

    @property
    def ntu(self):
        if self.u_h is None or self.area is None:
            raise TypeError("The overall coefficient and/or area is/are note defined.")
        return self.u_h * self.area / self.heat_capacity_rate_min

    @property
    def correction_factor(self):
        """Returns correction factor for complex type"""
        if self.flow1.temp_out is None or self.flow2.temp_out is None:
            self.solve_temperature()

        # Correction factor for N-shells 2N-passes (tube)
        if self.flow_type == FlowType.ShellAndTube:
            flow_in_tube = self.flow1 if self.flow1.is_in_tube else self.flow2
            flow_in_shell = self.flow2 if self.flow1.is_in_tube else self.flow1
            r = (flow_in_shell.temp_in - flow_in_shell.temp_out) / \
                (flow_in_tube.temp_out - flow_in_tube.temp_in)
            p = (flow_in_tube.temp_out - flow_in_tube.temp_in) / \
                (flow_in_shell.temp_in - flow_in_tube.temp_in)
            if r == 1:
                w = self.number_shells * (1 - p) / (self.number_shells - self.number_shells * p + p)
                log_term = (w / (1 - w) + 0.5 ** 0.5) / (w / (1 - w) - 0.5 ** 0.5)
                if log_term <= 0:
                    raise ValueError("The flow condition is not within the valid limit for the "
                                     f"correction factor. R={r:.3f}, P={p:.3f}")
                return 2 ** 0.5 * (1 - w) / np.log(log_term)
            else:
                s = (r ** 2 + 1) ** 0.5 / (r - 1)
                w = ((1 - p * r) / (1 - p)) ** (1 / self.number_shells)
                log_term = (1 + w - s + s * w) / (1 + w + s - s * w)
                if log_term <= 0:
                    raise ValueError(f"The flow condition is not within the valid limit for the "
                                     f"correction factor. R={r:.3f}, P={p:.3f}")
                return s * np.log(w) / np.log(log_term)

        # Correction factor for cross flow
        if self.flow_type == FlowType.Cross:
            if not self.flow1.is_mixed and not self.flow2.is_mixed:
                raise NotImplementedError("The correction factor for both flows unmixed is not "
                                          "implemented")
            elif self.flow1.is_mixed and self.flow2.is_mixed:
                r = (self.flow1.temp_in - self.flow1.temp_out) / \
                    (self.flow2.temp_out - self.flow2.temp_in)
                k1 = 1 - np.exp(-self.ntu)
                k2 = 1 - np.exp(-r)
                p = 1 / (1 / k1 + r / k2 - 1 / self.ntu)
            else:
                mixed_flow = self.flow1 if self.flow1.is_mixed else self.flow2
                unmixed_flow = self.flow2 if self.flow1.is_mixed else self.flow1
                r = (mixed_flow.temp_in - mixed_flow.temp_out) / \
                    (unmixed_flow.temp_out - unmixed_flow.temp_in)
                k = 1 - np.exp(-self.ntu)
                p = (1 - np.exp(-k * r)) / r
            if r == 1:
                return p / (self.ntu * (1 - p))
            else:
                if (1 - r * p) <= 0:
                    raise ValueError(f"The flow condition is not within the valid limit for the "
                                     f"correction factor, R={r:.3f}, P={p:.3f}")
                return 1 / (self.ntu * (1 - r)) * np.log((1 - r * p) / (1 - p))

        # Return 1 for other cases such as parallel or counter flow
        return 1

    def get_heat_exchange_area(self, heat_transfer_coefficient: float = None) -> float:
        """Returns heat exchange area given the overall heat transfer coefficient"""
        if heat_transfer_coefficient is not None:
            self.u_h = heat_transfer_coefficient
        # For cross flow heat exchanger, area should be calculated iteratively as it requires area
        # value to calculate NTU.
        if self.flow_type == FlowType.Cross:
            self.area = self.heat_transfer_rate / self.u_h / \
                        self.logarithmic_mean_temperature_difference / self.correction_factor
            area_prev = self.area * 2
            iteration = 0
            while abs(area_prev - self.area) / self.area > 0.001:
                iteration += 1
                area_prev = self.area
                self.area = self.heat_transfer_rate / self.u_h / \
                            self.logarithmic_mean_temperature_difference / self.correction_factor
                if iteration > 1000:
                    warnings.warn("The iteration exceeded its predefined number (1000).",
                                  UserWarning)
        else:
            self.area = self.heat_transfer_rate / self.u_h / \
                        self.logarithmic_mean_temperature_difference / self.correction_factor
        return self.area

    def get_overall_heat_transfer_coefficient(self, heat_exchange_area: float = None) -> float:
        """Returns heat transfer coefficient given the area"""
        if heat_exchange_area is not None:
            self.area = heat_exchange_area
        # For cross flow heat exchanger, u_h should be calculated iteratively as it requires u_h
        # value to calculate NTU.
        if self.flow1.temp_out is None or self.flow2.temp_out is None:
            self.solve_temperature()
        if self.flow_type == FlowType.Cross:
            self.u_h = self.heat_transfer_rate / self.area / \
                       self.logarithmic_mean_temperature_difference
            u_h_prev = self.u_h * 2
            iteration = 0
            while abs(u_h_prev - self.u_h) / self.u_h > 0.001:
                iteration += 1
                u_h_prev = self.u_h
                self.u_h = self.heat_transfer_rate / self.area / \
                           self.logarithmic_mean_temperature_difference / self.correction_factor
                if iteration > 1000:
                    warnings.warn("The iteration exceeded its predefined number (1000).",
                                  UserWarning)
                    return self.u_h
        else:
            self.u_h = self.heat_transfer_rate / self.area / \
                       self.logarithmic_mean_temperature_difference / self.correction_factor
        return self.u_h

    @property
    def heat_transfer_rate_from_lmtd(self):
        """Returns heat transfer rate from LMTD"""
        if self.u_h is None:
            raise TypeError("Overall heat transfer coefficient is None.")
        if self.area is None:
            raise TypeError("Area is none.")
        return self.u_h * self.area * self.correction_factor * \
               self.logarithmic_mean_temperature_difference


class HeatExchangerEffectivenessNTU(HeatExchanger):
    """Class for heat exchanger analysis using effectiveness-NTU method"""

    def __init__(
            self,
            flow_type: FlowType,
            flow1: FlowHeatExchanger,
            flow2: FlowHeatExchanger,
            number_shell_passes=1,
            u_h: float = None,
            area: float = None
    ):
        super().__init__(flow_type, flow1, flow2, u_h, area)
        self.number_shell_passes = number_shell_passes

    @property
    def heat_capacity_rate_min(self):
        """Returns minimum heat capacity rate"""
        return min([self.flow1.heat_capacity_rate, self.flow2.heat_capacity_rate])

    @property
    def heat_capacity_rate_max(self):
        """Returns maximum heat capacity rate"""
        return max([self.flow1.heat_capacity_rate, self.flow2.heat_capacity_rate])

    @property
    def heat_capacity_rate_ratio(self):
        """Returns heat capacity rate ratio (C_min / C_max)"""
        return self.heat_capacity_rate_min / self.heat_capacity_rate_max

    @property
    def heat_transfer_rate_max(self):
        """Returns maximum heat transfer rate"""
        return self.heat_capacity_rate_min * abs(self.flow1.temp_in - self.flow2.temp_in)

    @property
    def ntu(self):
        """Returns NTU calculated by its definition"""
        if self.u_h is None or self.area is None:
            raise TypeError("Overall heat transfer coefficient (u_h) or heat change area (area) "
                            "is note defined. Either define these attributes or solve the "
                            "temperature using `solve_temperature` method and find these "
                            "attributes by `get_overall_heat_transfer_coefficient` or "
                            "`get_heat_exchange_area`.")
        return self.u_h * self.area / self.heat_capacity_rate_min

    def get_effectiveness_from_ntu(self, ntu: float = None):
        if ntu is None:
            ntu = self.ntu
        """Returns effectiveness given the overall heat transfer coefficient and area"""
        if self.flow_type == FlowType.Parallel:
            return (1 - np.exp(-ntu * (1 + self.heat_capacity_rate_ratio))) / \
                   (1 + self.heat_capacity_rate_ratio)
        if self.heat_capacity_rate_ratio == 0:
            return 1 - np.exp(-ntu)
        if self.flow_type == FlowType.Counter:
            if self.heat_capacity_rate_ratio < 1:
                return (1 - np.exp(-ntu * (1 - self.heat_capacity_rate_ratio))) / \
                       (1 - self.heat_capacity_rate_ratio * np.exp(
                           -ntu * (1 + self.heat_capacity_rate_ratio)))
            elif self.heat_capacity_rate_ratio == 1:
                return ntu / (1 + ntu)
            else:
                raise ValueError("The ratio of heat capacity rate cannot exceed 1.")
        elif self.flow_type == FlowType.ShellAndTube:
            term1 = (1 + self.heat_capacity_rate_ratio ** 2) ** 0.5
            effectiveness1 = 2 * (1 + self.heat_capacity_rate_ratio + term1 *
                                  (1 + np.exp(-ntu * term1)) /
                                  (1 - np.exp(-ntu * term1))) ** -1
            if self.number_shell_passes == 1:
                return effectiveness1
            else:
                term1 = (1 - effectiveness1 * self.heat_capacity_rate_ratio) / (1 - effectiveness1)
                return (term1 ** self.number_shell_passes - 1) / \
                       (term1 ** self.number_shell_passes - self.heat_capacity_rate_ratio)
        elif self.flow_type == FlowType.Cross:
            if not self.flow1.is_mixed and not self.flow2.is_mixed:
                return 1 - np.exp(1 / self.heat_capacity_rate_ratio * ntu ** 0.22 *
                                  (np.exp(-self.heat_capacity_rate_ratio * ntu ** 0.78) - 1))
            flow_c_max = self.flow1 if self.flow1.heat_capacity_rate > \
                                       self.flow2.heat_capacity_rate else self.flow2
            if flow_c_max.is_mixed:
                return 1 / self.heat_capacity_rate_ratio * (1 - np.exp(
                    -self.heat_capacity_rate_ratio * (1 - np.exp(-ntu))))
            else:
                return 1 - np.exp(-self.heat_capacity_rate_ratio ** -1 * (1 - np.exp(
                    -self.heat_capacity_rate_ratio * ntu)))

    def solve_temperature(self):
        """Solve the temperature outlet using the energy balance and effectiveness method"""
        try:
            q = self.heat_transfer_rate
        except TypeError:
            effectiveness = self.get_effectiveness_from_ntu()
            q = effectiveness * self.heat_transfer_rate_max
        self.flow_hot_side.temp_out = self.flow_hot_side.temp_in - \
                                      q / self.flow_hot_side.heat_capacity_rate
        self.flow_cold_side.temp_out = self.flow_cold_side.temp_in + \
                                       q / self.flow_cold_side.heat_capacity_rate

    def get_ntu_from_effectiveness(self, effectiveness: float = None):
        """Returns NTU from heat capacity ratio and effectiveness"""
        if effectiveness is None:
            effectiveness = self.effectiveness()
        if self.heat_capacity_rate_ratio == 0:
            return -np.log(1 - effectiveness)
        if self.flow_type == FlowType.Parallel:
            return -np.log((1 - effectiveness * (1 + self.heat_capacity_rate_ratio))) / \
                   (1 + self.heat_capacity_rate_ratio)
        if self.flow_type == FlowType.Counter:
            if self.heat_capacity_rate_ratio < 1:
                return 1 / (self.heat_capacity_rate_ratio - 1) * \
                       np.log((effectiveness - 1) /
                              (effectiveness * self.heat_capacity_rate_ratio - 1))
            if self.heat_capacity_rate_ratio == 1:
                return effectiveness / (1 - effectiveness)
        if self.flow_type == FlowType.ShellAndTube:
            if self.number_shell_passes > 1:
                term1 = ((effectiveness * self.heat_capacity_rate_ratio - 1) /
                         (effectiveness - 1)) ** (1 / self.number_shell_passes)
                effectiveness = (term1 - 1) / (term1 - self.heat_capacity_rate_ratio)
            term1 = (2 / effectiveness - (1 + self.heat_capacity_rate_ratio)) / \
                    (1 + self.heat_capacity_rate_ratio ** 2) ** 0.5
            return -self.number_shell_passes * (1 + self.heat_capacity_rate_ratio ** 2) ** -0.5 * \
                   np.log((term1 - 1) / (term1 + 1))
        if self.flow_type == FlowType.Cross:
            if not self.flow1.is_mixed and not self.flow2.is_mixed:
                def get_effectiveness_cross_flow_unmixed(ntu: float, c_r: float) -> float:
                    return 1 - np.exp(1 / c_r * ntu ** 0.22 * (np.exp(-c_r * ntu ** 0.78) - 1))

                def equation_to_solve(ntu: float, c_r: float) -> float:
                    return effectiveness - \
                           get_effectiveness_cross_flow_unmixed(ntu, c_r)

                ntu0 = self.u_h * 1 / self.heat_capacity_rate_min
                solution = root(equation_to_solve, [ntu0], args=(self.heat_capacity_rate_ratio,))
                return solution.x[0]

            flow_c_max = self.flow1 if self.flow1.heat_capacity_rate > \
                                       self.flow2.heat_capacity_rate else self.flow2
            if flow_c_max.is_mixed:
                return -np.log(1 + (1 / self.heat_capacity_rate_ratio) *
                               np.log(1 - effectiveness * self.heat_capacity_rate_ratio))
            else:
                return -(1 / self.heat_capacity_rate_ratio) * np.log(
                    self.heat_capacity_rate_ratio * np.log(1 - effectiveness) + 1)

    @property
    def effectiveness(self):
        """Returns effectiveness by its definition given that the temperature has been
        defined or solved"""
        return self.heat_transfer_rate / self.heat_capacity_rate_max

    def get_overall_heat_transfer_coefficient(self, area: float):
        """Returns overall heat transfer coefficient given the area"""
        self.area = area
        self.solve_temperature()
        ntu = self.get_ntu_from_effectiveness(self.effectiveness)
        return ntu / self.area * self.heat_capacity_rate_min

    def get_heat_exchange_area(self, heat_transfer_coefficient: float):
        """Returns overall heat transfer coefficient given the heat transfer coefficient"""
        self.u_h = heat_transfer_coefficient
        self.solve_temperature()
        ntu = self.get_ntu_from_effectiveness(self.effectiveness)
        return ntu / self.u_h * self.heat_capacity_rate_min


###################################################################################################
# Exercise 4
NUMBER_PROPERTIES_TO_DETERMINE_STATE = 3


class HumidAirState(NamedTuple):
    pressure_pa: float
    temperature_k: float
    partial_pressure_water_vapor_pa: float
    specific_humidity_kg_per_kg_dry_air: float
    relative_humidity: float
    specific_enthalpy_j_per_kg_humid_air: float
    specific_enthalpy_j_per_kg_dry_air: float
    specific_volume_m3_per_kg_humid_air: float
    dew_point_temperature_k: float
    wet_bulb_temperature_k: float


humid_air_property_names = {
    'pressure_pa': 'P',
    'partial_pressure_water_vapor_pa': 'P_w',
    'temperature_k': 'T',
    'specific_humidity_kg_per_kg_dry_air': 'W',
    'relative_humidity': 'R',
    'specific_enthalpy_j_per_kg_humid_air': 'Hha',
    'specific_enthalpy_j_per_kg_dry_air': 'Hda',
    'dew_point_temperature_k': 'D',
    'specific_volume_m3_per_kg_humid_air': 'Vha',
    'wet_bulb_temperature_k': 'B'
}


def get_humid_air_state(
        pressure_pa: float = None,
        partial_pressure_water_vapor_pa: float = None,
        temperature_k: float = None,
        specific_humidity_kg_per_kg_dry_air: float = None,
        relative_humidity: float = None,
        specific_enthalpy_j_per_kg_humid_air: float = None,
        specific_enthalpy_j_per_kg_dry_air: float = None,
        dew_point_temperature_k: float = None,
        specific_volume_m3_per_kg_humid_air: float = None,
        wet_bulb_temperature_k: float = None
):
    # Use only three inputs
    arguments = locals()
    arguments_to_be_used = [key for key in arguments if arguments[key] is not None]
    arguments_to_be_used = arguments_to_be_used[:3]

    # Check if we have enough inputs
    if len(arguments) < NUMBER_PROPERTIES_TO_DETERMINE_STATE:
        raise TypeError(f"You need to provide at least {NUMBER_PROPERTIES_TO_DETERMINE_STATE} "
                        f"inputs to determine the thermodynamic state of the humid air")

    # Create inputs for CoolProp
    unknown_names = [humid_air_property_names[key] for key in humid_air_property_names]
    arguments_for_cool_prop = []
    for argument in arguments_to_be_used:
        known_name = unknown_names.pop(unknown_names.index(humid_air_property_names[argument]))
        arguments_for_cool_prop.extend([known_name, arguments[argument]])

    # Get the unknowns
    for var_name in unknown_names:
        args = [var_name, *arguments_for_cool_prop]
        prop = HAPropsSI(*args)
        arg_name = next(filter(
            lambda key: humid_air_property_names[key] == var_name, 
            humid_air_property_names))
        arguments[arg_name] = prop
    
    return HumidAirState(**arguments)
