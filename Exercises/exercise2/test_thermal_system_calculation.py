import logging
import random

from CoolProp.CoolProp import PropsSI
import numpy as np
import pytest

from thermal_system_calculation import get_heat_flux_by_conduction, \
    get_heat_flux_by_convection, \
    get_thermal_resistance_cylinder, TipBoundaryConditionForFin, \
    get_heat_transfer_1D_fin_with_uniform_cross_section, FinConfiguration, TubeBankConfiguration, \
    TubeBankArrangement, FluidState, Fluid, Properties, ThermodynamicState, \
    ForcedConvectionFlatPlateIsothermSurface, ForcedConvectionCylinderCrossFlow, \
    NonCircularCylinderGeometry, ForcedConvectionNonCircularCylinderCrossFlow, \
    ForcedConvectionTubeBanksCrossFlow

# Define logger
logger = logging.getLogger('test_thermal_system_calculation')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)


@pytest.fixture
def temp_from_random():
    return 273.15 + 1000 * random.random()


@pytest.fixture
def temp_to_random(temp_from_random):
    return temp_from_random * random.random()


@pytest.fixture
def thermal_conductivity_random():
    return random.random() * 200


@pytest.fixture
def convection_heat_transfer_coeff_random():
    return random.random() * 200


def test_get_heat_flux_by_conduction(
        temp_from_random,
        temp_to_random,
        thermal_conductivity_random
):
    """Test get_heat_flux_by_conduction method"""
    thickness = random.random()
    assert get_heat_flux_by_conduction(
        temperature_from=temp_from_random,
        temperature_to=temp_to_random,
        thickness=thickness,
        thermal_conductivity=thermal_conductivity_random
    ) == thermal_conductivity_random * (temp_from_random - temp_to_random) / thickness, \
        "The function output is not valid"


def test_get_flux_by_convection(
        temp_from_random,
        temp_to_random,
        convection_heat_transfer_coeff_random
):
    """Test get_heat_flux_by_convection method"""
    assert get_heat_flux_by_convection(
        h=convection_heat_transfer_coeff_random,
        temp_from=temp_from_random,
        temp_to=temp_to_random
    ) == convection_heat_transfer_coeff_random * (temp_from_random - temp_to_random), \
        "The function output is not valid"


def test_get_thermal_resistance_cylinder(thermal_conductivity_random):
    """Test get_heat_flux_by_convection method"""
    radius = random.random()
    thickness = random.random()
    length = random.random()
    outer_radius = radius + thickness
    assert get_thermal_resistance_cylinder(
        radius=radius,
        wall_thickness=thickness,
        thermal_conductivity=thermal_conductivity_random,
        length=length
    ) == np.log(outer_radius / radius) / (2 * np.pi * length * thermal_conductivity_random), \
        "The function output is not valid"


def test_get_heat_transfer_1D_fin_with_uniform_cross_section():
    """Test get_heat_transfer_1D_fin_with_uniform_cross_section function"""
    parameters = {
        'x': 0.02,
        'temp_base': 500,
        'temp_surr': 300,
        'fin_configuration': FinConfiguration(perimeter=0.032, area_cross_section=0.00012,
                                              length=0.05),
        'h': 100,
        'k': 180,
    }
    temp_tip = 375

    answers = [(30.2095, 377.706), (28.5584, 379.250), (66.3291, 343.931), (52.5814, 356.787)]

    for boundary_condition, answer_pair in zip(TipBoundaryConditionForFin, answers):
        parameters['boundary_condition'] = boundary_condition
        if boundary_condition == TipBoundaryConditionForFin.Temperature:
            parameters['temp_tip'] = temp_tip
        value_pair_to_test = get_heat_transfer_1D_fin_with_uniform_cross_section(**parameters)
        print(value_pair_to_test)
        for value, answer in zip(value_pair_to_test, answer_pair):
            assert np.isclose(value, answer), \
                f"Your value {value:.5e} for {boundary_condition} is not close to the answer, " \
                f"{answer:.5e}"


@pytest.fixture
def velocity_random() -> float:
    return random.random() * 10


def test_get_maximum_velocity_for_tube_bank_configuration(velocity_random):
    """Test get_maximum_velocity_for_tube_bank in TubeBankConfiguration class"""
    # First generate some random numbers for the inputs
    vertical_spacing = random.random()
    diameter = vertical_spacing * random.random()

    # Calculate the horizontal spacing for two cases, narrow and wide
    mean_length_for_vertical_spacing_diameter = np.mean([vertical_spacing, diameter])
    diagonal_spacing_narrow = np.max(
        [mean_length_for_vertical_spacing_diameter * 0.9, vertical_spacing / 2 * 1.001])
    diagonal_spacing_wide = mean_length_for_vertical_spacing_diameter * 1.1
    horizontal_spacing_narrow = np.sqrt(
        diagonal_spacing_narrow ** 2 - (0.5 * vertical_spacing) ** 2)
    horizontal_spacing_wide = np.sqrt(
        diagonal_spacing_wide ** 2 - (0.5 * vertical_spacing) ** 2)

    # Compare the results
    for arrangement_to_test in TubeBankArrangement:
        tube_bank_config_narrow = TubeBankConfiguration(
            arrangement=arrangement_to_test,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing_narrow,
            number_rows=random.randint(5, 10),
            number_tubes_each_row=10,
        )
        tube_bank_config_wide = TubeBankConfiguration(
            arrangement=arrangement_to_test,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing_wide,
            number_rows=random.randint(5, 10),
            number_tubes_each_row=10
        )
        velocity = velocity_random
        err_msg = f"The maximum velocity for {arrangement_to_test} is not valid."
        if arrangement_to_test == TubeBankArrangement.Aligned:
            logger.info("Testing the maximum velocity for banks of tubes, aligned")
            assert np.isclose(
                tube_bank_config_narrow.get_maximum_velocity(velocity, diameter),
                vertical_spacing / (vertical_spacing - diameter) * velocity), err_msg
            assert np.isclose(
                tube_bank_config_wide.get_maximum_velocity(velocity, diameter),
                vertical_spacing / (vertical_spacing - diameter) * velocity), err_msg
        else:
            logger.info("Testing the maximum velocity for banks of tubes, staggered")
            assert np.isclose(
                tube_bank_config_wide.get_maximum_velocity(velocity, diameter),
                vertical_spacing / (vertical_spacing - diameter) * velocity), err_msg
            assert np.isclose(
                tube_bank_config_narrow.get_maximum_velocity(velocity, diameter),
                vertical_spacing / (2 * (diagonal_spacing_narrow - diameter)) * velocity), err_msg


@pytest.fixture
def fluid_random() -> Fluid:
    return Fluid[random.choice(list(Fluid.__members__))]


@pytest.fixture
def characteristic_length_random() -> float:
    return random.random() * 10


@pytest.fixture
def pressure_random() -> float:
    return 101325 + random.random() * 10 ** 6


@pytest.fixture
def fluid_state(fluid_random, characteristic_length_random, pressure_random):
    return FluidState(
        fluid=fluid_random,
        pressure_pa=pressure_random,
        temp_k=300 + random.random() * 100,
        characteristic_length=characteristic_length_random
    )


def test_fluid_state_properties(fluid_state: FluidState):
    assert fluid_state.k == PropsSI(
        Properties.THERMAL_CONDUCTIVITY.value,
        ThermodynamicState.PRESSURE.value, fluid_state.pressure_pa,
        ThermodynamicState.TEMPERATURE.value, fluid_state.temp_k,
        fluid_state.fluid
    ), "The property value for k is not valid."

    assert fluid_state.dynamic_viscosity == PropsSI(
        Properties.DYNAMIC_VISCOSITY.value,
        ThermodynamicState.PRESSURE.value, fluid_state.pressure_pa,
        ThermodynamicState.TEMPERATURE.value, fluid_state.temp_k,
        fluid_state.fluid
    ), "The property value for dynamic viscosity is not valid."

    assert fluid_state.density == PropsSI(
        Properties.DENSITY.value,
        ThermodynamicState.PRESSURE.value, fluid_state.pressure_pa,
        ThermodynamicState.TEMPERATURE.value, fluid_state.temp_k,
        fluid_state.fluid
    ), "The property value for density is not valid."

    assert fluid_state.cp == PropsSI(
        Properties.SPECIFIC_HEAT_CAPACITY_CONSTANT_PRESSURE.value,
        ThermodynamicState.PRESSURE.value, fluid_state.pressure_pa,
        ThermodynamicState.TEMPERATURE.value, fluid_state.temp_k,
        fluid_state.fluid
    ), "The property value for specific heat capacity is not valid."


def test_fluid_state_dimensionless_numbers(
        fluid_state: FluidState,
        convection_heat_transfer_coeff_random,
):
    assert np.isclose(
        fluid_state.prantdl_number,
        fluid_state.cp * fluid_state.dynamic_viscosity / fluid_state.k
    ), "Prantdl number is not valid"

    nu_number = fluid_state.get_nusselt_number(convection_heat_transfer_coeff_random)
    assert np.isclose(
        fluid_state.get_nusselt_number(convection_heat_transfer_coeff_random),
        convection_heat_transfer_coeff_random * fluid_state.characteristic_length / fluid_state.k
    ), "Nusselt number is not valid"

    assert np.isclose(
        fluid_state.get_h_from_nusselt_number(nu=nu_number),
        convection_heat_transfer_coeff_random
    ), "The convection heat transfer coefficient is not valid"

    velocity = random.random() * 10 ** 6
    re_number = fluid_state.density * velocity * fluid_state.characteristic_length / \
                fluid_state.dynamic_viscosity
    assert np.isclose(
        fluid_state.get_reynolds_number(velocity), re_number), "Reynolds number is not valid"


def test_forced_convection_flat_plate(
        temp_from_random,
        temp_to_random,
        fluid_random,
        characteristic_length_random,
        pressure_random,
        velocity_random
):
    convection = ForcedConvectionFlatPlateIsothermSurface(
        velocity_random,
        temp_surface=temp_from_random,
        temp_infinity=temp_to_random,
        fluid=fluid_random,
        characteristic_length=characteristic_length_random,
        pressure_pa=pressure_random
    )

    assert convection.prantdl_number == convection.fluid_state.prantdl_number, \
        "The Prandtl number is not valid"

    assert np.isclose(
        convection.fluid_state.temp_k,
        np.mean([temp_to_random, temp_from_random])), \
        "The temperature for properties should be at the film."

    assert np.isclose(
        convection.reynolds_number,
        convection.fluid_state.get_reynolds_number(convection.velocity)), \
        "The Reynolds number is not valid"

    assert np.isclose(
        convection.critical_length,
        convection._REYNOLDS_NUMBER_AT_TRANSITION * convection.fluid_state.dynamic_viscosity / \
        (convection.fluid_state.density * convection.velocity)), \
        "The value of the critical length is not valid. "

    if convection.critical_length_ratio < 0.95:
        assert np.isclose(
            convection.nusselt_number,
            (0.037 * convection.reynolds_number ** 0.8 - 871) * convection.prantdl_number ** (
                        1 / 3)), \
            "The Nusselt number is not valid for the turbulent case."
    else:
        assert np.isclose(
            convection.nusselt_number,
            0.664 * convection.reynolds_number ** 0.5 * convection.prantdl_number ** (1 / 3)), \
            "The Nusselt number is not valid for the laminar case."

    h_correct = convection.nusselt_number * convection.fluid_state.k / \
                convection.fluid_state.characteristic_length
    assert np.isclose(convection.h, h_correct), "The value of convection heat transfer " \
                                                "coefficient is invalid."


def test_forced_convection_cylinder_cross_flow(
        velocity_random,
        temp_from_random,
        temp_to_random,
        fluid_random,
        characteristic_length_random,
        pressure_random
):
    """Tests Nusselt number calculation for forced convection for a circular cylinder in cross
    flow"""
    convection = ForcedConvectionCylinderCrossFlow(
        velocity_random,
        temp_surface=temp_from_random,
        temp_infinity=temp_to_random,
        fluid=fluid_random,
        characteristic_length=characteristic_length_random,
        pressure_pa=pressure_random
    )

    assert np.isclose(
        convection.nusselt_number,
        0.3 + (0.62 * convection.reynolds_number ** 0.5 * convection.prantdl_number ** (1 / 3)) / \
        (1 + (0.4 / convection.prantdl_number) ** (2 / 3)) ** 0.25 *
        (1 + (convection.reynolds_number / 282000) ** (5 / 8)) ** 0.8), \
        "The Nusselt number is not valid."


def test_forced_convection_non_circular_cylinder_cross_flow(
        velocity_random,
        temp_from_random,
        temp_to_random,
        fluid_random,
        characteristic_length_random,
        pressure_random
):
    """Test Nusselt number calculation for forced convection for a non-circular cylinder in cross
    flow"""
    for geometry in NonCircularCylinderGeometry:
        convection = ForcedConvectionNonCircularCylinderCrossFlow(
            geometry,
            velocity=velocity_random,
            temp_surface=temp_from_random,
            temp_infinity=temp_to_random,
            fluid=fluid_random,
            characteristic_length=characteristic_length_random,
            pressure_pa=pressure_random
        )
        if geometry == NonCircularCylinderGeometry.SQUARE_FLAT:
            c, m = 0.304, 0.59
        elif geometry == NonCircularCylinderGeometry.SQUARE_OBLIQUE:
            c, m = 0.158, 0.66
        elif geometry == NonCircularCylinderGeometry.HEXAGON_FLAT:
            if convection.reynolds_number < 20400:
                c, m = 0.164, 0.638
            else:
                c, m = 0.039, 0.78
        elif geometry == NonCircularCylinderGeometry.HEXAGON_OBLIQUE:
            c, m = 0.15, 0.638
        elif geometry == NonCircularCylinderGeometry.VERTICAL_FLAT_FRONT:
            c, m = 0.667, 0.500
        elif geometry == NonCircularCylinderGeometry.VERTICAL_FLAT_BACK:
            c, m = 0.191, 0.667
        else:
            raise TypeError("Geometry is not valid")
        nusselt_number_correct = c * convection.reynolds_number ** m * \
                                 convection.prantdl_number ** (1 / 3)
        assert np.isclose(convection.nusselt_number, nusselt_number_correct), \
            f"The Nusselt number is not valid (C={c}, m={m}, {geometry})."


@pytest.fixture
def tube_bank_configuration_random() -> TubeBankConfiguration:
    vertical_spacing = random.random()
    horizontal_spacing = random.random()
    return TubeBankConfiguration(
        arrangement=random.choice(list(TubeBankArrangement)),
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        number_rows=random.randint(5, 50),
        number_tubes_each_row=30
    )


def test_forced_convection_tube_banks_cross_flow(
        velocity_random: float,
        temp_from_random: float,
        temp_to_random: float,
        fluid_random: Fluid,
        pressure_random: float,
        tube_bank_configuration_random: TubeBankConfiguration,
):
    """Test ForcedConvectionTubeBanksCrossFlow class"""
    tube_diameter = np.min([tube_bank_configuration_random.vertical_spacing,
                            tube_bank_configuration_random.horizontal_spacing]) * random.random()
    convection = ForcedConvectionTubeBanksCrossFlow(
        configuration=tube_bank_configuration_random,
        velocity=velocity_random,
        temp_surface=temp_from_random,
        temp_infinity=temp_to_random,
        fluid=fluid_random,
        characteristic_length=tube_diameter,
        pressure_pa=pressure_random
    )

    # Test corrction factor
    if convection.configuration.number_rows >= 20:
        correction_factor = 1
    else:
        number_rows_ref = np.array([1, 2, 3, 4, 5, 7, 10, 13, 16])
        if convection.configuration.arrangement == TubeBankArrangement.Aligned:
            corr_factor_ref = np.array([0.7, 0.8, 0.86, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99])
        else:
            corr_factor_ref = np.array([0.64, 0.76, 0.84, 0.89, 0.92, 0.95, 0.97, 0.98, 0.99])
        correction_factor = np.interp(
            convection.configuration.number_rows, number_rows_ref, corr_factor_ref)
    assert np.isclose(convection.correction_factor, correction_factor), \
        "The correction factor is not valid."

    # Test reynolds number
    max_velocity = convection.configuration.get_maximum_velocity(
        convection.velocity, convection.fluid_state.characteristic_length
    )
    re_number_correct = convection.fluid_state.get_reynolds_number(max_velocity)
    assert np.isclose(convection.reynolds_number, re_number_correct), \
        "The Reynolds number is not valid"

    # Test Prantdl number at the surface
    fluid_state_surface = FluidState(
        fluid=Fluid[convection.fluid_state.fluid],
        pressure_pa=convection.fluid_state.pressure_pa,
        temp_k=convection.temp_surface,
        characteristic_length=convection.fluid_state.characteristic_length
    )
    assert np.isclose(convection.prantdl_number_surface, fluid_state_surface.prantdl_number), \
        "The Prantdl number at the surface is not valid"

    # Test Nusselt number
    if convection.reynolds_number < 100:
        if convection.configuration.arrangement == TubeBankArrangement.Aligned:
            c, m = 0.8, 0.4
        else:
            c, m = 0.9, 0.4
    elif convection.reynolds_number < 1000:
        convection_less_than_1000 = ForcedConvectionCylinderCrossFlow(
            velocity=convection.velocity,
            temp_surface=convection.temp_surface,
            temp_infinity=convection.temp_infinity,
            fluid=convection.fluid_state.fluid,
            characteristic_length=convection.fluid_state.characteristic_length,
            pressure_pa=convection.fluid_state.pressure_pa
        )
    elif convection.reynolds_number < 200000:
        if convection.configuration.arrangement == TubeBankArrangement.Aligned:
            c, m = 0.27, 0.63
        else:
            ratio = convection.configuration.vertical_spacing / \
                    convection.configuration.horizontal_spacing
            if ratio < 2:
                c, m = 0.35 * ratio ** 0.2, 0.6
            else:
                c, m = 0.4, 0.6
    else:
        if convection.configuration.arrangement == TubeBankArrangement.Aligned:
            c, m = 0.021, 0.84
        else:
            c, m = 0.022, 0.84
    if convection.reynolds_number < 1000 and convection.reynolds_number > 100:
        assert np.isclose(
            convection_less_than_1000.nusselt_number * convection.correction_factor,
            convection.nusselt_number
        ), "The Nusselt number is not valid"
    else:
        assert np.isclose(
            convection.nusselt_number,
            convection.correction_factor * c * convection.reynolds_number**m *
            convection.prantdl_number**0.36 *
            (convection.prantdl_number / convection.prantdl_number_surface)**0.25
        )

    # Test temp_out
    temp_out = convection.temp_surface - np.exp(
        -1 * np.pi * convection.fluid_state.characteristic_length *
        convection.configuration.number_tubes_total * convection.h /
        (convection.fluid_state.density * convection.velocity *
         convection.configuration.number_tubes_each_row *
         convection.configuration.vertical_spacing * convection.fluid_state.cp)
    ) * (convection.temp_surface - convection.temp_infinity)
    assert convection.temp_out == temp_out, "The value of the temp_out is not correct."

    # Test log_mean_temperature_difference
    log_mean_temperature_difference = (convection.temp_out - convection.temp_infinity) / \
        np.log(
            (convection.temp_surface - convection.temp_infinity) /
            (convection.temp_surface - convection.temp_out)
       )
    assert convection.log_mean_temperature_difference == log_mean_temperature_difference, \
        "The value fo the log_mean_temperature_difference is not correct."

    # Test heat transfer rate
    q = convection.configuration.number_tubes_total * convection.h * np.pi * \
        convection.fluid_state.characteristic_length * convection.log_mean_temperature_difference
    assert np.isclose(convection.get_heat_transfer_rate(1), q), \
        "The value of the heat transfer rate is not correct."


if __name__ == '__main__':
    pytest.main()
