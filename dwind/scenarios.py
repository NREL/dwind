"""Provides the scenario-specific mapping for varying financial and model configuration data."""

import json
from pathlib import Path

import pandas as pd

from dwind.config import Year, Scenario, Configuration, IncentiveScenario


def config_nem(scenario: Scenario, year: Year) -> str:
    """Provides NEM configuration based on :py:attr:`scenario` and :py:attr:`year`.

    Args:
        scenario (:py:class:`dwind.config.Scenario`): Valid :py:class:`dwind.config.Scenario`.
        year (:py:class:`dwind.config.Year`): Valid :py:class:`dwind.config.Year`.

    Returns:
        str: Name of the NEM scenario file to use.
    """
    if scenario in (Scenario.HIGHRECOST, Scenario.LOWRECOST, Scenario.RE100):
        return "nem_optimistic_der_value_2035.csv"

    if scenario is Scenario.BASELINE and year in (Year._2022, Year._2025, Year._2035):
        return f"nem_baseline_{year.value}.csv"

    return "nem_baseline_2035.csv"


def config_cambium(scenario: Scenario) -> str:
    """Loads the cambium configuration name based on :py:attr:`scenario`.

    Args:
        scenario (:py:class:`dwind.config.Scenario`): Valid :py:class:`dwind.config.Scenario`.

    Returns:
        str: Name of the Cambium scenario to use.
    """
    if scenario in (Scenario.HIGHRECOST, Scenario.RE100):
        return "StdScen20_HighRECost"

    if scenario is Scenario.LOWRECOST:
        return "StdScen20_LowRECost"

    return "Cambium24_MidCase"


def config_costs(scenario: Scenario, year: Year, config: Configuration) -> dict:
    """Loads the cost configuration based on the ATB analysis.

    Args:
        scenario (:py:class:`dwind.config.Scenario`): Valid :py:class:`dwind.config.Scenario`.
        year (:py:class:`dwind.config.Year`): Valid :py:class:`dwind.config.Year`.
        config (dwind.config.Configuration): Model configuration with universal settings.

    Returns:
        dict: Dictionary of ATB assumptions to be used for PySAM's cost inputs.
    """
    fstr = f"ATB24_costs_{scenario.value}_{year.value}.json"
    f = Path(__file__).resolve().parent.parent / "data" / fstr
    with f.open("r") as f_in:
        cost_inputs = json.load(f_in)

    return cost_inputs


def config_performance(scenario: Scenario, year: Year) -> pd.DataFrame:
    """Loads the technology performance configurations.

    Args:
        scenario (:py:class:`dwind.config.Scenario`): Valid :py:class:`dwind.config.Scenario`.
        year (:py:class:`dwind.config.Year`): Valid :py:class:`dwind.config.Year`.

    Returns:
        pd.DataFrame: Performance data based on the scale of each technology.
    """
    if scenario is Scenario.BASELINE and year is Year._2022:
        performance_inputs = {
            "solar": pd.DataFrame(
                [
                    ["res", 0.017709659, 0.005],
                    ["com", 0.017709659, 0.005],
                    ["ind", 0.017709659, 0.00],
                ],
                columns=["sector_abbr", "pv_kw_per_sqft", "pv_degradation_factor"],
            ),
            "wind": pd.DataFrame(
                [
                    [2.5, 0.083787756, 0.85],
                    [5.0, 0.083787756, 0.85],
                    [10.0, 0.083787756, 0.85],
                    [20.0, 0.083787756, 0.85],
                    [50.0, 0.116657183, 0.85],
                    [100.0, 0.116657183, 0.85],
                    [250.0, 0.106708234, 0.85],
                    [500.0, 0.106708234, 0.85],
                    [750.0, 0.106708234, 0.85],
                    [1000.0, 0.106708234, 0.85],
                    [1500.0, 0.106708234, 0.85],
                ],
                columns=["wind_turbine_kw_btm", "perf_improvement_factor", "wind_derate_factor"],
            ),
        }
    else:
        performance_inputs = {
            "solar": {
                "pv_kw_per_sqft": {"res": 0.021677397, "com": 0.021677397, "ind": 0.021677397},
                "pv_degradation_factor": {"res": 0.005, "com": 0.005, "ind": 0.005},
            },
            "wind": {
                "perf_improvement_factor": {
                    2.5: 0.23136759,
                    5.0: 0.23136759,
                    10.0: 0.23136759,
                    20.0: 0.23136759,
                    50.0: 0.23713196,
                    100.0: 0.23713196,
                    250.0: 0.23617185,
                    500.0: 0.23617185,
                    750.0: 0.23617185,
                    1000.0: 0.23617185,
                    1500.0: 0.23617185,
                },
                "wind_derate_factor": {
                    2.5: 0.85,
                    5.0: 0.85,
                    10.0: 0.85,
                    20.0: 0.85,
                    50.0: 0.85,
                    100.0: 0.85,
                    250.0: 0.85,
                    500.0: 0.85,
                    750.0: 0.85,
                    1000.0: 0.85,
                    1500.0: 0.85,
                },
            },
        }

    return performance_inputs


def config_financial(
    scenario: Scenario, inc_scenario: IncentiveScenario, year: Year, config: Configuration
) -> dict:
    """Loads the financial configuration based on the ATB analysis.

    Args:
        scenario (:py:class:`dwind.config.Scenario`): Valid :py:class:`dwind.config.Scenario`.
        inc_scenario (:py:class:`dwind.config.IncentiveScenario`): Valid
            :py:class:`dwind.config.IncentiveScenario`.
        year (:py:class:`dwind.config.Year`): Valid :py:class:`dwind.config.Year`.
        config (dwind.config.Configuration): Model configuration with universal settings.

    Returns:
        dict: Dictionary of ATB assumptions to be used for configuration PySAM.
    """
    cost_dir = Path(__file__).resolve().parent.parent / "data"
    if year is Year._2025:
        f = cost_dir / f"ATB24_financing_baseline_{year}.json"
        incentives = pd.read_parquet(
            f"{config.incentives.DIR}/{config.incentives.TABLE}", dtype_backend="pyarrow"
        )
    elif year in (Year._2035, Year._2040):
        f = cost_dir / "ATB24_financing_baseline_2035.json"
    else:
        # use old assumptions
        f = cost_dir / "ATB20_financing_baseline_2035.json"
    f = Path(f).resolve()

    with f.open("r") as f_in:
        financials = json.load(f_in)

    # TODO: determine if shared settings is applicable going forward, or separate should be reserved
    if year == 2025:
        financials["BTM"]["itc_fraction_of_capex"] = incentives
        financials["FOM"]["itc_fraction_of_capex"] = incentives
        financials["FOM"]["ptc_fed_dlrs_per_kwh"]["solar"] = 0.0
        financials["FOM"]["ptc_fed_dlrs_per_kwh"]["wind"] = 0.0
    else:
        if inc_scenario is IncentiveScenario.NOINCENTIVES:
            financials["BTM"]["itc_fraction_of_capex"] = 0.0
            financials["FOM"]["itc_fraction_of_capex"] = 0.0
        else:
            financials["BTM"]["itc_fraction_of_capex"] = 0.3
            financials["FOM"]["itc_fraction_of_capex"] = 0.3
        financials["FOM"]["ptc_fed_dlrs_per_kwh"]["solar"] = 0.0
        financials["FOM"]["ptc_fed_dlrs_per_kwh"]["wind"] = 0.0

    return financials
