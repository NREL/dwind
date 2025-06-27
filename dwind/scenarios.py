import json
from pathlib import Path

import pandas as pd


def config_nem(scenario, year):
    # NEM_SCENARIO_CSV
    nem_opt_scens = ["highrecost", "lowrecost", "re100"]
    # nem_opt_scens = ['der_value_HighREcost', 'der_value_LowREcost', 're_100']
    if scenario in nem_opt_scens:
        nem_scenario_csv = "nem_optimistic_der_value_2035.csv"
    elif scenario == "baseline" and year in (2022, 2025, 2035):
        nem_scenario_csv = f"nem_baseline_{year}.csv"
    else:
        nem_scenario_csv = "nem_baseline_2035.csv"

    return nem_scenario_csv


def config_cambium(scenario):
    # CAMBIUM_SCENARIO
    if scenario == "highrecost" or scenario == "re100":
        cambium_scenario = "StdScen20_HighRECost"
    elif scenario == "lowrecost":
        cambium_scenario = "StdScen20_LowRECost"
    else:
        # cambium_scenario = "StdScen20_MidCase"
        cambium_scenario = "Cambium23_MidCase"

    return cambium_scenario


def config_costs(scenario, year):
    # COST_INPUTS
    f = Path(f"/projects/dwind/configs/costs/atb24/ATB24_costs_{scenario}_{year}.json").resolve()
    with f.open("r") as f_in:
        cost_inputs = json.load(f_in)

    return cost_inputs


def config_performance(scenario, year):
    # PERFORMANCE_INPUTS
    if scenario == "baseline" and year == 2022:
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


def config_financial(scenario, year):
    # FINANCIAL_INPUTS
    scenarios = ("baseline", "metering", "billing")
    if scenario in scenarios and year == 2025:
        f = f"/projects/dwind/configs/costs/atb24/ATB24_financing_baseline_{year}.json"
        i = Path("/projects/dwind/data/incentives/2025_incentives.json").resolve()
        with i.open("r") as i_in:
            incentives = json.load(i_in)
    elif scenario in scenarios and year in (2035, 2040):
        f = "/projects/dwind/configs/costs/atb24/ATB24_financing_baseline_2035.json"
    else:
        # use old assumptions
        f = "/projects/dwind/configs/costs/atb20/ATB20_financing_baseline_2035.json"
    f = Path(f).resolve()

    with f.open("r") as f_in:
        financials = json.load(f_in)
    if year == 2025:
        financials["BTM"]["itc_fraction_of_capex"] = incentives
        financials["FOM"]["itc_fraction_of_capex"] = incentives
        financials["FOM"]["ptc_fed_dlrs_per_kwh"]["solar"] = 0.0
        financials["FOM"]["ptc_fed_dlrs_per_kwh"]["wind"] = 0.0
    else:
        financials["BTM"]["itc_fraction_of_capex"] = 0.3
        financials["FOM"]["itc_fraction_of_capex"] = 0.3
        financials["FOM"]["ptc_fed_dlrs_per_kwh"]["solar"] = 0.0
        financials["FOM"]["ptc_fed_dlrs_per_kwh"]["wind"] = 0.0

    return financials
