import time
import logging
import functools
import concurrent.futures as cf

import h5py as h5
import numpy as np
import pandas as pd
import PySAM.Battery as battery
import PySAM.Cashloan as cashloan
import PySAM.BatteryTools as battery_tools
import PySAM.Utilityrate5 as ur5
import PySAM.Merchantplant as mp
from scipy import optimize

from dwind import Configuration, loader, scenarios


log = logging.getLogger("dwfs")


class ValueFunctions:
    def __init__(
        self,
        scenario: str,
        year: int,
        configuration: Configuration,
        return_format="totals",
    ):
        """Primary model calculation engine responsible for the computation of individual agents.

        Args:
            scenario (str): Only option is "baseline" currently.
            year (int): Analysis year.
            configuration (dwind.config.Configuration): Model configuration with universal settings.
            return_format ['profiles', 'total_profile', 'totals', 'total'] -
                Return individual value stream 8760s, a cumulative value stream 8760,
                annual totals for each value stream, or a single cumulative total
        """
        self.scenario = scenario
        self.year = year
        self.config = configuration
        self.return_format = return_format

        self.CAMBIUM_SCENARIO = scenarios.config_cambium(self.scenario)
        self.COST_INPUTS = scenarios.config_costs(self.scenario, self.year)
        self.PERFORMANCE_INPUTS = scenarios.config_performance(self.scenario, self.year)
        self.FINANCIAL_INPUTS = scenarios.config_financial(self.scenario, self.year)

        self.load()

    def load(self):
        _load_csv = functools.partial(loader.load_df, year=self.year)
        _load_sql = functools.partial(
            loader.load_df,
            year=self.year,
            sql_constructur=self.confg.sql.ATLAS_PG_CON_STR,
        )

        self.retail_rate_inputs = _load_csv(self.config.cost.RETAIL_RATE_INPUT_TABLE)
        self.wholesale_rate_inputs = _load_csv(self.config.cost.WHOLESALE_RATE_INPUT_TABLE)
        self.depreciation_schedule_inputs = _load_csv(self.config.cost.DEPREC_INPUTS_TABLE)

        if "wind" in self.config.project.settings.TECHS:
            self.wind_price_inputs = _load_sql(self.config.cost.WIND_PRICE_INPUT_TABLE)
            self.wind_tech_inputs = _load_sql(self.config.cost.WIND_TECH_INPUT_TABLE)
            self.wind_derate_inputs = _load_sql(self.config.cost.WIND_DERATE_INPUT_TABLE)

        if "solar" in self.config.project.settings.TECHS:
            self.pv_price_inputs = _load_sql(self.config.cost.PV_PRICE_INPUT_TABLE)
            self.pv_tech_inputs = _load_sql(self.config.cost.PV_TECH_INPUT_TABLE)
            self.pv_plus_batt_price_inputs = _load_sql(
                self.config.cost.PV_PLUS_BATT_PRICE_INPUT_TABLE
            )

        self.batt_price_inputs = _load_sql(self.config.cost.BATT_PRICE_INPUT_TABLE)
        self.batt_tech_inputs = _load_sql(self.config.cost.BATT_TECH_INPUT_TABLE)

    def _process_btm_costs(self, cost_inputs: dict, tech: str) -> pd.DataFrame:
        """Convert the BTM dictionary data into a dataframe.

        Args:
            cost_inputs (dict): The BTM portion of the ATB cost dictionary.
            tech (str): One of "wind" or "solar".

        Returns:
            (pd.DataFrame): A reformatted data frame of the dictionary with either a
                "sector_abbr" column or "wind_turbine_kw_btm" for joining on the agent
                data, and "system_om_per_kw", "system_capex_per_kw",
                "system_variable_om_per_kw", and "cap_cost_multiplier" columns.
        """
        capex = pd.DataFrame.from_dict(
            cost_inputs["system_capex_per_kw"]["wind"], orient="index"
        ).reset_index()
        opex = pd.DataFrame.from_dict(
            cost_inputs["system_om_per_kw"]["wind"], orient="index"
        ).reset_index()
        if tech == "solar":
            capex.columns = ["sector_abbr", "system_capex_per_kw"]
            opex.columns = ["sector_abbr", "system_om_per_kw"]
            costs = pd.merge(capex, opex, how="left", left_on="sector_abbr", right_on="sector_abbr")
        else:
            capex.columns = ["wind_turbine_kw_btm", "system_capex_per_kw"]
            opex.columns = ["wind_turbine_kw_btm", "system_om_per_kw"]
            costs = pd.merge(
                capex,
                opex,
                how="left",
                left_on="wind_turbine_kw_btm",
                right_on="wind_turbine_kw_btm",
            )
            costs.wind_turbine_kw_btm = costs.wind_turbine_kw_btm.astype(float)

        costs["cap_cost_multiplier"] = cost_inputs["cap_cost_multiplier"][tech]
        costs["system_variable_om_per_kw"] = cost_inputs["system_variable_om_per_kw"][tech]
        return costs

    def _preprocess_btm(self, df, tech="wind"):
        # sec = row['sector_abbr']
        # county = int(row['county_id'])
        df["county_id_int"] = df.county_id.astype(int)

        # Get the electricity rates
        df = pd.merge(
            df,
            self.retail_rate_inputs[["county_id", "sector_abbr", "elec_price_multiplier"]],
            how="left",
            left_on=["county_id_int", "sector_abbr"],
            right_on=["county_id", "sector_abbr"],
        )
        df = pd.merge(
            df,
            self.wholesale_rate_inputs[["county_id", "wholesale_elec_price_dollars_per_kwh"]],
            how="left",
            left_on="county_id_int",
            right_on="county_id",
        )
        df = df.drop(columns="county_id_int")

        # Technology-specific factors
        tech_join = "sector_abbr" if tech == "solar" else "wind_turbine_kw_btm"
        cost_df = self._process_btm_costs(self.COST_INPUTS["BTM"], tech)
        df = pd.merge(
            df,
            cost_df,
            how="left",
            left_on=tech_join,
            right_on=tech_join,
        )
        if tech == "solar":
            df = pd.merge(
                df,
                self.PERFORMANCE_INPUTS[tech],
                how="left",
                left_on=tech_join,
                right_on=tech_join,
            )
        else:
            df = pd.merge(
                df,
                self.wind_tech_inputs[["turbine_size_kw", "perf_improvement_factor"]],
                how="left",
                left_on="wind_turbine_kw_btm",
                right_on="turbine_size_kw",
            )
            df = pd.merge(
                df,
                self.wind_derate_inputs[["turbine_size_kw", "wind_derate_factor"]],
                how="left",
                left_on="wind_turbine_kw_btm",
                right_on="turbine_size_kw",
            )
            df["system_degradation"] = 0  # wind degradation already accounted for

        # Financial factors
        # For 2025, the incentives JSON data are located in the itc_fraction_of_capex
        # field, and need to be removed, then rejoined with the appropriate column names
        financial = self.FINANCIAL_INPUTS["BTM"].copy()
        if self.year == 2025:
            incentives = pd.DataFrame.from_dict(
                self.FINANCIAL_INPUTS["BTM"].pop("itc_fraction_of_capex")
            ).T
            incentives.index.name = "census_tract_id"

        deprec_sch = pd.DataFrame()
        deprec_sch["sector_abbr"] = financial["deprec_sch"].keys()
        deprec_sch["deprec_sch"] = financial["deprec_sch"].values()
        df = pd.merge(
            df,
            deprec_sch,
            how="left",
            left_on="sector_abbr",
            right_on="sector_abbr",
        )
        df = df.assign(
            economic_lifetime_yrs=financial["economic_lifetime_yrs"],
            loan_term_yrs=financial["loan_term_yrs"],
            loan_interest_rate=financial["loan_interest_rate"],
            down_payment_fraction=financial["down_payment_fraction"],
            real_discount_rate=financial["real_discount_rate"],
            tax_rate=financial["tax_rate"],
            inflation_rate=financial["inflation_rate"],
            elec_price_escalator=financial["elec_price_escalator"],
            itc_fraction_of_capex=0.3,
        )

        if self.year == 2025:
            df = df.set_index("census_tract_id", drop=False).join(incentives).reset_index(drop=True)
            df["itc_fraction_of_capex"] = df.applicable_credit.fillna(0.3)

        df = pd.merge(
            df,
            self.batt_price_inputs[
                [
                    "sector_abbr",
                    "batt_replace_frac_kw",
                    "batt_replace_frac_kwh",
                    "batt_capex_per_kwh",
                    "batt_capex_per_kw",
                    "linear_constant",
                    "batt_om_per_kw",
                    "batt_om_per_kwh",
                ]
            ],
            how="left",
            left_on="sector_abbr",
            right_on="sector_abbr",
        )
        df = pd.merge(
            df,
            self.batt_tech_inputs[["sector_abbr", "batt_eff", "batt_lifetime_yrs"]],
            how="left",
            left_on="sector_abbr",
            right_on="sector_abbr",
        )

        return df

    def _preprocess_fom(self, df, tech="wind"):
        columns = [
            "yr",
            "cambium_scenario",
            "analysis_period",
            "debt_option",
            "debt_percent",
            "inflation_rate",
            "dscr",
            "real_discount_rate",
            "term_int_rate",
            "term_tenor",
            f"ptc_fed_amt_{tech}",
            "itc_fed_pct",
            "deg",
            "system_capex_per_kw",
            "system_om_per_kw",
        ]

        itc_fraction_of_capex = self.FINANCIAL_INPUTS["FOM"]["itc_fraction_of_capex"]
        values = [
            self.year,
            self.CAMBIUM_SCENARIO,
            self.FINANCIAL_INPUTS["FOM"]["system_lifetime"],
            self.FINANCIAL_INPUTS["FOM"]["debt_option"],
            self.FINANCIAL_INPUTS["FOM"]["debt_percent"] * 100,
            self.FINANCIAL_INPUTS["FOM"]["inflation"] * 100,
            self.FINANCIAL_INPUTS["FOM"]["dscr"],
            self.FINANCIAL_INPUTS["FOM"]["discount_rate"] * 100,
            self.FINANCIAL_INPUTS["FOM"]["interest_rate"] * 100,
            self.FINANCIAL_INPUTS["FOM"]["system_lifetime"],
            self.FINANCIAL_INPUTS["FOM"]["ptc_fed_dlrs_per_kwh"][tech],
            itc_fraction_of_capex if self.year != 2025 else 0.3,
            self.FINANCIAL_INPUTS["FOM"]["degradation"],
            self.COST_INPUTS["FOM"]["system_capex_per_kw"][tech],
            self.COST_INPUTS["FOM"]["system_om_per_kw"][tech],
        ]
        df[columns] = values
        # 2025 uses census-tract based applicable credit for the itc_fed_pct, so update accordingly
        if self.year == 2025:
            incentives = pd.DataFrame.from_dict(
                self.FINANCIAL_INPUTS["FOM"]["itc_fraction_of_capex"]
            ).T
            incentives.index.name = "census_tract_id"
            df = df.set_index("census_tract_id", drop=False).join(incentives).reset_index(drop=True)
            df.itc_fed_pct = df.applicable_credit
            df.itc_fed_pct = df.itc_fed_pct.fillna(0.3)

        return df

    def run(self, agents: pd.DataFrame, sector: str):
        # self._connect_to_sql()

        max_w = self.config.project.settings.THREAD_WORKERS
        verb = self.config.project.settings.VERBOSITY

        if max_w > 1:
            results_list = []

            # btw, multithreading is NOT multiprocessing
            with cf.ThreadPoolExecutor(max_workers=max_w) as executor:
                # log.info(
                #     f'....beginning multiprocess execution of valuation with {max_w} threads')
                log.info(f"....beginning execution of valuation with {max_w} threads")

                start = time.time()
                checkpoint = max(1, int(len(agents) * verb))

                # submit to worker
                futures = [
                    executor.submit(self.worker, job, sector, self.config)
                    for _, job in agents.iterrows()
                ]

                # return results *as completed* - not in same order as input
                for f in cf.as_completed(futures):
                    results_list.append(f.result())
                    if len(results_list) % checkpoint == 0:
                        sec_per_agent = (time.time() - start) / len(results_list)
                        sec_per_agent = round(sec_per_agent, 3)

                        eta = (sec_per_agent * (len(agents) - len(results_list))) / 60 / 60
                        eta = round(eta, 2)

                        l_results = len(results_list)
                        l_agents = len(agents)

                        log.info(f"........finished job {l_results} / {l_agents}")
                        log.info(f"{sec_per_agent} seconds per agent")
                        log.info(f"ETA: {eta} hours")
        else:
            results_list = [self.worker(job, sector, self.config) for _, job in agents.iterrows()]

        # create results df from workers
        new_index = [r[0] for r in results_list]
        new_dicts = [r[1] for r in results_list]

        new_df = pd.DataFrame(new_dicts)
        new_df["gid"] = new_index
        new_df.set_index("gid", inplace=True)

        # merge valuation results to agents dataframe
        agents = agents.merge(new_df, on="gid", how="left")

        return agents

    def run_multiprocessing(self, agents, sector):
        # uses cf.ProcessPoolExecutor rather than cf.ThreadPoolExecutor in run.py
        if sector == "btm":
            agents = self._preprocess_btm(agents)
        else:
            agents = self._preprocess_fom(agents)

        max_w = self.config.project.settings.CORES
        verb = self.config.project.settings.VERBOSITY

        if max_w > 1:
            results_list = []

            with cf.ProcessPoolExecutor(max_workers=max_w) as executor:
                log.info(f"....beginning multiprocess execution of valuation with {max_w} cores")

                start = time.time()
                checkpoint = max(1, int(len(agents) * verb))

                # submit to worker
                futures = [
                    executor.submit(worker, job, sector, self.config)
                    for _, job in agents.iterrows()
                ]

                # return results *as completed* - not in same order as input
                for f in cf.as_completed(futures):
                    results_list.append(f.result())

                    if len(results_list) % checkpoint == 0:
                        sec_per_agent = (time.time() - start) / len(results_list)
                        sec_per_agent = round(sec_per_agent, 3)

                        eta = (sec_per_agent * (len(agents) - len(results_list))) / 60 / 60
                        eta = round(eta, 2)

                        l_results = len(results_list)
                        l_agents = len(agents)

                        log.info(f"........finished job {l_results} / {l_agents}")
                        log.info(f"{sec_per_agent} seconds per agent")
                        log.info(f"ETA: {eta} hours")
        else:
            results_list = [worker(job, sector, self.config) for _, job in agents.iterrows()]

        # create results df from workers
        new_index = [r[0] for r in results_list]
        new_dicts = [r[1] for r in results_list]

        new_df = pd.DataFrame(new_dicts)
        new_df["gid"] = new_index
        new_df.set_index("gid", inplace=True)

        # merge valuation results to agents dataframe
        agents = agents.merge(new_df, on="gid", how="left")

        return agents


def calc_financial_performance(capex_usd_p_kw, row, loan, batt_costs):
    system_costs = capex_usd_p_kw * row["system_size_kw"]

    # calculate system costs
    direct_costs = (system_costs + batt_costs) * row["cap_cost_multiplier"]
    sales_tax = 0.0
    loan.SystemCosts.total_installed_cost = direct_costs + sales_tax

    # execute financial module
    loan.execute()

    return loan.Outputs.npv


def calc_financial_performance_fom(capex_usd_p_kw, row, financial):
    system_costs = capex_usd_p_kw * row.loc["system_size_kw"]

    financial.SystemCosts.total_installed_cost = system_costs
    financial.FinancialParameters.construction_financing_cost = system_costs * 0.009

    financial.execute(1)

    return financial.Outputs.project_return_aftertax_npv


def find_cf_from_rev_wind(rev_dir, generation_scale_offset, tech_config, rev_index, year=2012):
    file_str = rev_dir / f"rev_{tech_config}_generation_{year}.h5"

    with h5.File(file_str, "r") as hf:
        cf_prof = np.array(hf["cf_profile"][:, int(rev_index)], dtype="float32")
        scale_factor = hf["cf_profile"].attrs.get("scale_factor")

    if len(cf_prof) == 17520:
        cf_prof = cf_prof[::2]

    if scale_factor is None:
        scale_factor = generation_scale_offset

    cf_prof /= scale_factor

    return cf_prof


def fetch_cambium_values(row, generation_hourly, cambium_dir, cambium_value, lower_thresh=0.01):
    # read processed cambium dataframe from pickle
    cambium_f = cambium_dir / f"{row['cambium_scenario']}_pca_{row['yr']}_processed.pqt"
    cambium_df = pd.read_parquet(cambium_f)

    cambium_df["year"] = cambium_df["year"].astype(str)
    cambium_df["pca"] = cambium_df["pca"].astype(str)
    cambium_df["variable"] = cambium_df["variable"].astype(str)

    # filter on pca, desired year, cambium variable
    mask = (
        (cambium_df["year"] == str(row["yr"]))
        & (cambium_df["pca"] == str(row["ba"]))
        & (cambium_df["variable"] == cambium_value)
    )

    cambium_output = cambium_df[mask]
    cambium_output = cambium_output.reset_index(drop=True)
    cambium_output = cambium_output["value"].values[0]

    # duplicate gen and cambium_output * analysis_period
    analysis_period = row["analysis_period"]
    generation_hourly = list(generation_hourly) * analysis_period
    cambium_output = list(cambium_output) * analysis_period

    rev = pd.DataFrame(columns=["gen", "value"])
    rev["value"] = np.array(cambium_output, dtype=np.float16)
    rev["gen"] = generation_hourly
    rev["cleared"] = rev["gen"] / 1000  # kW to MW

    # clip minimum output to 1% of maximum output so merchantplant works
    rev.loc[rev["cleared"] < rev["cleared"].max() * lower_thresh, "cleared"] = 0
    rev["cleared"] = rev["cleared"].apply(np.floor)

    rev = rev[["cleared", "value"]]
    tup = tuple(map(tuple, rev.values))

    return tup


def process_tariff(utilityrate, row, net_billing_sell_rate):
    """Instantiate the utilityrate5 PySAM model and process the agent's
    rate information to conform with PySAM input formatting.

    Parameters
    ----------
    agent : 'pd.Series'
        Individual agent object.

    Returns:
    -------
    utilityrate: 'PySAM.Utilityrate5'
    """
    # Monthly fixed charge [$]
    utilityrate.ElectricityRates.ur_monthly_fixed_charge = row["ur_monthly_fixed_charge"]

    # Annual minimum charge [$]
    # not currently tracked in URDB rate attribute downloads
    utilityrate.ElectricityRates.ur_annual_min_charge = 0.0

    # Monthly minimum charge [$]
    utilityrate.ElectricityRates.ur_monthly_min_charge = row["ur_monthly_min_charge"]

    # enable demand charge
    utilityrate.ElectricityRates.ur_dc_enable = row["ur_dc_enable"]

    # create matrix for demand charges
    if utilityrate.ElectricityRates.ur_dc_enable:
        if row["ur_dc_flat_mat"] is not None:
            flat_mat = [list(x) for x in row["ur_dc_flat_mat"]]
            utilityrate.ElectricityRates.ur_dc_flat_mat = flat_mat
        if row["ur_dc_tou_mat"] is not None:
            tou_mat = [list(x) for x in row["ur_dc_tou_mat"]]
            utilityrate.ElectricityRates.ur_dc_tou_mat = tou_mat

        d_wkdy_mat = [list(x) for x in row["ur_dc_sched_weekday"]]
        utilityrate.ElectricityRates.ur_dc_sched_weekday = d_wkdy_mat

        # energy charge weekend schedule
        d_wknd_mat = [list(x) for x in row["ur_dc_sched_weekday"]]
        utilityrate.ElectricityRates.ur_dc_sched_weekend = d_wknd_mat

    # create matrix for energy charges
    if row["en_electricity_rates"]:
        # energy rates table
        ec_mat = []
        for x in row["ur_ec_tou_mat"]:
            x = x.copy()
            x[-1] = net_billing_sell_rate
            ec_mat.append(list(x))
        utilityrate.ElectricityRates.ur_ec_tou_mat = ec_mat

        # energy charge weekday schedule
        wkdy_mat = [list(x) for x in row["ur_ec_sched_weekday"]]
        utilityrate.ElectricityRates.ur_ec_sched_weekday = wkdy_mat

        # energy charge weekend schedule
        wknd_mat = [list(x) for x in row["ur_ec_sched_weekday"]]
        utilityrate.ElectricityRates.ur_ec_sched_weekend = wknd_mat

    return utilityrate


def find_breakeven(
    row,
    loan,
    batt_costs,
    pysam_outputs,
    pre_calc_bounds_and_tolerances=True,
    **kwargs,
):
    # calculate theoretical min/max NPV values
    min_npv = calc_financial_performance(1e9, row, loan, batt_costs)

    if min_npv > 0.0:
        # if "infinite" cost system yields positive NPV
        # then system economical at any price - return 1e9 as flag
        out = 1e9
        return out, {"msg": "Inf cost w/ positive NPV"}

    max_npv = calc_financial_performance(0.0, row, loan, batt_costs)

    if max_npv < 0.0:
        # if zero-cost system yields negative NPV
        # then no breakeven cost exists - return -1 as flag
        out = -1.0
        return out, {"msg": "Zero cost w/ negative NPV"}

    # pre-calculate bounds for bisect and brentq methods
    if pre_calc_bounds_and_tolerances:
        pre_results = []
        pre_capex_array = np.logspace(9, 0, num=500)

        for capex_usd_p_kw in pre_capex_array:
            pre_npv = calc_financial_performance(capex_usd_p_kw, row, loan, batt_costs)
            pre_results.append(pre_npv)

        pre_results = np.asarray(pre_results)

        # find index where NPV flips from negative to positive
        ind = np.where(np.diff(np.sign(pre_results)))[0][0]

        # get lower (price with negative NPV) and upper (price with positive NPV) bounds
        a = pre_capex_array[ind]
        b = pre_capex_array[ind + 1]

        # depending on magnitude of capex values,
        # tolerance FOR NEWTON METHOD ONLY can be too small
        # anecdotally, ratio of tol:capex should be ~ 1e-6
        # pre-calculate tolerance as a proportion of capex value pre-calculated directly above
        tol = 1e-6 * ((a + b) / 2)

    if kwargs["method"] == "grid_search":
        if kwargs["capex_array"] is None:
            capex_array = np.arange(5000.0, -500.0, -500.0)
        else:
            capex_array = kwargs["capex_array"]

        try:
            results = []
            for capex_usd_p_kw in capex_array:
                npv = calc_financial_performance(capex_usd_p_kw, row, loan, batt_costs)
                results.append([capex_usd_p_kw, npv])

            out = pd.DataFrame(results, columns=["capex_usd_p_kw", "npv"])

            return out, {"msg": "Grid search cannot return additional PySAM outputs"}

        except Exception as e:
            raise ValueError("Grid search failed.") from e
    elif kwargs["method"] == "bisect":
        try:
            # required args for 'bisect'
            if not pre_calc_bounds_and_tolerances:
                a = kwargs["a"]
                b = kwargs["b"]

            # optional args for 'bisect'
            xtol = kwargs["xtol"] if "xtol" in kwargs.keys() else 2e-12
            rtol = kwargs["rtol"] if "rtol" in kwargs.keys() else 4 * np.finfo(float).eps
            maxiter = kwargs["maxiter"] if "maxiter" in kwargs.keys() else 100
            full_output = kwargs["full_output"] if "full_output" in kwargs.keys() else False
            disp = kwargs["disp"] if "disp" in kwargs.keys() else True
        except Exception as e:
            raise ValueError("Make sure method-specific inputs are specified correctly.") from e

        try:
            breakeven_cost_usd_p_kw, r = optimize.bisect(
                calc_financial_performance,
                args=(row, loan, batt_costs),
                a=a,
                b=b,
                xtol=xtol,
                rtol=rtol,
                maxiter=maxiter,
                full_output=full_output,
                disp=disp,
            )

            return breakeven_cost_usd_p_kw, {
                k: loan.Outputs.export().get(k, None) for k in pysam_outputs
            }

        except Exception as e:
            raise ValueError("Root finding failed.") from e
    elif kwargs["method"] == "brentq":
        try:
            # required args for 'brentq'
            if not pre_calc_bounds_and_tolerances:
                a = kwargs["a"]
                b = kwargs["b"]

            # optional args for 'brentq'
            xtol = kwargs["xtol"] if "xtol" in kwargs.keys() else 2e-12
            rtol = kwargs["rtol"] if "rtol" in kwargs.keys() else 4 * np.finfo(float).eps
            maxiter = kwargs["maxiter"] if "maxiter" in kwargs.keys() else 100
            full_output = kwargs["full_output"] if "full_output" in kwargs.keys() else False
            disp = kwargs["disp"] if "disp" in kwargs.keys() else True
        except Exception as e:
            raise ValueError("Make sure method-specific inputs are specified correctly.") from e

        try:
            breakeven_cost_usd_p_kw, r = optimize.brentq(
                calc_financial_performance,
                args=(row, loan, batt_costs),
                a=a,
                b=b,
                xtol=xtol,
                rtol=rtol,
                maxiter=maxiter,
                full_output=full_output,
                disp=disp,
            )

            return breakeven_cost_usd_p_kw, {
                k: loan.Outputs.export().get(k, None) for k in pysam_outputs
            }

        except Exception as e:
            raise ValueError("Root finding failed.") from e

    elif kwargs["method"] == "newton":
        try:
            # required args for 'newton'
            x0 = kwargs["x0"]

            if not pre_calc_bounds_and_tolerances:
                tol = kwargs["tol"] if "tol" in kwargs.keys() else 1.48e-08

            # --- optional args for 'newton' ---
            fprime = kwargs["fprime"] if "fprime" in kwargs.keys() else None
            maxiter = kwargs["maxiter"] if "maxiter" in kwargs.keys() else 50
            fprime2 = kwargs["fprime2"] if "fprime2" in kwargs.keys() else None
            x1 = kwargs["x1"] if "x1" in kwargs.keys() else None
            rtol = kwargs["rtol"] if "rtol" in kwargs.keys() else 0.0
            full_output = kwargs["full_output"] if "full_output" in kwargs.keys() else False
            disp = kwargs["disp"] if "disp" in kwargs.keys() else True
        except Exception as e:
            raise ValueError("Make sure method-specific inputs are specified correctly.") from e

        try:
            breakeven_cost_usd_p_kw, r = optimize.newton(
                calc_financial_performance,
                args=(row, loan, batt_costs),
                x0=x0,
                fprime=fprime,
                tol=tol,
                maxiter=maxiter,
                fprime2=fprime2,
                x1=x1,
                rtol=rtol,
                full_output=full_output,
                disp=disp,
            )

            return breakeven_cost_usd_p_kw, {
                k: loan.Outputs.export().get(k, None) for k in pysam_outputs
            }

        except Exception as e:
            raise ValueError("Root finding failed.") from e

    else:
        raise ValueError("Invalid method passed to find_breakeven function")


def find_breakeven_fom(
    row,
    financial,
    pysam_outputs,
    pre_calc_bounds_and_tolerances=True,
    **kwargs,
):
    # calculate theoretical min/max NPV values
    min_npv = calc_financial_performance_fom(1e9, row, financial)

    if min_npv > 0.0:
        # if "infinite" cost system yields positive NPV,
        # then system economical at any price - return 1e9 as flag
        out = 1e9
        return out, {"msg": "Inf cost w/ positive NPV"}

    max_npv = calc_financial_performance_fom(0.0, row, financial)

    if max_npv < 0.0:
        # if zero-cost system yields negative NPV,
        # then no breakeven cost exists - return -1 as flag
        out = -1.0
        return out, {"msg": "Zero cost w/ negative NPV"}

    # pre-calculate bounds for bisect and brentq methods
    if pre_calc_bounds_and_tolerances:
        pre_results = []
        pre_capex_array = np.logspace(9, 0, num=500)
        for capex_usd_p_kw in pre_capex_array:
            pre_npv = calc_financial_performance_fom(capex_usd_p_kw, row, financial)
            pre_results.append(pre_npv)

        pre_results = np.asarray(pre_results)

        # find index where NPV flips from negative to positive
        ind = np.where(np.diff(np.sign(pre_results)))[0][0]

        # get lower (price with negative NPV) and upper (price with positive NPV) bounds
        a = pre_capex_array[ind]
        b = pre_capex_array[ind + 1]

        # depending on magnitude of capex values,
        # tolerance FOR NEWTON METHOD ONLY can be too small
        # anecdotally, ratio of tol:capex should be ~ 1e-6
        # pre-calculate tolerance as a proportion of
        # capex value pre-calculated directly above
        tol = 1e-6 * ((a + b) / 2)

    if kwargs["method"] == "grid_search":
        if kwargs["capex_array"] is None:
            capex_array = np.arange(5000.0, -500.0, -500.0)
        else:
            capex_array = kwargs["capex_array"]

        try:
            results = []
            for capex_usd_p_kw in capex_array:
                npv = calc_financial_performance_fom(capex_usd_p_kw, row, financial)
                results.append([capex_usd_p_kw, npv])

            out = pd.DataFrame(results, columns=["capex_usd_p_kw", "npv"])

            return out, {"msg": "Grid search cannot return additional PySAM outputs"}

        except Exception as e:
            raise ValueError("Grid search failed.") from e
    elif kwargs["method"] == "bisect":
        try:
            # required args for 'bisect
            if not pre_calc_bounds_and_tolerances:
                a = kwargs["a"]
                b = kwargs["b"]

            # optional args for 'bisect'
            xtol = kwargs["xtol"] if "xtol" in kwargs.keys() else 2e-12
            rtol = kwargs["rtol"] if "rtol" in kwargs.keys() else 4 * np.finfo(float).eps
            maxiter = kwargs["maxiter"] if "maxiter" in kwargs.keys() else 100
            full_output = kwargs["full_output"] if "full_output" in kwargs.keys() else False
            disp = kwargs["disp"] if "disp" in kwargs.keys() else True
        except Exception as e:
            raise ValueError("Make sure method-specific inputs are specified correctly.") from e

        try:
            breakeven_cost_usd_p_kw, r = optimize.bisect(
                calc_financial_performance_fom,
                args=(row, financial),
                a=a,
                b=b,
                xtol=xtol,
                rtol=rtol,
                maxiter=maxiter,
                full_output=full_output,
                disp=disp,
            )

            return breakeven_cost_usd_p_kw, {
                k: financial.Outputs.export().get(k, None) for k in pysam_outputs
            }

        except Exception as e:
            raise ValueError("Root finding failed.") from e
    elif kwargs["method"] == "brentq":
        try:
            # required args for 'brentq'
            if not pre_calc_bounds_and_tolerances:
                a = kwargs["a"]
                b = kwargs["b"]

            # optional args for 'brentq'
            xtol = kwargs["xtol"] if "xtol" in kwargs.keys() else 2e-12
            rtol = kwargs["rtol"] if "rtol" in kwargs.keys() else 4 * np.finfo(float).eps
            maxiter = kwargs["maxiter"] if "maxiter" in kwargs.keys() else 100
            full_output = kwargs["full_output"] if "full_output" in kwargs.keys() else False
            disp = kwargs["disp"] if "disp" in kwargs.keys() else True
        except Exception as e:
            raise ValueError("Make sure method-specific inputs are specified correctly.") from e

        try:
            breakeven_cost_usd_p_kw, r = optimize.brentq(
                calc_financial_performance_fom,
                args=(row, financial),
                a=a,
                b=b,
                xtol=xtol,
                rtol=rtol,
                maxiter=maxiter,
                full_output=full_output,
                disp=disp,
            )

            return breakeven_cost_usd_p_kw, {
                k: financial.Outputs.export().get(k, None) for k in pysam_outputs
            }

        except Exception as e:
            raise ValueError("Root finding failed.") from e
    elif kwargs["method"] == "newton":
        try:
            # required args for 'newton'
            x0 = kwargs["x0"]

            if not pre_calc_bounds_and_tolerances:
                tol = kwargs["tol"] if "tol" in kwargs.keys() else 1.48e-08

            # optional args for 'newton'
            fprime = kwargs["fprime"] if "fprime" in kwargs.keys() else None
            maxiter = kwargs["maxiter"] if "maxiter" in kwargs.keys() else 50
            fprime2 = kwargs["fprime2"] if "fprime2" in kwargs.keys() else None
            x1 = kwargs["x1"] if "x1" in kwargs.keys() else None
            rtol = kwargs["rtol"] if "rtol" in kwargs.keys() else 0.0
            full_output = kwargs["full_output"] if "full_output" in kwargs.keys() else False
            disp = kwargs["disp"] if "disp" in kwargs.keys() else True
        except Exception as e:
            raise ValueError("Make sure method-specific inputs are specified correctly") from e

        try:
            breakeven_cost_usd_p_kw, r = optimize.newton(
                calc_financial_performance_fom,
                args=(row, financial),
                x0=x0,
                fprime=fprime,
                tol=tol,
                maxiter=maxiter,
                fprime2=fprime2,
                x1=x1,
                rtol=rtol,
                full_output=full_output,
                disp=disp,
            )

            return breakeven_cost_usd_p_kw, {
                k: financial.Outputs.export().get(k, None) for k in pysam_outputs
            }

        except Exception as e:
            raise ValueError("Root finding failed") from e
    else:
        raise ValueError("Invalid method passed to find_breakeven function")


def process_btm(
    row,
    tech,
    generation_hourly,
    consumption_hourly,
    pysam_outputs,
    en_batt=False,
    batt_dispatch=None,
):
    """Behind-the-meter ...

    This function processes a BTM agent by:
        1)

    Parameters
    ----------
    **row** : 'DataFrame row'
        The row of the dataframe on which the function is performed
    **exported_hourly** : ''
        8760 of generation
    **consumption_hourly** : ''
        8760 of consumption
    **tariff_dict** : 'dict'
        Dictionary containing tariff parameters
    **btm_nem** : 'bool'
        Enable NEM for BTM parcels
    **en_batt** : 'bool'
        Enable battery modeling
    **batt_dispatch** : 'string'
        Specify battery dispatch strategy type

    Returns:
    -------

    """
    # extract agent load and generation profiles
    generation_hourly = np.array(generation_hourly)
    consumption_hourly = np.array(consumption_hourly, dtype=np.float32)

    # specify tech-agnostic system size column
    row["system_size_kw"] = row[f"{tech}_size_kw_btm"]

    # instantiate PySAM battery model based on agent sector
    if row.loc["sector_abbr"] == "res":
        batt = battery.default("GenericBatteryResidential")
    else:
        batt = battery.default("GenericBatteryCommercial")

    # instantiate PySAM utilityrate5 model based on agent sector
    if row.loc["sector_abbr"] == "res":
        utilityrate = ur5.default("GenericBatteryResidential")
    else:
        utilityrate = ur5.default("GenericBatteryCommercial")

    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###--- SYSTEM LIFETIME SETTINGS ---###
    ######################################

    # Inflation rate [%]
    utilityrate.Lifetime.inflation_rate = row.loc["inflation_rate"] * 100

    # Number of years in analysis [years]
    utilityrate.Lifetime.analysis_period = row.loc["economic_lifetime_yrs"]

    # Lifetime hourly system outputs [0/1];
    # Options: 0=hourly first year,1=hourly lifetime
    utilityrate.Lifetime.system_use_lifetime_output = 0

    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###---- DEGRADATION/ESCALATION ----###
    ######################################

    # Annual energy degradation [%]
    utilityrate.SystemOutput.degradation = [row.loc["system_degradation"] * 100]
    # Annual electricity rate escalation [%/year]
    utilityrate.ElectricityRates.rate_escalation = [row.loc["elec_price_escalator"] * 100]

    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###---- NET METERING SETTINGS -----###
    ######################################

    # dictionary to map dGen compensation styles to PySAM options
    nem_options = {"net metering": 0, "net billing": 2, "buy all sell all": 4, "none": 2}

    # metering options
    # 0 = net energy metering
    # 1 = net energy metering with $ credits
    # 2 = net billing
    # 3 = net billing with carryover to next month
    # 4 = buy all - sell all
    c_style = row.loc["compensation_style"]
    utilityrate.ElectricityRates.ur_metering_option = nem_options[c_style]

    # year end sell rate [$/kWh]
    ws_price = row.loc["wholesale_elec_price_dollars_per_kwh"]
    mult = row.loc["elec_price_multiplier"]
    utilityrate.ElectricityRates.ur_nm_yearend_sell_rate = ws_price * mult

    if c_style == "none":
        net_billing_sell_rate = 0.0
    else:
        net_billing_sell_rate = ws_price * mult

    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###-------- BUY/SELL RATES --------###
    ######################################

    # Enable time step sell rates [0/1]
    utilityrate.ElectricityRates.ur_en_ts_sell_rate = 0

    # Time step sell rates [0/1]
    utilityrate.ElectricityRates.ur_ts_sell_rate = [0.0]

    # Set sell rate equal to buy rate [0/1]
    utilityrate.ElectricityRates.ur_sell_eq_buy = 0

    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###-------- MISC. SETTINGS --------###
    ######################################

    # Use single monthly peak for TOU demand charge;
    # options:
    # 0 = use TOU peak
    # 1 = use flat peak
    utilityrate.ElectricityRates.TOU_demand_single_peak = 0  # ?

    # Optionally enable/disable electricity_rate [years]
    utilityrate.ElectricityRates.en_electricity_rates = 1

    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###----- TARIFF RESTRUCTURING -----###
    ######################################
    utilityrate = process_tariff(utilityrate, row, net_billing_sell_rate)

    ######################################
    ###----------- CASHLOAN -----------###
    ###----- FINANCIAL PARAMETERS -----###
    ######################################

    # Initiate cashloan model and set market-specific variables
    # Assume res agents do not evaluate depreciation at all
    # Assume non-res agents only evaluate federal depreciation (not state)
    if row.loc["sector_abbr"] == "res":
        loan = cashloan.default("GenericBatteryResidential")
        loan.FinancialParameters.market = 0
    else:
        loan = cashloan.default("GenericBatteryCommercial")
        loan.FinancialParameters.market = 1

    loan.FinancialParameters.analysis_period = row.loc["economic_lifetime_yrs"]
    loan.FinancialParameters.debt_fraction = 100 - (row.loc["down_payment_fraction"] * 100)
    loan.FinancialParameters.federal_tax_rate = [(row.loc["tax_rate"] * 100) * 0.7]  # SAM default
    loan.FinancialParameters.inflation_rate = row.loc["inflation_rate"] * 100
    loan.FinancialParameters.insurance_rate = 0
    loan.FinancialParameters.loan_rate = row.loc["loan_interest_rate"] * 100
    loan.FinancialParameters.loan_term = row.loc["loan_term_yrs"]
    loan.FinancialParameters.mortgage = 0  # default value - standard loan (no mortgage)
    loan.FinancialParameters.prop_tax_assessed_decline = 5  # PySAM default
    loan.FinancialParameters.prop_tax_cost_assessed_percent = 95  # PySAM default
    loan.FinancialParameters.property_tax_rate = 0  # PySAM default
    loan.FinancialParameters.real_discount_rate = row.loc["real_discount_rate"] * 100
    loan.FinancialParameters.salvage_percentage = 0
    loan.FinancialParameters.state_tax_rate = [(row.loc["tax_rate"] * 100) * 0.3]  # SAM default
    loan.FinancialParameters.system_heat_rate = 0

    ######################################
    ###----------- CASHLOAN -----------###
    ###--------- SYSTEM COSTS ---------###
    ######################################

    # System costs that are input to loan.SystemCosts will depend on system configuration
    # (PV, batt, PV+batt) and are therefore specified in calc_system_performance()

    system_costs = {}
    system_costs["system_om_per_kw"] = row["system_om_per_kw"]
    system_costs["system_variable_om_per_kw"] = row["system_variable_om_per_kw"]
    system_costs["cap_cost_multiplier"] = row["cap_cost_multiplier"]
    system_costs["batt_capex_per_kw"] = row["batt_capex_per_kw"]
    system_costs["batt_capex_per_kwh"] = row["batt_capex_per_kwh"]
    system_costs["batt_om_per_kw"] = row["batt_om_per_kw"]
    system_costs["batt_om_per_kwh"] = row["batt_om_per_kwh"]
    system_costs["linear_constant"] = row["linear_constant"]

    # costs for PV+batt configuration are distinct from standalone techs
    # TODO: _combined costs are only valid for PV+batt -- process these differently for wind?

    ######################################
    ###----------- CASHLOAN -----------###
    ###---- DEPRECIATION PARAMETERS ---###
    ######################################

    if row.loc["sector_abbr"] == "res":
        loan.Depreciation.depr_fed_type = 0
        loan.Depreciation.depr_sta_type = 0
    else:
        loan.Depreciation.depr_fed_type = 1
        loan.Depreciation.depr_sta_type = 0

    ######################################
    ###----------- CASHLOAN -----------###
    ###----- TAX CREDIT INCENTIVES ----###
    ######################################

    itc_fed_pct = row.loc["itc_fraction_of_capex"]
    itc_fed_pct = itc_fed_pct * 100
    if itc_fed_pct != 0:
        loan.TaxCreditIncentives.itc_fed_percent = itc_fed_pct  # [itc_fed_pct]

    ######################################
    ###----------- CASHLOAN -----------###
    ###-------- BATTERY SYSTEM --------###
    ######################################

    loan.BatterySystem.batt_replacement_option = 2  # user schedule

    batt_replacement_schedule = [0 for i in range(0, row.loc["batt_lifetime_yrs"] - 1)] + [100]
    loan.BatterySystem.batt_replacement_schedule_percent = batt_replacement_schedule

    ######################################
    ###----------- CASHLOAN -----------###
    ###-------- SYSTEM OUTPUT ---------###
    ######################################

    loan.SystemOutput.degradation = [row.loc["system_degradation"] * 100]

    ######################################
    ###----------- CASHLOAN -----------###
    ###----------- LIFETIME -----------###
    ######################################

    loan.Lifetime.system_use_lifetime_output = 0

    ######################################
    ###--- INITIALIZE PYSAM MODULES ---###
    ######################################

    inv_eff = 0.96  # default SAM inverter efficiency for PV
    gen = [i * inv_eff for i in generation_hourly]

    # set up battery, with system generation conditional
    # on the battery generation being included
    if en_batt:
        batt.BatterySystem.en_batt = 1
        batt.BatterySystem.batt_ac_or_dc = 1  # default value
        batt.BatteryCell.batt_chem = 1  # default value is 1: li ion for residential
        batt.BatterySystem.batt_meter_position = 0

        # need to consider lifetime since pysam needs
        # profiles for all years if considering replacement
        batt.Lifetime.system_use_lifetime_output = 0
        batt.BatterySystem.batt_replacement_option = 0

        batt.Inverter.inverter_model = 4  # default value
        batt.Load.load = consumption_hourly

        # set different ratios for residential and comm/industrial systems
        sec = row["sector_abbr"]
        if sec == "res":
            # pv to Battery ratio (kW) - From Ashreeta, 02/08/2020 (1.31372) updated 9/24
            pv_to_batt_ratio = 1.21
        else:
            # updated 9/24 to reflect values from cost report, Denholm et.al.
            pv_to_batt_ratio = 1.67

        batt_capacity_to_power_ratio = 2  # hours of operation

        # default SAM value for residential systems is 10
        desired_size = row["system_size_kw"] / pv_to_batt_ratio
        desired_power = desired_size / batt_capacity_to_power_ratio

        battery_tools.battery_model_sizing(batt, desired_power, desired_size, 500)

        # copy over gen and load
        batt.Load.load = consumption_hourly  # kw
        batt.SystemOutput.gen = gen

        # dispatch options in detailed battery:
        if batt_dispatch == "peak_shaving":
            batt.BatteryDispatch.batt_dispatch_choice = 0
        else:
            batt.BatteryDispatch.batt_dispatch_choice = 5

        batt.BatteryDispatch.batt_dispatch_auto_can_charge = 1
        batt.BatteryDispatch.batt_dispatch_auto_can_clipcharge = 1
        batt.BatteryDispatch.batt_dispatch_auto_can_gridcharge = 1
        cycle_cost_list = [0.1]
        batt.BatterySystem.batt_cycle_cost = cycle_cost_list[0]
        batt.BatterySystem.batt_cycle_cost_choice = 0
        # batt.BatteryDispatch.batt_pv_ac_forecast = batt_model.Battery.ac
        batt.execute(0)

        loan.BatterySystem.en_batt = 1
        loan.BatterySystem.batt_computed_bank_capacity = batt.Outputs.batt_bank_installed_capacity
        loan.BatterySystem.batt_bank_replacement = batt.Outputs.batt_bank_replacement

        # specify number of O&M types (1 = PV+batt)
        loan.SystemCosts.add_om_num_types = 1

        loan.BatterySystem.battery_per_kWh = system_costs["batt_capex_per_kwh"]

        loan.SystemCosts.om_capacity = [
            system_costs["system_om_per_kw"] + system_costs["system_variable_om_per_kw"]
        ]
        loan.SystemCosts.om_capacity1 = [system_costs["batt_om_per_kw"]]
        loan.SystemCosts.om_production1 = [system_costs["batt_om_per_kwh"] * 1000.0]
        loan.SystemCosts.om_batt_replacement_cost = [0.0]

        # specify linear constant adder for standalone battery system
        row["linear_constant"]

        # Battery capacity for System Costs values [kW]
        loan.SystemCosts.om_capacity1_nameplate = batt.BatterySystem.batt_power_charge_max_kwdc
        # Battery production for System Costs values [kWh]
        loan.SystemCosts.om_production1_values = [batt.Outputs.batt_bank_installed_capacity]

        batt_costs = (
            system_costs["batt_capex_per_kw"] * batt.BatterySystem.batt_power_charge_max_kwdc
        ) + (system_costs["batt_capex_per_kwh"] * batt.Outputs.batt_bank_installed_capacity)

        value_of_resiliency = 0.0  # TODO: add value of resiliency?
        utilityrate.SystemOutput.gen = batt.SystemOutput.gen

    else:
        batt.BatterySystem.en_batt = 0
        loan.BatterySystem.en_batt = 0

        loan.Battery.batt_annual_charge_from_system = [0.0]
        loan.Battery.batt_annual_discharge_energy = [0.0]
        loan.Battery.batt_capacity_percent = [0.0]
        loan.Battery.battery_total_cost_lcos = 0.0
        loan.Battery.grid_to_batt = [0.0]
        loan.Battery.monthly_batt_to_grid = [0.0]
        loan.Battery.monthly_grid_to_batt = [0.0]
        loan.Battery.monthly_grid_to_load = [0.0]
        loan.Battery.monthly_system_to_grid = [0.0]

        # for PySAM 4.0
        # loan.LCOS.batt_annual_charge_energy = [0.]
        # loan.LCOS.batt_annual_charge_from_system = [0.]
        # loan.LCOS.batt_annual_discharge_energy = [0.]
        # loan.LCOS.batt_capacity_percent = [0.]
        # loan.LCOS.battery_total_cost_lcos = 0.
        # loan.LCOS.charge_w_sys_ec_ym = [[0.]]
        # loan.LCOS.grid_to_batt = [0.]
        # loan.LCOS.monthly_batt_to_grid = [0.]
        # loan.LCOS.monthly_grid_to_batt = [0.]
        # loan.LCOS.monthly_grid_to_load = [0.]
        # loan.LCOS.monthly_system_to_grid = [0.]
        # loan.LCOS.true_up_credits_ym = [[0.]]
        # loan.LCOS.year1_monthly_ec_charge_gross_with_system = [0.]
        # loan.LCOS.year1_monthly_ec_charge_with_system = [0.]
        # loan.LCOS.year1_monthly_electricity_to_grid = [0.]

        # specify number of O&M types (0 = PV only)
        loan.SystemCosts.add_om_num_types = 0
        # since battery system size is zero, specify standalone PV O&M costs
        loan.SystemCosts.om_capacity = [
            system_costs["system_om_per_kw"] + system_costs["system_variable_om_per_kw"]
        ]
        loan.SystemCosts.om_batt_replacement_cost = [0.0]

        batt_costs = 0.0

        # linear constant for standalone PV system is 0.

        value_of_resiliency = 0.0
        utilityrate.SystemOutput.gen = gen

    # Execute utility rate module
    utilityrate.Load.load = consumption_hourly
    utilityrate.execute(1)

    # Process payment incentives
    # TODO: apply incentives?
    # loan = process_incentives(
    #     loan, row['system_size_kw'],
    #     batt.BatterySystem.batt_power_discharge_max_kwdc,
    #     batt.Outputs.batt_bank_installed_capacity,
    #     generation_hourly,
    #     row
    # )

    # specify final Cashloan parameters
    loan.FinancialParameters.system_capacity = row["system_size_kw"]

    # add value_of_resiliency -- should only apply from year 1 onwards, not to year 0
    aev = utilityrate.Outputs.annual_energy_value
    annual_energy_value = [aev[0]] + [x + value_of_resiliency for i, x in enumerate(aev) if i != 0]

    loan.SystemOutput.annual_energy_value = annual_energy_value
    loan.SystemOutput.gen = utilityrate.SystemOutput.gen
    loan.ThirdPartyOwnership.elec_cost_with_system = utilityrate.Outputs.elec_cost_with_system
    loan.ThirdPartyOwnership.elec_cost_without_system = utilityrate.Outputs.elec_cost_without_system

    _ = calc_financial_performance(row["system_capex_per_kw"], row, loan, batt_costs)

    row["additional_pysam_outputs"] = {k: loan.Outputs.export().get(k, None) for k in pysam_outputs}

    # run root finding algorithm to find breakeven cost based on calculated NPV
    out, _ = find_breakeven(
        row=row,
        loan=loan,
        pysam_outputs=pysam_outputs,
        batt_costs=batt_costs,
        pre_calc_bounds_and_tolerances=False,
        **{"method": "newton", "x0": 10000.0, "full_output": True},
    )

    row["breakeven_cost_usd_p_kw"] = out

    return row


def process_fom(
    row,
    tech,
    generation_hourly,
    market_profile,
    pysam_outputs,
    en_batt=False,
    batt_dispatch=None,
):
    """Front-of-the-meter ...

    This function processes a FOM agent by:
        1)

    Parameters
    ----------
    **row** : 'DataFrame row'
        The row of the dataframe on which the function is performed
    **exported_hourly** : ''
        8760 of generation
    **cambium_grid_value** : ''
        8760 of Cambium values

    Returns:
    -------

    """
    # extract generation profile
    generation_hourly = np.array(generation_hourly)

    # specify tech-agnostic system size column
    row["system_size_kw"] = row[f"{tech}_size_kw_fom"]

    inv_eff = 1.0  # required inverter efficiency for FOM systems
    gen = [i * inv_eff for i in generation_hourly]

    # set up battery, with system generation conditional
    # on the battery generation being included
    if en_batt:
        # TODO: implement FOM battery
        pass
    else:
        # initialize PySAM model
        if tech == "solar":
            financial = mp.default("PVWattsMerchantPlant")
        elif tech == "wind":
            financial = mp.default("WindPowerMerchantPlant")
        else:
            msg = "Please write a wrapper to account for the new technology type"
            raise NotImplementedError(f"{msg} {tech}")

        ptc_fed_amt = row[f"ptc_fed_amt_{tech}"]
        itc_fed_pct = row["itc_fed_pct"]
        deg = row["deg"]
        system_capex_per_kw = row["system_capex_per_kw"]
        system_om_per_kw = row["system_om_per_kw"]

        financial.Lifetime.system_use_lifetime_output = 0
        financial.FinancialParameters.analysis_period = row["analysis_period"]
        financial.FinancialParameters.debt_option = row["debt_option"]
        financial.FinancialParameters.debt_percent = row["debt_percent"]
        financial.FinancialParameters.inflation_rate = row["inflation_rate"]
        financial.FinancialParameters.dscr = row["dscr"]
        financial.FinancialParameters.real_discount_rate = row["real_discount_rate"]
        financial.FinancialParameters.term_int_rate = row["term_int_rate"]
        financial.FinancialParameters.term_tenor = row["term_tenor"]
        financial.FinancialParameters.insurance_rate = 0
        financial.FinancialParameters.federal_tax_rate = [21]
        financial.FinancialParameters.state_tax_rate = [7]
        financial.FinancialParameters.property_tax_rate = 0
        financial.FinancialParameters.prop_tax_cost_assessed_percent = 100
        financial.FinancialParameters.prop_tax_assessed_decline = 0

        financial.FinancialParameters.cost_debt_closing = 0
        financial.FinancialParameters.cost_debt_fee = 1.5
        financial.FinancialParameters.cost_other_financing = 0
        financial.FinancialParameters.dscr_reserve_months = 0
        financial.FinancialParameters.equip1_reserve_cost = 0
        financial.FinancialParameters.equip1_reserve_freq = 0
        financial.FinancialParameters.equip2_reserve_cost = 0
        financial.FinancialParameters.equip2_reserve_freq = 0
        financial.FinancialParameters.equip3_reserve_cost = 0
        financial.FinancialParameters.equip3_reserve_freq = 0
        financial.FinancialParameters.months_receivables_reserve = 0
        financial.FinancialParameters.months_working_reserve = 0
        financial.FinancialParameters.reserves_interest = 0
        financial.FinancialParameters.salvage_percentage = 0

        financial.TaxCreditIncentives.ptc_fed_amount = [ptc_fed_amt]
        financial.TaxCreditIncentives.ptc_fed_escal = 2.5
        financial.TaxCreditIncentives.ptc_fed_term = 10

        itc_fed_pct = itc_fed_pct * 100
        if itc_fed_pct != 0:
            financial.TaxCreditIncentives.itc_fed_percent = itc_fed_pct  # [itc_fed_pct]

        financial.Depreciation.depr_custom_schedule = [0]

        financial.SystemCosts.om_fixed = [0]
        financial.SystemCosts.om_fixed_escal = 0
        financial.SystemCosts.om_production = [0]
        financial.SystemCosts.om_production_escal = 0
        financial.SystemCosts.om_capacity_escal = 0
        financial.SystemCosts.om_fuel_cost = [0]
        financial.SystemCosts.om_fuel_cost_escal = 0
        financial.SystemCosts.om_replacement_cost_escal = 0

        financial.SystemOutput.degradation = [deg * 100]
        financial.SystemOutput.system_capacity = row.loc["system_size_kw"]
        financial.SystemOutput.gen = gen
        financial.SystemOutput.system_pre_curtailment_kwac = gen
        financial.SystemOutput.annual_energy_pre_curtailment_ac = np.sum(gen[:8760])

        financial.Revenue.mp_enable_energy_market_revenue = 1
        financial.Revenue.mp_energy_market_revenue = market_profile
        financial.Revenue.mp_enable_ancserv1 = 0
        financial.Revenue.mp_enable_ancserv2 = 0
        financial.Revenue.mp_enable_ancserv3 = 0
        financial.Revenue.mp_enable_ancserv4 = 0
        financial.Revenue.mp_ancserv1_revenue = [(0, 0) for i in range(len(market_profile))]
        financial.Revenue.mp_ancserv2_revenue = [(0, 0) for i in range(len(market_profile))]
        financial.Revenue.mp_ancserv3_revenue = [(0, 0) for i in range(len(market_profile))]
        financial.Revenue.mp_ancserv4_revenue = [(0, 0) for i in range(len(market_profile))]

        financial.CapacityPayments.cp_capacity_payment_type = 0
        financial.CapacityPayments.cp_capacity_payment_amount = [0]
        financial.CapacityPayments.cp_capacity_credit_percent = [0]
        financial.CapacityPayments.cp_capacity_payment_esc = 0
        financial.CapacityPayments.cp_system_nameplate = row.loc["system_size_kw"]

        if en_batt:
            # TODO: add battery
            financial.CapacityPayments.cp_battery_nameplate = 0
        else:
            financial.CapacityPayments.cp_battery_nameplate = 0

        financial.SystemCosts.om_capacity = [system_om_per_kw]

        log.info(f"row {row.loc['gid']} calculating financial performance")
        _ = calc_financial_performance_fom(system_capex_per_kw, row, financial)
        row["additional_pysam_outputs"] = {
            k: financial.Outputs.export().get(k, None) for k in pysam_outputs
        }

        # run root finding algorithm to find breakeven cost based on calculated NPV
        log.info(f"row {row.loc['gid']} breakeven")
        out, _ = find_breakeven_fom(
            row=row,
            financial=financial,
            pysam_outputs=pysam_outputs,
            pre_calc_bounds_and_tolerances=False,
            **{"method": "newton", "x0": 10000.0, "full_output": True},
        )

        row["breakeven_cost_usd_p_kw"] = out

    return row


def worker(row: pd.Series, sector: str, config: Configuration):
    try:
        results = {}
        for tech in config.project.settings.TECHS:
            if tech == "wind":
                tech_config = row["turbine_class"]
            elif tech == "solar":
                # TODO: Not updated for the actual configuration, so solar is likely out of date
                azimuth = config.rev.settings.azimuth_direction_to_degree[row[f"azimuth_{sector}"]]
                tilt = row[f"tilt_{sector}"]
                tech_config = str(azimuth) + "_" + str(tilt)

            if row[f"{tech}_size_kw_{sector}"] > 0:
                tech_config = row["turbine_class"]
                cf_hourly = find_cf_from_rev_wind(
                    config.rev.DIR,
                    config.project.settings.GENERATION_SCALE_OFFSET["wind"],
                    tech_config,
                    row[f"rev_index_{tech}"],
                )
                generation_hourly = cf_hourly * row[f"{tech}_size_kw_{sector}"]
                generation_hourly = generation_hourly.round(2)
                generation_hourly = np.array(generation_hourly).astype(np.float32).round(2)
            else:
                # if system size is zero, return dummy values
                results[f"{tech}_breakeven_cost_{sector}"] = -1
                results[f"{tech}_pysam_outputs_{sector}"] = {"msg": "System size is zero"}
                continue

            if sector == "btm":
                row = process_btm(
                    row=row,
                    tech=tech,
                    generation_hourly=generation_hourly,
                    consumption_hourly=row["consumption_hourly"],
                    pysam_outputs=config.pysam.outputs.btm,
                    en_batt=False,
                    batt_dispatch=None,  # TODO: enable battery switch earlier
                )
            else:
                # fetch 8760 cambium values for FOM
                market_profile = fetch_cambium_values(
                    row,
                    generation_hourly,
                    config.project.settings.CAMBIUM_DATA_DIR,
                    config.CAMBIUM_VALUE,
                )

                row = process_fom(
                    row=row,
                    tech=tech,
                    generation_hourly=generation_hourly,
                    market_profile=market_profile,
                    pysam_outputs=config.pysam.outputs.fom,
                    en_batt=False,
                    batt_dispatch=None,  # TODO: enable battery switch earlier
                )

            # store results in dictionary
            results[f"{tech}_breakeven_cost_{sector}"] = row["breakeven_cost_usd_p_kw"]
            results[f"{tech}_pysam_outputs_{sector}"] = row["additional_pysam_outputs"]

        return (row["gid"], results)

    except Exception as e:
        log.exception(e)
        log.info("\n")
        log.info("Problem row:")
        log.info(row)
        log.info(results)

        return (row["gid"], {})
