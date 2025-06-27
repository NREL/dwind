import numpy as np
import pandas as pd
import json
import h5py as h5
from scipy.spatial import KDTree


"""
RETAIL_RATE_INPUT_TABLE: 
- old data: 'diffusion_shared.ATB20_Mid_Case_retail' (dgen_db_fy17q3_dwind)
- 2023 data (source): https://github.com/NREL/dgen/blob/master/dgen_os/input_data/elec_prices/ATB23_Mid_Case_retail.csv
- 2023 data (destination): /projects/dwind/configs/costs/atb23/ATB23_Mid_Case_retail.csv and /projects/dwind/configs/costs/atb24/AEO23_Mid_Case_retail.csv
- 2024 data (not available as of 01/09/2025, pretty sure this came from AEO and not ATB) 

WHOLESALE_RATE_INPUT_TABLE:
- old data: 'diffusion_shared.ATB20_Mid_Case_wholesale'
- 2023 data (source): https://github.com/NREL/dgen/blob/master/dgen_os/input_data/wholesale_electricity_prices/ATB23_Mid_Case_wholesale.csv
- 2023 data (destination): /projects/dwind/configs/costs/atb23/ATB23_Mid_Case_wholesale.csv and /projects/dwind/configs/costs/atb24/AEO23_Mid_Case_wholesale.csv
- 2024 data (not available as of 01/09/2025, pretty sure this came from AEO and not ATB)

FINANCIAL_INPUTS_TABLE (not actually used in valuation.py):
- old data: 'diffusion_shared.financing_atb_FY20'
- 2023 data (source): https://github.com/NREL/dgen/blob/master/dgen_os/input_data/financing_terms/financing_atb_FY23.csv
- 2023 data (destination): /projects/dwind/configs/costs/atb23/ATB23_financing.csv
- 2024 data (source): ATB 2024
- 2024 data (destination): /projects/dwind/configs/costs/atb24/ATB24_financing_baseline_2025.json

DEPREC_INPUTS_TABLE:
- old data: 'diffusion_shared.deprec_sch_FY17'
- 2023 data (source): /projects/dwind/configs/costs/atb23/ATB23_depc_factors_orig.csv
- 2023 data (destination): /projects/dwind/configs/costs/atb23/ATB23_depc_factors.csv
- 2024 data (source): https://github.com/NREL/dgen/blob/master/dgen_os/input_data/depreciation_schedules/deprec_sch_FY24.csv 
- 2024 data (destination): /projects/dwind/configs/costs/atb24/ATB24_depc_factors.csv

WIND_PRICE_INPUT_TABLE: 
- old data: 'diffusion_shared.wind_prices_1_reference'
- 2023 data (source): /projects/dwind/configs/costs/atb23/ATB23_DistributedWind.csv
- 2023 data (destination): /projects/dwind/configs/costs/atb23/ATB23_wind_prices.csv
- 2024 data (source): /projects/dwind/configs/costs/atb23/ATB24_DistributedWind.csv
- 2024 data (destination): /projects/dwind/configs/costs/atb24/ATB24_wind_prices.csv

CAMBIUM DATA:
- old data: Cambium 2020, Mid-Case, hourly, balancing areas
- new data: Cambium 2023, Mid-Case, hourly, balancing areas
- 2023 data (source): https://scenarioviewer.nrel.gov/?project=0f92fe57-3365-428a-8fe8-0afc326b3b43&mode=download&layout=Default
- 2023 data (destination): /projects/dwind/data/cambium_processed/Cambium23_MidCase_pca_2025_processed.pkl
- 2024 data (not available as of 01/09/2025)

RESOURCE DATA
- 2012 to 2018 weather year

CONSUMPTION:
- old data: NREL (2016)
- new data (source): ResStock/ComStock (2024)
- new data (destination): /projects/dwind/data/consumption/2024/load_scaling_factors.csv
- notes: ComStock uses 2018 weather year, ResStock uses TMY
- TODO: Update load growth parameters for future BTM scenarios?
"""

def atb_retail():
    def atb_retail_2023():
        # ba, year, elec_price_res, elec_price_com, elec_price_ind
        # county_id, year, elec_price_multiplier, sector_abbr
        df = pd.read_csv('atb23/ATB23_Mid_Case_retail_orig.csv')

        # res
        df_res = df[['ba', 'year', 'elec_price_res']]
        df_res = df_res.rename(columns={'elec_price_res': 'elec_price_multiplier'})
        df_res['sector_abbr'] = 'res'

        # com
        df_com = df[['ba', 'year', 'elec_price_com']]
        df_com = df_com.rename(columns={'elec_price_com': 'elec_price_multiplier'})
        df_com['sector_abbr'] = 'com'

        # ind
        df_ind = df[['ba', 'year', 'elec_price_ind']]
        df_ind = df_ind.rename(columns={'elec_price_ind': 'elec_price_multiplier'})
        df_ind['sector_abbr'] = 'ind'

        df = pd.concat([df_res, df_com, df_ind])

        # merge cnty to ba lkup
        lkup = pd.read_csv('county_to_ba_mapping.csv')
        df = df.merge(lkup, on='ba', how='right')

        df = df[['county_id', 'year', 'elec_price_multiplier', 'sector_abbr']]
        df.to_csv('atb23/ATB23_Mid_Case_retail.csv', index=False)
    
    def check_2023():
        f = "/projects/dwind/configs/costs/atb23/ATB23_Mid_Case_retail_orig.csv"
        df = pd.read_csv(f)
        for i, row in df.iterrows():
            res = row["elec_price_res"]
            com = row["elec_price_com"]
            ind = row["elec_price_ind"]
            if res != com and com != ind and res != ind:
                print(i)  # all the same values

    def atb_retail_2024():
        f_in = "/projects/dwind/configs/costs/atb23/AT23_Mid_Case_retail.csv"
        df = pd.read_csv(f_in)
        f_out = "/projects/dwind/configs/costs/atb24/AEO23_Mid_Case_retail.csv"
        df.to_csv(f_out, index=False)

   
def atb_wholesale():
    def atb_wholesale_2023():
        df = pd.read_csv('atb23/ATB23_Mid_Case_wholesale_orig.csv')

        yrs = range(2014, 2051)
        dfs = []
        
        for yr in yrs:
            yr_df = df[['ba', str(yr)]]
            yr_df['year'] = yr
            yr_df['wholesale_elec_price_dollars_per_kwh'] = df[str(yr)]
            yr_df = yr_df[['ba', 'year', 'wholesale_elec_price_dollars_per_kwh']]
            dfs.append(yr_df)
            
        df = pd.concat(dfs)
        
        # merge cnty to ba lkup
        lkup = pd.read_csv('county_to_ba_mapping.csv')
        df = df.merge(lkup, on='ba', how='right')
        
        df = df[['county_id', 'year', 'wholesale_elec_price_dollars_per_kwh']]
        df.to_csv('atb23/ATB23_Mid_Case_wholesale.csv', index=False)

    def atb_wholesale_2024():
        f_in = "/projects/dwind/configs/costs/atb23/AT23_Mid_Case_wholesale.csv"
        df = pd.read_csv(f_in)
        f_out = "/projects/dwind/configs/costs/atb24/AEO23_Mid_Case_wholesale.csv"
        df.to_csv(f_out, index=False)


def atb_financing():
    def atb_financing_2023():
        df = pd.read_csv('atb23/ATB23_financing_orig.csv')
        print(df)
        
        res_df = df[[
            "year",
            "economic_lifetime_yrs",
            "loan_term_yrs_res",
            "loan_interest_rate_res",
            "down_payment_fraction_res",
            "real_discount_rate_res",
            "tax_rate_res"
        ]]
        res_df.rename(columns={
            "loan_term_yrs_res": "loan_term_yrs",
            "loan_interest_rate_res": "loan_interest_rate",
            "down_payment_fraction_res": "down_payment_fraction",
            "real_discount_rate_res": "real_discount_rate",
            "tax_rate_res": "tax_rate"
        }, inplace=True)
        res_df["sector_abbr"] = "res"
        print(res_df)

        com_df = df[[
            "year",
            "economic_lifetime_yrs",
            "loan_term_yrs_nonres",
            "loan_interest_rate_nonres",
            "down_payment_fraction_nonres",
            "real_discount_rate_nonres",
            "tax_rate_nonres"
        ]]
        com_df.rename(columns={
            "loan_term_yrs_nonres": "loan_term_yrs",
            "loan_interest_rate_nonres": "loan_interest_rate",
            "down_payment_fraction_nonres": "down_payment_fraction",
            "real_discount_rate_nonres": "real_discount_rate",
            "tax_rate_nonres": "tax_rate"
        }, inplace=True)
        com_df["sector_abbr"] = "com"

        ind_df = df[[
            "year",
            "economic_lifetime_yrs",
            "loan_term_yrs_nonres",
            "loan_interest_rate_nonres",
            "down_payment_fraction_nonres",
            "real_discount_rate_nonres",
            "tax_rate_nonres"
        ]]
        ind_df.rename(columns={
            "loan_term_yrs_nonres": "loan_term_yrs",
            "loan_interest_rate_nonres": "loan_interest_rate",
            "down_payment_fraction_nonres": "down_payment_fraction",
            "real_discount_rate_nonres": "real_discount_rate",
            "tax_rate_nonres": "tax_rate"
        }, inplace=True)
        ind_df["sector_abbr"] = "ind"

        final_df = pd.concat([res_df, com_df, ind_df])
        final_df.to_csv("atb23/ATB23_financing.csv", index=False)
    
    # def atb_financing_2024():


def atb_wind_prices(scenario="Moderate", case="Market", crpyears=30):
    def atb_wind_prices_2023():
        df = pd.read_csv("/projects/dwind/configs/costs/atb23/ATB23_DistributedWind.csv")
        df = df[df["scenario"] == scenario]
        df = df[df["core_metric_case"] == case]
        df = df[df["crpyears"] == crpyears]

        turbine_heights = {
            2.5: 35,
            5: 35,
            10: 35,
            20: 35,
            50: 45,
            100: 45,
            250: 55,
            500: 55,
            750: 55,
            1000: 80,
            1500: 80
        }

        turbine_classes = {
            2.5: "Residential DW ",
            5: "Residential DW ",
            10: "Residential DW ",
            20: "Residential DW ",
            50: "Commercial DW ",
            100: "Commercial DW ",
            250: "Midsize DW ",
            500: "Midsize DW ",
            750: "Midsize DW ",
            1000: "Large DW ",
            1500: "Large DW "
        }

        dfs = []
        for year in np.unique(df.core_metric_variable.values):
            df_yr = df[df["core_metric_variable"] == year]

            for turbine in turbine_heights:
                turbine_class = turbine_classes[turbine]
                df_turbine = df_yr[df_yr["technology_alias"] == turbine_class]

                capex = df_turbine[df_turbine["core_metric_parameter"] == "CAPEX"].value.values[0]
                om = df_turbine[df_turbine["core_metric_parameter"] == "Fixed O&M"].value.values[0]
                
                new_df = pd.DataFrame({
                    "year": [year],
                    "turbine_size_kw": [turbine],
                    "capital_cost_dollars_per_kw": [capex],
                    "fixed_om_dollars_per_kw_per_yr": [om],
                    "variable_om_dollars_per_kwh": [0],
                    "default_tower_height_m": [turbine_heights[turbine]],
                    "cost_for_higher_towers_dollars_per_kw_per_m": 0  # no idea how this was calculated
                })
                dfs.append(new_df)

        df_final = pd.concat(dfs)
        df_final.to_csv("/projects/dwind/configs/costs/atb23/ATB23_wind_prices.csv", index=False)

    def atb_wind_prices_2024():
        df = pd.read_csv("/projects/dwind/configs/costs/atb24/ATB24_DistributedWind.csv")
        df = df[df["scenario"] == scenario]
        df = df[df["core_metric_case"] == case]
        df = df[df["crpyears"] == crpyears]

        turbine_heights = {
            2.5: 35,
            5: 35,
            10: 35,
            20: 35,
            50: 45,
            100: 45,
            250: 55,
            500: 55,
            750: 55,
            1000: 80,
            1500: 80
        }

        turbine_classes = {
            2.5: "Residential DW ",
            5: "Residential DW ",
            10: "Residential DW ",
            20: "Residential DW ",
            50: "Commercial DW ",
            100: "Commercial DW ",
            250: "Midsize DW ",
            500: "Midsize DW ",
            750: "Midsize DW ",
            1000: "Large DW ",
            1500: "Large DW "
        }

        dfs = []
        for year in np.unique(df.core_metric_variable.values):
            df_yr = df[df["core_metric_variable"] == year]

            for turbine in turbine_heights:
                turbine_class = turbine_classes[turbine]
                df_turbine = df_yr[df_yr["technology_alias"] == turbine_class]

                capex = df_turbine[df_turbine["core_metric_parameter"] == "CAPEX"].value.values[0]
                om = df_turbine[df_turbine["core_metric_parameter"] == "Fixed O&M"].value.values[0]
                
                new_df = pd.DataFrame({
                    "year": [year],
                    "turbine_size_kw": [turbine],
                    "capital_cost_dollars_per_kw": [capex],
                    "fixed_om_dollars_per_kw_per_yr": [om],
                    "variable_om_dollars_per_kwh": [0],
                    "default_tower_height_m": [turbine_heights[turbine]],
                    "cost_for_higher_towers_dollars_per_kw_per_m": 0  # no idea how this was calculated
                })
                dfs.append(new_df)

        df_final = pd.concat(dfs)
        df_final.to_csv("/projects/dwind/configs/costs/atb24/ATB24_wind_prices.csv", index=False)


def atb_deprec_factors():
    def atb_deprec_factors_2023():
        df = pd.read_csv("/projects/dwind/configs/costs/atb23/ATB23_depc_factors_orig.csv")
        df = df.drop(columns=["technology", "scenario", "year"])
        df = df.T

        dfs = []
        for i, row in df.iterrows():
            deprec_sch = [0] * 6 
            for j in range(6):
                deprec_sch[j] = row[j]

            for sector in ["res", "com", "ind"]:
                new_df = pd.DataFrame({
                    "year": [i],
                    "sector_abbr": [sector],
                    "deprec_sch": [deprec_sch]
                })
                dfs.append(new_df)

        df_final = pd.concat(dfs)
        df_final.to_csv("/projects/dwind/configs/costs/atb23/ATB23_depc_factors.csv", index=False)
    
    def atb_deprec_factors_2024():
        def get_sch(row):
            deprec_sch = [0] * 6 
            for j in range(1, 7):
                deprec_sch[j-1] = row[str(j)]
            row["deprec_sch"] = deprec_sch
            return row

        f_in = "/projects/dwind/configs/costs/atb24/ATB24_depc_factors_orig.csv"
        df_in = pd.read_csv(f_in)
        df = df_in.apply(get_sch, axis=1)
        
        df_out = df[["year", "sector_abbr", "deprec_sch"]]
        f_out = "/projects/dwind/configs/costs/atb24/ATB24_depc_factors.csv"
        df_out.to_csv(f_out, index=False)


def cambium():
    def check_old():
        f_2022 = "/projects/dwind/data/cambium_processed/StdScen20_MidCase_pca_2022_processed.pkl"
        f_2035 = "/projects/dwind/data/cambium_processed/StdScen20_MidCase_pca_2035_processed.pkl"

        df_old_2022 = pd.read_pickle(f_2022)
        df_old_2035 = pd.read_pickle(f_2035)

        print(df_old_2022)  # 134 rows, 4 columns
        for c in df_old_2022.columns: print(c) #  year, pca, variable, value
        print(np.unique(df_old_2022.year.values))  # '2022' or '2035'
        print(len(np.unique(df_old_2022.pca.values)))  # 134 balancing areas
        print(np.unique(df_old_2022.variable.values))  # 'cambium_grid_value'
        print(len(df_old_2022.value.values[0]))  # 8760

    # 8760 of Energy market revenue input, Price($/MWh), financial.Revenue.mp_energy_market_revenue
    # "cambium_energy_value" == "energy_cost_enduse"
    # "cambium_grid_value" == "total_cost_enduse"
    for yr in ["2025", "2035"]:
        dfs = []
        for i in range(1, 135):
            f = f"/projects/dwind/data/cambium_raw/Cambium23_MidCase/Cambium23_MidCase_hourly_p{i}_{yr}.csv"
            df = pd.read_csv(f, header=5)
            val = df.total_cost_enduse.values

            df_new = pd.DataFrame({
                "year": [yr],
                "pca": [f"p{i}"],
                "variable": ["cambium_grid_value"],
                "value": [val]
            })
            dfs.append(df_new)
        
        df_final = pd.concat(dfs)
        df_final = df_final.reset_index(drop=True)
        df_final.to_pickle(f"/projects/dwind/data/cambium_processed/Cambium23_MidCase_pca_{yr}_processed.pkl")


def resource():
    def create_project_points():
        f_new = f"/datasets/WIND/conus/v2.0.0/2018/conus_2018_wind_hourly.h5"
        with h5.File(f_new, 'r') as hf:
            meta = pd.DataFrame(hf['meta'][...])
            meta["country"] = meta.country.apply(lambda x: x.decode())
            meta["state"] = meta.state.apply(lambda x: x.decode())
            meta = meta[meta["country"] == 'United States']
            meta = meta[meta["state"] != "None"]
            meta = meta[meta["offshore"] == 0]
            index = meta.index.values

        for config in ["res", "com", "mid", "large"]:
            df = pd.DataFrame({
                "gid": index,
            })
            df["config"] = config
            f = f"/projects/dwind/configs/rev/wind/wind_project_points_conus_2018_{config}.csv"
            df.to_csv(f)

    def wtk_2012_to_2018():
        # lookup wtk 2012 indexes to 2018 indexes
        f_old = "/projects/dwind/data/rev/rev_res_generation_2012.h5"
        with h5.File(f_old, 'r') as hf:
            meta = pd.DataFrame(hf['meta'][...])
            meta["country"] = meta.country.apply(lambda x: x.decode())
            meta["state"] = meta.state.apply(lambda x: x.decode())
            meta["county"] = meta.county.apply(lambda x: x.decode())

            # meta = meta[meta["country"] == 'United States']
            # meta = meta[meta["state"] != "None"]
            # meta = meta[meta["offshore"] == 0]
            df_old = meta[["latitude", "longitude", "state", "county"]]

        f_new = "/projects/dwind/data/rev/rev_res_generation_2018.h5"
        with h5.File(f_new, 'r') as hf:
            meta = pd.DataFrame(hf['meta'][...])
            meta["country"] = meta.country.apply(lambda x: x.decode())
            meta["state"] = meta.state.apply(lambda x: x.decode())
            meta["county"] = meta.county.apply(lambda x: x.decode())

            # meta = meta[meta["country"] == 'United States']
            # meta = meta[meta["state"] != "None"]
            # meta = meta[meta["offshore"] == 0]
            df_new = meta[["latitude", "longitude", "state", "county"]]

        pts_0 = np.array(list(zip(df_old.longitude.values, df_old.latitude.values)))
        pts_1 = np.array(list(zip(df_new.longitude.values, df_new.latitude.values)))

        btree = KDTree(pts_1)
        _, idx = btree.query(pts_0)

        nearest_value_idx = df_new.iloc[idx]
        nearest_value_idx = nearest_value_idx.reset_index(drop=False, names="rev_index_wind_2018")
        nearest_value_idx["index"] = nearest_value_idx.index

        df_old = df_old.reset_index(drop=False, names="rev_index_wind_2012")
        df_old["index"] = df_old.index
        df_merged = df_old.merge(nearest_value_idx, on="index", how="left")

        df_merged = df_merged[["rev_index_wind_2012", "rev_index_wind_2018"]]
        df_merged.to_csv("/projects/dwind/configs/rev/wind/lkup_rev_index_2012_to_2018.csv", index=False)

    def create_new_resource_file():
        # create new .h5 file from /datasets/WIND/conus/v2.0.0/2018/conus_2018_wind_hourly.h5
        # and /datasets/WIND/conus/v2.0.0/2018/conus_2018_non_wind_hourly.h5

        f_wind = "/datasets/WIND/conus/v2.0.0/2018/conus_2018_wind_hourly.h5"
        f_non_wind = "/datasets/WIND/conus/v2.0.0/2018/conus_2018_non_wind_hourly.h5"
        f_out = "/scratch/jlockshi/wtk_led_conus_2018_hourly.h5"

        cols_wind = [
            'coordinates',  # wind
            'meta',  # wind
            'time_index',  # wind
            'winddirection_10m',  # wind
            'winddirection_20m',  # wind
            'winddirection_40m',  # wind 
            'winddirection_60m',  # wind 
            'winddirection_80m',  # wind 
            'winddirection_100m', # wind  
            'windspeed_10m', # wind 
            'windspeed_20m', # wind 
            'windspeed_40m', # wind
            'windspeed_60m', # wind
            'windspeed_80m'  # wind
            'windspeed_100m' # wind
        ]
        
        cols_non_wind = [
            'inversemoninobukhovlength_2m',  # non_wind
            'precipitation_0m', # non_wind
            'pressure_0m',  # non_wind
            'pressure_100m',  # non_wind
            'pressure_200m',  # non_wind
            'relativehumidity_2m',  # non_wind
            'temperature_2m',  # non_wind
            'temperature_20m',  # non_wind
            'temperature_40m',  # non_wind
            'temperature_60m',  # non_wind
            'temperature_80m',  # non_wind
            'temperature_100m', # non_wind
        ]

        with h5.File(f_wind, 'r') as hf1, h5.File(f_non_wind, 'r') as hf2, h5.File(f_out, 'w') as hfout:
            # copy datasets from the first file
            for dataset_name in cols_wind:
                if dataset_name in hf1:
                    hf1.copy(dataset_name, hfout)

            # copy datasets from the second file
            for dataset_name in cols_non_wind:
                if dataset_name in hf2:
                    hf2.copy(dataset_name, hfout)

    # create_new_resource_file()

    f = "/scratch/jlockshi/wtk_led_conus_2018_hourly.h5"
    with h5.File(f, 'r') as hf:
        print(hf.keys())


def consumption():
    """Applies load scaling factors at parcel level.
    Steps:
    - Finds 2016 annual load by state/building type (used in 2022 scenarios)
    - Finds 2024 annual load by state/building type
    - Calculate scaling factor by state/building type
    - Updates model by applying scaling factor to agents by state/building type
    """
    def load_by_state_bldg_2016():
        # find block to pgid
        block_to_pgid_df = pd.read_csv(
            "/projects/dwind/configs/sizing/wind/lkup_block_to_pgid_2020.csv",
            dtype={"fips_block": str, "pgid": str}
        )[["fips_block", "pgid"]]

        # find pgid to hdf index
        pgid_to_hdf_df = pd.read_csv(
            "/projects/dwind/data/lkup_pgid_to_hdf_index.csv",
            dtype={"pgid": str}

        )[["pgid", "hdf_index"]]

        df = block_to_pgid_df.merge(pgid_to_hdf_df, on="pgid", how="left")
        df = df[~df["hdf_index"].isna()]
        df["hdf_index"] = df["hdf_index"].astype(int)
        df["state_fips"] = df['fips_block'].apply(lambda x: x[:2])

        # find hdf index to annual load
        hdf_to_load_df = pd.read_csv(
            "/projects/dwind/data/lkup_hdf_index_to_load_kwh.csv",
            dtype={"hdf_index": int}
        )[["hdf_index", "crb_model", "max_demand_kw", "load_kwh"]]
        
        df = df.merge(hdf_to_load_df, on="hdf_index", how="left")

        # join crb reference floor areas
        crb_to_area_df = pd.read_csv(
            "/projects/dwind/data/crb_model_to_floor_area.csv",
            dtype={"ref_floor_area_sq_ft": int}
        )[["crb_model", "ref_floor_area_sq_ft"]]

        df = df.merge(crb_to_area_df, on="crb_model", how="left")

        # map crb building types to res/comstock building types
        bldg_types = {
            "reference": "Single Family",
            "midrise_apartment": "Multifamily",
            "primary_school": "Education",
            "secondary_school": "Education",
            "quick_service_restaurant": "Food Service",
            "full_service_restaurant": "Food Service",
            "out_patient": "Healthcare",
            "hospital": "Healthcare",
            "small_hotel": "Lodging",
            "large_hotel": "Lodging",
            "stand_alone_retail": "Mercantile",
            "supermarket": "Mercantile",
            "strip_mall": "Mercantile",
            "small_office": "Office",
            "medium_office": "Office",
            "large_office": "Office",
            "warehouse": "Warehouse and Storage"
        }
        df["bldg_type"] = df["crb_model"].map(bldg_types)

        # group by state and building type
        gb = df.groupby(["state_fips", "bldg_type"]).agg({
            "ref_floor_area_sq_ft": "sum",
            "load_kwh": "sum"
        })

        gb.columns = ["bldg_sqft_2022", "load_kwh_2022"]
        gb = gb.reset_index(drop=False)
        gb["kwh_per_sqft_2022"] = gb["load_kwh_2022"] / gb["bldg_sqft_2022"]

        return gb
    
    def load_by_state_bldg_2024_com():
        # https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2024%2Fcomstock_amy2018_release_1%2Fmetadata_and_annual_results%2Fnational%2Fcsv%2F
        f_com = "/projects/dwind/data/consumption/2024/comstock_amy2018_release_1/baseline_basic_metadata_and_annual_results.csv"
        df_com = pd.read_csv(f_com)

        df_com["state_fips"] = df_com["in.nhgis_tract_gisjoin"].apply(lambda x: x[1:3])
        df_com["bldg_type"] = df_com["in.comstock_building_type_group"]
        df_com["bldg_sqft"] = df_com["calc.weighted.sqft"]
        df_com["btu"] = df_com["calc.weighted.electricity.total.energy_consumption..tbtu"] * 1e12
        df_com["load_kwh"] = df_com["btu"] / 3412

        gb = df_com.groupby(["state_fips", "bldg_type"]).agg({
            "bldg_sqft": "sum", 
            "load_kwh": "sum"
        })
        
        gb.columns = [
            "bldg_sqft_2024", 
            "load_kwh_2024"
        ]

        gb = gb.reset_index(drop=False)
        gb["kwh_per_sqft_2024"] = gb["load_kwh_2024"] / gb["bldg_sqft_2024"]

        return gb
    
    def load_by_state_bldg_2024_res():
        # https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2024%2Fresstock_tmy3_release_2%2Fmetadata_and_annual_results%2Fnational%2Fcsv%2F
        f_res = "/projects/dwind/data/consumption/2024/resstock_tmy3_release_2/baseline_metadata_and_annual_results.csv"
        df_res = pd.read_csv(f_res)

        res_types = {
            "Single-Family Detached": "Single Family",
            "Single-Family Attached": "Single Family",
            "Multi-Family with 2 - 4 Units": "Multifamily",
            "Multi-Family with 5+ Units": "Multifamily",
            "Mobile Home": "Single Family"
        }

        df_res["state_fips"] = df_res["in.county"].apply(lambda x: x[1:3])
        df_res["bldg_type_recs"] = df_res["in.geometry_building_type_recs"]
        df_res["bldg_type"] = df_res["bldg_type_recs"].map(res_types)
        df_res["bldg_sqft"] = df_res["in.sqft"] * df_res["weight"]
        df_res["load_kwh"] = df_res["out.electricity.total.energy_consumption.kwh"] * df_res["weight"]

        gb = df_res.groupby(["state_fips", "bldg_type"]).agg({
            "bldg_sqft": "sum",
            "load_kwh": "sum"
        })
        
        gb.columns = [
            "bldg_sqft_2024",
            "load_kwh_2024"
        ]
        
        gb = gb.reset_index(drop=False)
        gb["kwh_per_sqft_2024"] = gb["load_kwh_2024"] / gb["bldg_sqft_2024"]

        """
        6.8 kwh/sqft (resstock 2024)
        4.9 kwh/sqft (fom_baseline_2022 dwind results from 2024 - res only)
        0.69 kwh/sqft (fom_baseline_2022 dwfs results from 2022 - res only)
        """

        return gb
    
    # find 2016 annual load by state/building type (used in 2022 scenarios)
    load_2016 = load_by_state_bldg_2016() 

    # find 2024 annual load by state/building type
    load_2024_com = load_by_state_bldg_2024_com()
    load_2024_res = load_by_state_bldg_2024_res()
    load_2024 = pd.concat([load_2024_com, load_2024_res])

    # calculate scaling factor by state/building type
    load = load_2016.merge(
        load_2024, 
        on=["state_fips", "bldg_type"], 
        how="left"
    )

    load["load_sf_2024"] = load["kwh_per_sqft_2024"] / load["kwh_per_sqft_2022"]
    load = load[[
        "state_fips", 
        "bldg_type", 
        "kwh_per_sqft_2022", 
        "kwh_per_sqft_2024", 
        "load_sf_2024"
    ]]


    food = load[load["bldg_type"] == "Food Services"]
    food["bldg_type"] = "Food Sales"

    hotel = load[load["bldg_type"] == "Lodging"]
    hotel["bldg_type"] = "Hotel"

    mobile = load[load["bldg_type"] == "Single-Family"]
    mobile["bldg_type"] = "Mobile Home"

    load = pd.concat([load, food, hotel, mobile])

    load = load.sort_values(["state_fips"])
    f = "/projects/dwind/data/consumption/2024/load_scaling_factors.csv"
    load.to_csv(f, index=False)

