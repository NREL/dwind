import os
import numpy as np
import pandas as pd


def collect_by_state(scenario, state):
    app = 'fom' if 'fom' in scenario else 'btm'
    p_dwind = '/projects/dwind/runs_2025'
    p_dwind = os.path.join(p_dwind, state, app)
    
    p_chunks = os.path.join(p_dwind, f'chunk_files_{state}_{scenario}')
    name = f'{state}_{scenario}'
    p_result = os.path.join(p_dwind, f'{name}.pkl')
    
    dfs = []
    for f in os.listdir(p_chunks):
        if name in f:
            p_file = os.path.join(p_chunks, f)
            df = pd.read_pickle(p_file)
            dfs.append(df)
            
    df = pd.concat(dfs)
    df.to_pickle(p_result)
    print(df)


def collect_by_priority(sector: str, scenario: str, year: int, location: str, rerun:bool = False):
    full_scenario = f"{location}_{sector}_{scenario}_{year}"

    p_dwind = os.path.join(
        '/projects/dwind/runs_2025', 
        f'{location}', 
        sector,
        f'{scenario}_{year}'
    )

    if rerun:
        p_dwind = os.path.join(p_dwind, "rerun")
    
    p_chunks = os.path.join(p_dwind, f'chunk_files_{full_scenario}')
    p_result = os.path.join(p_dwind, f'{location}.pkl')
    
    dfs = []
    for f in os.listdir(p_chunks):
        if location in f:
            p_file = os.path.join(p_chunks, f)
            df = pd.read_pickle(p_file)
            
            if sector == 'btm':
                # df.rename(
                #     columns={
                #         'wind_turbine_kw_btm': 'wind_turbine_kw', 
                #         'wind_size_kw_btm': 'wind_size_kw'
                #     },
                #     inplace=True
                # )
                df.drop(columns=['consumption_hourly', 'deprec_sch'], inplace=True)
                apps = ['BTM', 'BTM, FOM', 'BTM, FOM, Utility']
            else:
                apps = ['BTM, FOM', 'BTM, FOM, Utility', 'FOM, Utility']

            # join back load application mapping
            load_df = pd.read_csv(
                '/projects/dwind/data/parcel_landuse_load_application_mapping.csv')
            load_df = load_df[['land_use', 'application']]

            df = df.drop(columns='application')
            df = df.merge(load_df, on='land_use', how='left')
            df = df[df['application'].isin(apps)]
                
            dfs.append(df)
        
    if len(dfs) == 0:
        print(f"No results found in: {p_chunks}")
        return
            
    df = pd.concat(dfs)
    
    # remove duplicates
    df = df.sort_values(f'wind_breakeven_cost_{sector}', ascending=False)
    df = df.drop_duplicates(subset=['gid'], keep='first')

    # fix state_abbr and census_division_abbr
    msk = df["state"] == "new_york"
    df.loc[msk, "state_abbr"] = "NY"
    df.loc[msk, "census_division_abbr"] = "MA"

    msk = df["state"] == "north_dakota"
    df.loc[msk, "state_abbr"] = "ND"
    df.loc[msk, "census_division_abbr"] = "WNC"

    msk = df["state"] == "south_dakota"
    df.loc[msk, "state_abbr"] = "SD"
    df.loc[msk, "census_division_abbr"] = "WNC"
    
    msk = df["state"] == "new_mexico"
    df.loc[msk, "state_abbr"] = "NM"
    df.loc[msk, "census_division_abbr"] = "M"

    msk = df["state"] == "new_hampshire"
    df.loc[msk, "state_abbr"] = "NH"
    df.loc[msk, "census_division_abbr"] = "NE"

    msk = df["state"] == "rhode_island"
    df.loc[msk, "state_abbr"] = "RI"
    df.loc[msk, "census_division_abbr"] = "NE"

    msk = df["state"] == "districtofcolumbia"
    df.loc[msk, "state_abbr"] = "DC"
    df.loc[msk, "census_division_abbr"] = "SA"

    msk = df["state"] == "new_jersey"
    df.loc[msk, "state_abbr"] = "NJ"
    df.loc[msk, "census_division_abbr"] = "MA"

    msk = df["state"] == "west_virginia"
    df.loc[msk, "state_abbr"] = "WV"
    df.loc[msk, "census_division_abbr"] = "SA"

    msk = df["state"] == "north_carolina"
    df.loc[msk, "state_abbr"] = "NC"
    df.loc[msk, "census_division_abbr"] = "SA"

    msk = df["state"] == "south_carolina"
    df.loc[msk, "state_abbr"] = "SC"
    df.loc[msk, "census_division_abbr"] = "SA"

    # keep cols
    cols = [
        "gid",
        "pgid",
        "hdf_index",
        "county_id",
        "lat",
        "lon",
        "state_abbr",
        "census_division_abbr",
        "sector_abbr",
        "application",
        "state",
        "county_file",
        "fips_code",
        "rev_gid_wind",
        "parcel_size_acres",
        "land_use",
        "crb_model",
        "canopy_pct",
        "canopy_ht_m",
        "turbine_height_m",
        "turbine_instances",
        "floor_area_sq_ft",
        "num_floors",
        "max_demand_kw",
        "load_kwh",
        "turbine_class",
        "rev_index_wind",
        "wind_naep",
        "wind_cf",
        "wind_aep",
        "wind_size_kw_techpot",
        "wind_size_kw_fom",
    ]
    
    if sector == 'btm':
        cols.append("rate_id_alias")
        cols.append("tariff_name")
        cols.append("compensation_style")
        cols.append("nem_system_kw_limit")
        
        cols.append("wind_aep_btm")
        cols.append("wind_size_kw_btm")
        cols.append("wind_turbine_kw_btm")
        cols.append("wind_breakeven_cost_btm")
        cols.append("wind_pysam_outputs_btm")
    else:
        cols.append("wind_aep_fom")
        cols.append("wind_size_kw")
        cols.append("wind_turbine_kw")
        cols.append("wind_breakeven_cost_fom")
        cols.append("wind_pysam_outputs_fom")

    # save to pickle
    df = df[cols]
    df = df.reset_index(drop=True)
    df.to_pickle(p_result)
    print(df)
