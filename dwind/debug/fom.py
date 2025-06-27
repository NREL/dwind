import os
import sys
import numpy as np
import pandas as pd


def fetch_cambium_values():
    cambium_data_dir = '/projects/dwind/kmccabe/github/dwfs/data/cambium_processed'
    cambium_value = 'cambium_grid_value'
    cambium_scenario = 'StdScen20_MidCase'
    year = 2022
    ba = 'p129'
    
    # read processed cambium dataframe from pickle
    cambium_f = os.path.join(
        cambium_data_dir,
        cambium_scenario + f'_pca_{year}_processed.pkl'
    )
    cambium_df = pd.read_pickle(cambium_f)

    cambium_df['year'] = cambium_df['year'].astype(str)
    cambium_df['pca'] = cambium_df['pca'].astype(str)
    cambium_df['variable'] = cambium_df['variable'].astype(str)

    # filter on pca, desired year, cambium variable
    mask = (
        (cambium_df['year'] == str(year)) &
        (cambium_df['pca'] == str(ba)) &
        (cambium_df['variable'] == cambium_value)
    )

    cambium_output = cambium_df[mask]
    cambium_output = cambium_output.reset_index(drop=True)
    cambium_output = cambium_output['value'].values[0]
    
    print(cambium_output)
    

# fetch_cambium_values()

# f = '/projects/dwind/agents/vermont/agents_dwind_fom.pickle'
# df = pd.read_pickle(f)
# print(df)  # 14,320 rows
# df = df[~df['ba'].isna()]
# print(df)  # 13,499 rows
# print(np.unique(df.ba.values))
# 13,499 / 14,320 = 94%


def check_vars(f):
    df = pd.read_pickle(f)
    for c in df.columns: print(c)
    print(df)
    print(np.unique(df.ba.values))
    
    
def fix_arkansas():
    f = '/projects/dwind/configs/costs/dgen_county_fips_mapping.csv'
    df = pd.read_csv(f, dtype={'fips_code': str, 'state_fips': str})
    df = df[df['state_fips'] == '05']
    df = df[['fips_code', 'county_id']]
    
    f = '/projects/dwind/configs/costs/county_to_ba_mapping.csv'
    ba = pd.read_csv(f)
    
    df = df.merge(ba, on='county_id', how='left')
    
    f = '/projects/dwind/agents/arkansas/agents_dwind_fom.pickle'
    a_df = pd.read_pickle(f)
    a_df = a_df.drop(columns=['county_id', 'ba'])
    a_df = a_df.merge(df, on='fips_code', how='left')
    a_df = a_df[[
        'gid',
        'pgid',
        'hdf_index',
        'lat',
        'lon',
        'state_abbr',
        'census_division_abbr',
        'sector_abbr',
        'application',
        'weight',
        'rev_gid_wind',
        'rev_gid_solar',
        'parcel_size_acres',
        'land_use',
        'crb_model',
        'canopy_pct',
        'canopy_ht_m',
        'turbine_height_m',
        'turbine_instances',
        'tilt_btm',
        'azimuth_btm',
        'tilt_fom',
        'azimuth_fom',
        'developable_roof_sqft',
        'solar_groundmount',
        'wind_turbine_kw',
        'wind_size_kw',
        'wind_size_kw_fom',
        'solar_size_kw_fom',
        'floor_area_sq_ft',
        'num_floors',
        'power_curve_1',
        'power_curve_2',
        'power_curve_interp_factor',
        'max_demand_kw',
        'load_kwh',
        'wind_naep',
        'wind_cf',
        'wind_aep',
        'state',
        'county_file',
        'fips_code',
        'county_id',
        'rev_index_wind',
        'ba'
    ]]
    print(a_df)
    a_df.to_pickle(f)


f = '/projects/dwind/agents/arkansas/agents_dwind_fom.pickle'
check_vars(f)
# fix_arkansas()