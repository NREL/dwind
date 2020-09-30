import psycopg2 as pg
import numpy as np
import pandas as pd
import decorators
import utility_functions as utilfunc

# load logger
logger = utilfunc.get_logger()

# configure psycopg2 to treat numeric values as floats (improves
# performance of pulling data from the database)
DEC2FLOAT = pg.extensions.new_type(pg.extensions.DECIMAL.values,
                                   'DEC2FLOAT',
                                   lambda value, curs: float(value) if value is not None else None)
pg.extensions.register_type(DEC2FLOAT)


#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def apply_elec_price_multiplier_and_escalator(dataframe, year, elec_price_change_traj):
    '''
    Obtain a single scalar multiplier for each agent, that is the cost of
    electricity relative to 2016 (when the tariffs were curated).
    Also calculate the compound annual growth rate (CAGR) for the price of
    electricity from present year to 2050, which will be the escalator that
    agents use to project electricity changes in their bill calculations.
    
    elec_price_multiplier = change in present-year elec cost to 2016
    elec_price_escalator = agent's assumption about future price changes
    Note that many customers will not differentiate between real and nominal,
    and therefore many would overestimate the real escalation of electriicty
    prices.
    '''
    
    dataframe = dataframe.reset_index()

    # get current year multiplier values
    elec_price_multiplier_df = elec_price_change_traj[elec_price_change_traj['year']==year].reset_index(drop=True)

    # copy to the multiplier_df for escalator calcs
    year_cap = min(year, 2040)
    elec_price_escalator_df = elec_price_change_traj[elec_price_change_traj['year']==year_cap].reset_index(drop=True)

    # get final year multiplier values and attach to escalator_df
    final_year = np.max(elec_price_change_traj['year'])
    final_year_df = elec_price_change_traj[elec_price_change_traj['year']==final_year].reset_index(drop=True)
    elec_price_escalator_df['final_year_values'] = final_year_df['elec_price_multiplier'].reset_index(drop=True)
    
    # calculate CAGR for time period between final year and current year
    elec_price_escalator_df['elec_price_escalator'] = (elec_price_escalator_df['final_year_values'] / elec_price_escalator_df['elec_price_multiplier'])**(1.0/(final_year-year_cap)) - 1.0

    # et upper and lower bounds of escalator at 1.0 and -1.0, based on historic elec price growth rates
    elec_price_escalator_df['elec_price_escalator'] = np.clip(elec_price_escalator_df['elec_price_escalator'], -.01, .01)

    # merge multiplier and escalator values back to agent dataframe
    dataframe = pd.merge(dataframe, elec_price_multiplier_df[['elec_price_multiplier', 'sector_abbr', 'county_id']], how='left', on=['sector_abbr', 'county_id'])
    dataframe = pd.merge(dataframe, elec_price_escalator_df[['sector_abbr', 'county_id', 'elec_price_escalator']],
                         how='left', on=['sector_abbr', 'county_id'])

    dataframe = dataframe.set_index('agent_id')

    return dataframe


#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def apply_export_tariff_params(dataframe, net_metering_state_df, net_metering_utility_df):

    dataframe = dataframe.reset_index()
    
    # specify relevant NEM columns
    nem_columns = ['compensation_style','nem_system_kw_limit']
    
    # check if utility-specific NEM parameters apply to any agents - need to join on state too (e.g. Pacificorp UT vs Pacificorp ID)
    temp_df = pd.merge(dataframe, net_metering_utility_df[
                        ['eia_id','sector_abbr','state_abbr']+nem_columns], how='left', on=['eia_id','sector_abbr','state_abbr'])
    
    # filter agents with non-null nem_system_kw_limit - these are agents WITH utility NEM
    agents_with_utility_nem = temp_df[pd.notnull(temp_df['nem_system_kw_limit'])]
    
    # filter agents with null nem_system_kw_limit - these are agents WITHOUT utility NEM
    agents_without_utility_nem = temp_df[pd.isnull(temp_df['nem_system_kw_limit'])].drop(nem_columns, axis=1)
    # merge agents with state-specific NEM parameters
    agents_without_utility_nem = pd.merge(agents_without_utility_nem, net_metering_state_df[
                        ['state_abbr', 'sector_abbr']+nem_columns], how='left', on=['state_abbr', 'sector_abbr'])
    
    # re-combine agents list and fill nan's
    dataframe = pd.concat([agents_with_utility_nem, agents_without_utility_nem], sort=False)
    dataframe['compensation_style'].fillna('none', inplace=True)
    dataframe['nem_system_kw_limit'].fillna(0, inplace=True)
    
    dataframe = dataframe.set_index('agent_id')

    return dataframe


#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def apply_pv_tech_performance(dataframe, pv_tech_traj):

    dataframe = dataframe.reset_index()

    dataframe = pd.merge(dataframe, pv_tech_traj, how='left', on=['sector_abbr', 'year'])
                         
    dataframe = dataframe.set_index('agent_id')

    return dataframe
    

#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def apply_depreciation_schedule(dataframe, deprec_sch):

    dataframe = dataframe.reset_index()

    dataframe = pd.merge(dataframe, deprec_sch[['sector_abbr', 'deprec_sch', 'year']],
                         how='left', on=['sector_abbr', 'year'])
                         
    dataframe = dataframe.set_index('agent_id')


    return dataframe

    
#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def apply_pv_prices(dataframe, pv_price_traj):

    dataframe = dataframe.reset_index()

    # join the data
    dataframe = pd.merge(dataframe, pv_price_traj, how='left', on=['sector_abbr', 'year'])

    dataframe = dataframe.set_index('agent_id')

    return dataframe


#%%
@decorators.fn_timer(logger = logger, tab_level = 2, prefix = '')
def apply_batt_prices(dataframe, batt_price_traj, batt_tech_traj, year):

    dataframe = dataframe.reset_index()

    # Merge on prices
    dataframe = pd.merge(dataframe, batt_price_traj[['year','sector_abbr',
                                                     'batt_capex_per_kwh','batt_capex_per_kw',
                                                     'batt_om_per_kwh','batt_om_per_kw']], 
                         how = 'left', on = ['sector_abbr', 'year'])
    
    dataframe = dataframe.set_index('agent_id')

    return dataframe

    
#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def apply_batt_tech_performance(dataframe, batt_tech_traj):

    dataframe = dataframe.reset_index()

    dataframe = dataframe.merge(batt_tech_traj, how='left', on=['year', 'sector_abbr'])
    
    dataframe = dataframe.set_index('agent_id')
    
    return dataframe


#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def apply_financial_params(dataframe, financing_terms, itc_options, inflation_rate, techs):

    in_cols = list(dataframe.columns)
    in_cols.remove('agent_id')
    
    dataframe = dataframe.reset_index()

    dataframe = dataframe.merge(financing_terms, how='left', on=['year', 'sector_abbr'])
    dataframe = dataframe.merge(itc_options, how='left', on=['year', 'tech', 'sector_abbr'])
    
    if 'wind' in techs:    
        dataframe = dataframe[(dataframe['system_size_kw'] > dataframe['min_size_kw']) & (dataframe['system_size_kw'] <= dataframe['max_size_kw'])]

    dataframe['inflation_rate'] = inflation_rate
    
    return_cols = list(financing_terms.columns) + ['itc_fraction_of_capex', 'inflation_rate']
    out_cols = list(pd.unique(in_cols + return_cols))
    
    dataframe = dataframe.set_index('agent_id')
    
    return dataframe[out_cols]


#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def apply_load_growth(dataframe, load_growth_df):

    dataframe = dataframe.reset_index()
    
    dataframe["county_id"] = dataframe.county_id.astype(int)

    dataframe = pd.merge(dataframe, load_growth_df, how='left', on=['year', 'sector_abbr', 'county_id'])
    
    # for res, load growth translates to kwh_per_customer change
    dataframe['load_kwh_per_customer_in_bin'] = np.where(dataframe['sector_abbr']=='res',
                                                dataframe['load_kwh_per_customer_in_bin_initial'] * dataframe['load_multiplier'],
                                                dataframe['load_kwh_per_customer_in_bin_initial'])
                                                
    # for C&I, load growth translates to customer count change
    dataframe['customers_in_bin'] = np.where(dataframe['sector_abbr']!='res',
                                                dataframe['customers_in_bin_initial'] * dataframe['load_multiplier'],
                                                dataframe['customers_in_bin_initial'])
                                                
    # for all sectors, total kwh_in_bin changes
    dataframe['load_kwh_in_bin'] = dataframe['load_kwh_in_bin_initial'] * dataframe['load_multiplier']
    
    dataframe = dataframe.set_index('agent_id')

    return dataframe


#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def calculate_developable_customers_and_load(dataframe):

    dataframe = dataframe.reset_index()

    dataframe['developable_agent_weight'] = dataframe['pct_of_bldgs_developable'] * dataframe['customers_in_bin']
    dataframe['developable_load_kwh_in_bin'] = dataframe['pct_of_bldgs_developable'] * dataframe['load_kwh_in_bin']

    dataframe = dataframe.set_index('agent_id')

    return dataframe


#%%
def get_electric_rates_json(con, unique_rate_ids):

    inputs = locals().copy()

    # reformat the rate list for use in postgres query
    inputs['rate_id_list'] = utilfunc.pylist_2_pglist(unique_rate_ids)
    inputs['rate_id_list'] = inputs['rate_id_list'].replace("L", "")

    # get (only the required) rate jsons from postgres
    sql = """SELECT a.rate_id_alias, a.rate_name, a.eia_id, a.json as rate_json
             FROM diffusion_shared.urdb3_rate_jsons_20200721 a
             WHERE a.rate_id_alias in ({rate_id_list});""".format(**inputs)
    df = pd.read_sql(sql, con, coerce_float=False)

    return df


#%%
def filter_nem_year(df, year):

    # Filter by Sector Specific Sunset Years
    df = df.loc[(df['first_year'] <= year) & (df['sunset_year'] >= year)]

    return df


#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def get_nem_settings(state_limits, state_by_sector, utility_by_sector, selected_scenario, year, state_capacity_by_year, cf_during_peak_demand):

    # Find States That Have Not Sunset
    valid_states = filter_nem_year(state_limits, year)

    # Filter States to Those That Have Not Exceeded Cumulative Capacity Constraints
    valid_states['filter_year'] = pd.to_numeric(valid_states['max_reference_year'], errors='coerce')
    valid_states['filter_year'][valid_states['max_reference_year'] == 'previous'] = year - 2
    valid_states['filter_year'][valid_states['max_reference_year'] == 'current'] = year
    valid_states['filter_year'][pd.isnull(valid_states['filter_year'])] = year

    state_df = pd.merge(state_capacity_by_year, valid_states , how='left', on=['state_abbr'])
    state_df = state_df[state_df['year'] == state_df['filter_year'] ]
    state_df = state_df.merge(cf_during_peak_demand, on = 'state_abbr')

    state_df = state_df.loc[ pd.isnull(state_df['max_cum_capacity_mw']) | ( pd.notnull( state_df['max_cum_capacity_mw']) & (state_df['cum_capacity_mw'] < state_df['max_cum_capacity_mw']))]
    # Calculate the maximum MW of solar capacity before reaching the NEM cap. MW are determine on a generation basis during the period of peak demand, as determined by ReEDS.
    # CF during peak period is based on ReEDS H17 timeslice, assuming average over south-facing 15 degree tilt systems (so this could be improved by using the actual tilts selected)
    state_df['max_mw'] = (state_df['max_pct_cum_capacity']/100) * state_df['peak_demand_mw'] / state_df['solar_cf_during_peak_demand_period']
    state_df = state_df.loc[ pd.isnull(state_df['max_pct_cum_capacity']) | ( pd.notnull( state_df['max_pct_cum_capacity']) & (state_df['max_mw'] > state_df['cum_capacity_mw']))]

    # Filter state and sector data to those that have not sunset
    selected_state_by_sector = state_by_sector.loc[state_by_sector['scenario'] == selected_scenario]
    valid_state_sector = filter_nem_year(selected_state_by_sector, year)

    # Filter state and sector data to those that match states which have not sunset/reached peak capacity
    valid_state_sector = valid_state_sector[valid_state_sector['state_abbr'].isin(state_df['state_abbr'].values)]
    
    # Filter utility and sector data to those that have not sunset
    selected_utility_by_sector = utility_by_sector.loc[utility_by_sector['scenario'] == selected_scenario]
    valid_utility_sector = filter_nem_year(selected_utility_by_sector, year)
    
    # Filter out utility/sector combinations in states where capacity constraints have been reached
    # Assumes that utilities adhere to broader state capacity constraints, and not their own
    valid_utility_sector = valid_utility_sector[valid_utility_sector['state_abbr'].isin(state_df['state_abbr'].values)]

    # Return State/Sector data (or null) for all combinations of states and sectors
    full_state_list = state_by_sector.loc[ state_by_sector['scenario'] == 'BAU' ].loc[:, ['state_abbr', 'sector_abbr']]
    state_result = pd.merge( full_state_list.drop_duplicates(), valid_state_sector, how='left', on=['state_abbr','sector_abbr'] )
    state_result['nem_system_kw_limit'].fillna(0, inplace=True)
    
    # Return Utility/Sector data (or null) for all combinations of utilities and sectors
    full_utility_list = utility_by_sector.loc[ utility_by_sector['scenario'] == 'BAU' ].loc[:, ['eia_id','sector_abbr','state_abbr']]
    utility_result = pd.merge( full_utility_list.drop_duplicates(), valid_utility_sector, how='left', on=['eia_id','sector_abbr','state_abbr'] )
    utility_result['nem_system_kw_limit'].fillna(0, inplace=True)

    return state_result, utility_result


#%%
def get_and_apply_agent_load_profiles(con, agent):

    inputs = locals().copy()

    inputs['bldg_id'] = agent.loc['bldg_id']
    inputs['sector_abbr'] = agent.loc['sector_abbr']
    inputs['state_abbr'] = agent.loc['state_abbr']
    
    sql = """SELECT bldg_id, sector_abbr, state_abbr,
                    kwh_load_profile as consumption_hourly
             FROM diffusion_load_profiles.{sector_abbr}stock_load_profiles b
                 WHERE bldg_id = {bldg_id} 
                 AND sector_abbr = '{sector_abbr}'
                 AND state_abbr = '{state_abbr}';""".format(**inputs)
                           
    df = pd.read_sql(sql, con, coerce_float=False)

    df = df[['consumption_hourly']]
    df['load_kwh_per_customer_in_bin'] = agent.loc['load_kwh_per_customer_in_bin']
    
    # scale the normalized profile to sum to the total load
    df = df.apply(scale_array_sum, axis=1, args=(
        'consumption_hourly', 'load_kwh_per_customer_in_bin'))

    return df


#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def get_annual_resource_wind(con, schema, year, sectors):

    inputs = locals().copy()

    df_list = []
    for sector_abbr, sector in sectors.items():
        inputs['sector_abbr'] = sector_abbr
        sql = """SELECT '{sector_abbr}'::VARCHAR(3) as sector_abbr,
                        a.county_id, a.bin_id,
                    	COALESCE(b.turbine_height_m, 0) as turbine_height_m,
                    	COALESCE(b.turbine_size_kw, 0) as turbine_size_kw,
                    	coalesce(c.interp_factor, 0) as power_curve_interp_factor,
                    	COALESCE(c.power_curve_1, -1) as power_curve_1,
                    	COALESCE(c.power_curve_2, -1) as power_curve_2,
                    	COALESCE(d.aep, 0) as naep_1,
                    	COALESCE(e.aep, 0) as naep_2
                FROM  {schema}.agent_core_attributes_{sector_abbr} a
                LEFT JOIN {schema}.agent_allowable_turbines_lkup_{sector_abbr} b
                    	ON a.county_id = b.county_id
                    	and a.bin_id = b.bin_id
                LEFT JOIN {schema}.wind_performance_power_curve_transitions c
                    	ON b.turbine_size_kw = c.turbine_size_kw
                         AND c.year = {year}
                LEFT JOIN diffusion_resource_wind.wind_resource_annual d
                    	ON a.i = d.i
                    	AND a.j = d.j
                    	AND a.cf_bin = d.cf_bin
                    	AND b.turbine_height_m = d.height
                    	AND c.power_curve_1 = d.turbine_id
                LEFT JOIN diffusion_resource_wind.wind_resource_annual e
                    	ON a.i = e.i
                    	AND a.j = e.j
                    	AND a.cf_bin = e.cf_bin
                    	AND b.turbine_height_m = e.height
                    	AND c.power_curve_2 = e.turbine_id;""".format(**inputs)
        df_sector = pd.read_sql(sql, con, coerce_float=False)
        df_list.append(df_sector)

    df = pd.concat(df_list, axis=0, ignore_index=True, sort=False)

    return df


#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def apply_technology_performance_wind(wind_resource_df, wind_derate_traj, year):
    
    in_cols = list(wind_resource_df.columns)

    wind_resource_df = pd.merge(wind_resource_df, wind_derate_traj[wind_derate_traj['year'] == year], how='left', on=['turbine_size_kw'])
    wind_resource_df['naep'] = (wind_resource_df['power_curve_interp_factor'] * (wind_resource_df['naep_2'] -
                                                                                 wind_resource_df['naep_1']) + wind_resource_df['naep_1']) * wind_resource_df['wind_derate_factor']
    
    return_cols = ['wind_derate_factor', 'naep']
    out_cols = list(pd.unique(in_cols + return_cols))

    return wind_resource_df[out_cols]


#%%
def get_and_apply_normalized_hourly_resource_solar(con, agent):

    inputs = locals().copy()
    
    inputs['solar_re_9809_gid'] = agent.loc['solar_re_9809_gid']
    inputs['tilt'] = agent.loc['tilt']
    inputs['azimuth'] = agent.loc['azimuth']
    
    sql = """SELECT solar_re_9809_gid, tilt, azimuth,
                    cf as generation_hourly,
                    1e6 as scale_offset
            FROM diffusion_resource_solar.solar_resource_hourly
                WHERE solar_re_9809_gid = '{solar_re_9809_gid}'
                AND tilt = '{tilt}'
                AND azimuth = '{azimuth}';""".format(**inputs)
                
    df = pd.read_sql(sql, con, coerce_float=False)

    df = df[['generation_hourly', 'scale_offset']]
    
    # rename the column generation_hourly to solar_cf_profile
    df.rename(columns={'generation_hourly':'solar_cf_profile'}, inplace=True)
          
    return df


#%%
def get_and_apply_normalized_hourly_resource_wind(con, agent):

    inputs = locals().copy()
    
    inputs['i'] = agent.loc['i']
    inputs['j'] = agent.loc['j']
    inputs['cf_bin'] = agent.loc['cf_bin']
    inputs['turbine_height_m'] = agent.loc['turbine_height_m']
    inputs['power_curve_1'] = agent.loc['power_curve_1']
    inputs['power_curve_2'] = agent.loc['power_curve_2']
    
    sql = """WITH selected_turbines AS(
                 SELECT {i} as i, {j} as j, {cf_bin} as cf_bin,
                 {turbine_height_m} as turbine_height_m,
                 {power_curve_1} as power_curve_1, {power_curve_2} as power_curve_2
             )
             SELECT COALESCE(b.cf, array_fill(1, array[8760])) as generation_hourly_1,
                    COALESCE(c.cf, array_fill(1, array[8760])) as generation_hourly_2
             FROM selected_turbines a 
             LEFT JOIN diffusion_resource_wind.wind_resource_hourly b
                 ON a.i = b.i
                     AND a.j = b.j
                     AND a.cf_bin = b.cf_bin
                     AND a.turbine_height_m = b.height
                     AND a.power_curve_1 = b.turbine_id
             LEFT JOIN diffusion_resource_wind.wind_resource_hourly c
                 ON a.i = c.i
                     AND a.j = c.j
                     AND a.cf_bin = c.cf_bin
                     AND a.turbine_height_m = c.height
                     AND a.power_curve_2 = c.turbine_id;""".format(**inputs)
    df = pd.read_sql(sql, con, coerce_float=False).reset_index(drop=True)

    # get necessary values from agent attributes    
    power_curve_interp_factor = agent.loc['power_curve_interp_factor']
    annual_energy_production_kwh = agent.loc['annual_energy_production_kwh']
    
    # apply the scale offset to convert values to float with correct precision
    scale_offset = 1e3
    df.at[0,'generation_hourly_1'] = np.array(df.loc[0,'generation_hourly_1'], dtype='float64') / scale_offset
    df.at[0,'generation_hourly_2'] = np.array(df.loc[0,'generation_hourly_2'], dtype='float64') / scale_offset
    
    # interpolate power curves
    df.loc[0,'generation_hourly'] = np.nan
    df['generation_hourly'] = df['generation_hourly'].astype(object)
    df.at[0,'generation_hourly'] = (power_curve_interp_factor *
                                     (np.array(df.loc[0,'generation_hourly_2'], dtype='float64') - np.array(df.loc[0,'generation_hourly_1'], dtype='float64')) +
                                     np.array(df.loc[0,'generation_hourly_1'], dtype='float64'))
    
    # scale the normalized profile by the annual generation
    df.at[0,'generation_hourly'] = (np.array(df.loc[0,'generation_hourly'], dtype='float64') /
                                     np.array(df.loc[0,'generation_hourly'], dtype='float64').sum() *
                                     np.float64(annual_energy_production_kwh))
    
    # subset to only the desired output columns
    out_cols = ['generation_hourly']
          
    return df[out_cols]


#%%
def scale_array_precision(row, array_col, prec_offset_col):

    row[array_col] = np.array(
        row[array_col], dtype='float64') / row[prec_offset_col]

    return row


#%%
def scale_array_sum(row, array_col, scale_col):

    hourly_array = np.array(row[array_col], dtype='float64')
    row[array_col] = hourly_array / \
        hourly_array.sum() * np.float64(row[scale_col])

    return row


#%%
def interpolate_array(row, array_1_col, array_2_col, interp_factor_col, out_col):

    if row[interp_factor_col] != 0:
        interpolated = row[interp_factor_col] * \
            (row[array_2_col] - row[array_1_col]) + row[array_1_col]
    else:
        interpolated = row[array_1_col]
    row[out_col] = interpolated

    return row


#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def apply_carbon_intensities(dataframe, carbon_intensities):

    dataframe = dataframe.reset_index()

    dataframe = pd.merge(dataframe, carbon_intensities, how='left', on=['state_abbr', 'year'])

    dataframe = dataframe.set_index('agent_id')

    return dataframe
    

#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def apply_wholesale_elec_prices(dataframe, wholesale_elec_prices):

    dataframe = dataframe.reset_index()

    dataframe = pd.merge(dataframe, wholesale_elec_prices, how='left', on=['county_id', 'year'])

    dataframe = dataframe.set_index('agent_id')

    return dataframe


#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def get_state_starting_capacities(con, schema):

    inputs = locals().copy()

    sql = """SELECT *
             FROM {schema}.state_starting_capacities_to_model;""".format(**inputs)
    df = pd.read_sql(sql, con)

    return df


#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def apply_state_incentives(dataframe, state_incentives, year, start_year, state_capacity_by_year, end_date = pd.to_datetime('2029/01/01')):

    dataframe = dataframe.reset_index()
    
    # Convert date columns to datetime format
    state_incentives['start_date'] = pd.to_datetime(state_incentives['start_date'])
    state_incentives['end_date'] = pd.to_datetime(state_incentives['end_date'])
    
    # Remove incentives with end year equal to start_year
    state_incentives = state_incentives.loc[state_incentives['end_date'].dt.year != start_year]

    # Fill in missing end_dates
    if bool(end_date):
        state_incentives['end_date'][pd.isnull(state_incentives['end_date'])] = end_date

    #Adjust incenctives to account for reduced values as adoption increases
    yearly_escalation_function = lambda value, end_year: max(value - value * (1.0 / (end_year - start_year)) * (year-start_year), 0)
    for field in ['pbi_usd_p_kwh','cbi_usd_p_w','ibi_pct','cbi_usd_p_wh']:
        state_incentives[field] = state_incentives.apply(lambda row: yearly_escalation_function(row[field], row['end_date'].year), axis=1)
        
    # Filter Incentives by the Years in which they are valid
    state_incentives = state_incentives.loc[
        pd.isnull(state_incentives['start_date']) | (state_incentives['start_date'].dt.year <= year)]
    state_incentives = state_incentives.loc[
        pd.isnull(state_incentives['end_date']) | (state_incentives['end_date'].dt.year >= year)]

    # Combine valid incentives with the cumulative metrics for each state up until the current year
    state_incentives_mg = state_incentives.merge(state_capacity_by_year.loc[state_capacity_by_year['year'] == year],
                                                 how='left', on=["state_abbr"])

    # Filter where the states have not exceeded their cumulative installed capacity (by mw or pct generation) or total program budget
    #state_incentives_mg = state_incentives_mg.loc[pd.isnull(state_incentives_mg['incentive_cap_total_pct']) | (state_incentives_mg['cum_capacity_pct'] < state_incentives_mg['incentive_cap_total_pct'])]
    state_incentives_mg = state_incentives_mg.loc[pd.isnull(state_incentives_mg['incentive_cap_total_mw']) | (state_incentives_mg['cum_capacity_mw'] < state_incentives_mg['incentive_cap_total_mw'])]
    state_incentives_mg = state_incentives_mg.loc[pd.isnull(state_incentives_mg['budget_total_usd']) | (state_incentives_mg['cum_incentive_spending_usd'] < state_incentives_mg['budget_total_usd'])]

    output  =[]
    for i in state_incentives_mg.groupby(['state_abbr', 'sector_abbr']):
        row = i[1]
        state, sector = i[0]
        output.append({'state_abbr':state, 'sector_abbr':sector,"state_incentives":row})

    state_inc_df = pd.DataFrame(columns=['state_abbr', 'sector_abbr', 'state_incentives'])
    state_inc_df = pd.concat([state_inc_df, pd.DataFrame.from_records(output)], sort=False)
    
    dataframe = pd.merge(dataframe, state_inc_df, on=['state_abbr','sector_abbr'], how='left')
    
    dataframe = dataframe.set_index('agent_id')

    return dataframe


#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def estimate_initial_market_shares(dataframe, state_starting_capacities_df):

    # record input columns
    in_cols = list(dataframe.columns)

    # find the total number of customers in each state (by technology and
    # sector)
    state_total_developable_customers = dataframe[['state_abbr', 'sector_abbr', 'tech', 'developable_agent_weight']].groupby(
        ['state_abbr', 'sector_abbr', 'tech']).sum().reset_index()
    state_total_agents = dataframe[['state_abbr', 'sector_abbr', 'tech', 'developable_agent_weight']].groupby(
        ['state_abbr', 'sector_abbr', 'tech']).count().reset_index()
    # rename the final columns
    state_total_developable_customers.columns = state_total_developable_customers.columns.str.replace(
        'developable_agent_weight', 'developable_customers_in_state')
    state_total_agents.columns = state_total_agents.columns.str.replace(
        'developable_agent_weight', 'agent_count')
    # merge together
    state_denominators = pd.merge(state_total_developable_customers, state_total_agents, how='left', on=[
                                  'state_abbr', 'sector_abbr', 'tech'])

    # merge back to the main dataframe
    dataframe = pd.merge(dataframe, state_denominators, how='left', on=[
                         'state_abbr', 'sector_abbr', 'tech'])

    # merge in the state starting capacities
    dataframe = pd.merge(dataframe, state_starting_capacities_df, how='left',
                         on=['tech', 'state_abbr', 'sector_abbr'])

    # determine the portion of initial load and systems that should be allocated to each agent
    # (when there are no developable agnets in the state, simply apportion evenly to all agents)
    dataframe['portion_of_state'] = np.where(dataframe['developable_customers_in_state'] > 0,
                                             dataframe[
                                                 'developable_agent_weight'] / dataframe['developable_customers_in_state'],
                                             1. / dataframe['agent_count'])
    # apply the agent's portion to the total to calculate starting capacity and systems
    dataframe['adopters_cum_last_year'] = dataframe['portion_of_state'] * dataframe['systems_count']
    dataframe['system_kw_cum_last_year'] = dataframe['portion_of_state'] * dataframe['capacity_mw'] * 1000.0
    dataframe['batt_kw_cum_last_year'] = 0.0
    dataframe['batt_kwh_cum_last_year'] = 0.0

    dataframe['market_share_last_year'] = np.where(dataframe['developable_agent_weight'] == 0, 0,
                                                   dataframe['adopters_cum_last_year'] / dataframe['developable_agent_weight'])

    dataframe['market_value_last_year'] = dataframe['system_capex_per_kw'] * dataframe['system_kw_cum_last_year']

    # reproduce these columns as "initial" columns too
    dataframe['initial_number_of_adopters'] = dataframe['adopters_cum_last_year']
    dataframe['initial_pv_kw'] = dataframe['system_kw_cum_last_year']
    dataframe['initial_market_share'] = dataframe['market_share_last_year']
    dataframe['initial_market_value'] = 0

    # isolate the return columns
    return_cols = ['initial_number_of_adopters', 'initial_pv_kw', 'initial_market_share', 'initial_market_value',
                   'adopters_cum_last_year', 'system_kw_cum_last_year', 'batt_kw_cum_last_year', 'batt_kwh_cum_last_year', 'market_share_last_year', 'market_value_last_year']

    dataframe[return_cols] = dataframe[return_cols].fillna(0)

    out_cols = in_cols + return_cols

    return dataframe[out_cols]


#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def apply_market_last_year(dataframe, market_last_year_df):
    
    dataframe = dataframe.merge(market_last_year_df, on=['agent_id'], how='left')
    
    return dataframe


#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def estimate_total_generation(dataframe):

    dataframe['total_gen_twh'] = ((dataframe['number_of_adopters'] - dataframe['initial_number_of_adopters'])
                                  * dataframe['annual_energy_production_kwh'] * 1e-9) + (0.23 * 8760 * dataframe['initial_pv_kw'] * 1e-6)

    return dataframe


#%%   
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def calc_state_capacity_by_year(con, schema, load_growth, peak_demand_mw, is_first_year, year, agents, last_year_installed_capacity):

    if is_first_year:
        df = last_year_installed_capacity.loc[last_year_installed_capacity['tech'] == 'solar'].groupby('state_abbr')['capacity_mw'].sum().reset_index()
        
        # Not all states have starting capacity, don't want to drop any states thus left join on peak_demand
        df = peak_demand_mw.merge(df,how = 'left').fillna(0)
        df['peak_demand_mw'] = df['peak_demand_mw_2014']
        df['cum_capacity_mw'] = df['capacity_mw']

    else:
        df = last_year_installed_capacity.copy()
        df['cum_capacity_mw'] = df['system_kw_cum']/1000
        
        load_growth_this_year = load_growth.loc[(load_growth['year'] == year) & (load_growth['sector_abbr'] == 'res')]
        load_growth_this_year = pd.merge(agents.df[['state_abbr', 'county_id']], load_growth_this_year, how='left', on=['county_id'])
        load_growth_this_year = load_growth_this_year.groupby('state_abbr')['load_multiplier'].mean().reset_index()
        df = df.merge(load_growth_this_year, on = 'state_abbr')
        
        df = peak_demand_mw.merge(df,how = 'left', on = 'state_abbr').fillna(0)
        df['peak_demand_mw'] = df['peak_demand_mw_2014'] * df['load_multiplier']

    df['cum_capacity_pct'] = 0
    df['cum_incentive_spending_usd'] = 0
    df['year'] = year
    
    df = df[['state_abbr','cum_capacity_mw','cum_capacity_pct','cum_incentive_spending_usd','peak_demand_mw','year']]
    
    return df


#%%
def get_rate_switch_table(con):
    
    # get rate switch table from database
    sql = 'SELECT * FROM diffusion_shared.rate_switch_lkup_2019;'
    rate_switch_table = pd.read_sql(sql, con, coerce_float=False)
    rate_switch_table = rate_switch_table.reset_index(drop=True)
    
    return rate_switch_table


def apply_rate_switch(rate_switch_table, agent, system_size_kw):
    
    rate_switch_table.rename(columns={'rate_id_alias':'tariff_id', 'json':'tariff_dict'}, inplace=True)
    rate_switch_table = rate_switch_table[(rate_switch_table['eia_id'] == agent.loc['eia_id']) &
                                          (rate_switch_table['res_com'] == str(agent.loc['sector_abbr']).upper()[0]) &
                                          (rate_switch_table['min_pv_kw_limit'] <= system_size_kw) &
                                          (rate_switch_table['max_pv_kw_limit'] > system_size_kw)]
    rate_switch_table = rate_switch_table.reset_index(drop=True)
    
    # check if a DG rate is applicable to agent
    if (system_size_kw > 0) & (len(rate_switch_table) == 1):
        # if valid DG rate available to agent, force NEM on
        agent['nem_system_kw_limit'] = 1e6
        # update agent attributes to DG rate
        agent['tariff_id'] = rate_switch_table['tariff_id'][0]
        agent['tariff_dict'] = rate_switch_table['tariff_dict'][0]
        # return any one time charges (e.g., interconnection fees)
        one_time_charge = rate_switch_table['one_time_charge'][0]
    else:
        # don't update agent attributes, return one time charge of $0
        one_time_charge = 0.
    
    return agent, one_time_charge


#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def reassign_agent_tariffs(dataframe, con):

    # define rates to use in replacement of incorrect tariffs
    
    # map res/com tariffs based on most likely tariff in state
    res_tariffs = {
                    'AL':17279, # Family Dwelling Service
                    'AR':16671, # Optional Residential Time-Of-Use (RT) Single Phase
                    'AZ':15704, # Residential Time of Use (Saver Choice) TOU-E
                    'CA':15747, # E-1 -Residential Service Baseline Region P
                    'CO':17078, # Residential Service (Schedule R)
                    'CT':16678, # Rate 1 - Residential Electric Service
                    'DC':16806, # Residential - Schedule R
                    'DE':11569, # Residential Service
                    'FL':16986, # RS-1 Residential Service
                    'GA':16649, # SCHEDULE R-22 RESIDENTIAL SERVICE
                    'IA':11693, # Optional Residential Service
                    'ID':16227, # Schedule 1: Residential Rates
                    'IL':16045, # DS-1 Residential Zone 1
                    'IN':15491, # RS - Residential Service
                    'KS':8178, # M System Residential Service
                    'KY':16566, # Residential Service
                    'LA':16352, # Residential and Farm Service - Single Phase (RS-L)
                    'MA':15953, # Greater Boston Residential R-1 (A1)
                    'MD':14779, # Residential Service (R)
                    'ME':15984, # A Residential Standard Offer Service (Bundled)
                    'MI':16265, # Residential Service - Secondary (Rate RS)
                    'MN':15556, # Residential Service - Overhead Standard (A01)
                    'MO':17207, # 1(M) Residential Service Rate
                    'MS':16788, # Residential Service Single Phase (RS-38C)
                    'MT':5216, # Single Phase
                    'NC':16938, # Residential Service (RES-41) Single Phase
                    'ND':14016, # Residential Service Rate 10
                    'NE':13817, # Residential Service
                    'NH':16605, # Residential Service
                    'NJ':16229, # RS - Residential Service
                    'NM':8692, # 1A (Residential Service)
                    'NV':16701, # D-1 (Residential Service)
                    'NY':16902, # SC1- Zone A
                    'OH':16892, # RS (Residential Service)
                    'OK':15258, # Residential Service (R-1)
                    'OR':15847, # Schedule 4 - Residential (Single Phase)
                    'PA':17237, # RS (Residential Service)
                    'RI':16598, # A-16 (Residential Service)
                    'SC':15744, # Residential - RS (SC)
                    'SD':1216, # Town and Rural Residential Rate
                    'TN':15149, # Residential Electric Service
                    'TX':16710, # Residential Service - Time Of Day
                    'UT':15847, # Schedule 4 - Residential (Single Phase)
                    'VA':17067, # Residential Schedule 1
                    'VT':16544, # Rate 01 Residential Service
                    'WA':16305, # 10 (Residential and Farm Primary General Service)
                    'WI':15543, # Residential Rg-1
                    'WV':15515, # Residential Service A
                    'WY':15847 # Schedule 4 - Residential (Single Phase)
                    }
    
    com_tariffs = {
                    'AL':15494, # BTA - BUSINESS TIME ADVANTAGE (OPTIONAL) - Primary
                    'AR':16674, # Small General Service (SGS)
                    'AZ':10742, # LGS-TOU- N - Large General Service Time-of-Use
                    'CA':17057, # A-10 Medium General Demand Service (Secondary Voltage)
                    'CO':17102, # Commercial Service (Schedule C)
                    'CT':16684, # Rate 35 Intermediate General Electric Service
                    'DC':15336, # General Service (Schedule GS)
                    'DE':1199, # Schedule LC-P Large Commercial Primary
                    'FL':13790, # SDTR-1 (Option A)
                    'GA':1905, # SCHEDULE TOU-MB-4 TIME OF USE - MULTIPLE BUSINESS
                    'IA':11705, # Three Phase Farm
                    'ID':14782, # Large General Service (3 Phase)-Schedule 21
                    'IL':1567, # General Service Three Phase standard
                    'IN':15492, # CS - Commercial Service
                    'KS':14736, # Generation Substitution Service
                    'KY':17179, # General Service (Single Phase)
                    'LA':17220, # Large General Service (LGS-L)
                    'MA':16005, # Western Massachusetts Primary General Service G-2
                    'MD':2659, # Commercial
                    'ME':16125, # General Service Rate
                    'MI':5355, # Large Power Service (LP4)
                    'MN':15566, # General Service (A14) Secondary Voltage
                    'MO':17208, # 2(M) Small General Service - Single phase
                    'MS':13427, # General Service - Low Voltage Single-Phase (GS-LVS-14)
                    'MT':10707, # Three Phase
                    'NC':16947, # General Service (GS-41)
                    'ND':14035, # Small General Electric Service rate 20 (Demand Metered; Non-Demand)
                    'NE':13818, # General Service Single-Phase
                    'NH':16620, # GV Commercial and Industrial Service
                    'NJ':17095, # AGS Secondary- BGS-RSCP
                    'NM':15769, # 2A (Small Power Service)
                    'NV':13724, # OGS-2-TOU
                    'NY':15940, # SC-9 - General Large High Tension Service [Westchester]
                    'OH':16873, # GS (General Service-Secondary)
                    'OK':17144, # GS-TOU (General Service Time-Of-Use)
                    'OR':15829, # Small Non-Residential Direct Access Service, Single Phase (Rate 532)
                    'PA':7066, # Large Power 2 (LP2)
                    'RI':16600, # G-02 (General C & I Rate)
                    'SC':16207, # 3 (Municipal  Power Service)
                    'SD':3650, # Small Commercial
                    'TN':15154, # Medium General Service (Primary)
                    'TX':6001, # Medium Non-Residential LSP POLR
                    'UT':3478, # SCHEDULE GS - 3 Phase General Service
                    'VA':16557, # Small General Service Schedule 5
                    'VT':16543, # Rate 06: General Service
                    'WA':16306, # 40 (Large Demand General Service over 3MW - Primary)
                    'WI':6620, # Cg-7 General Service Time-of-Day (Primary)
                    'WV':15518, # General Service C
                    'WY':3878 # General Service (GS)-Three phase
                    }
    
    # map industrial tariffs based on census division
    ind_tariffs = {
                    'SA':16657, # Georgia Power Co, Schedule TOU-GSD-10 Time Of Use - General Service Demand
                    'WSC':15919, # Southwestern Public Service Co (Texas), Large General Service - Inside City Limits 115 KV
                    'PAC':15864, # PacifiCorp (Oregon), Schedule 47 - Secondary (Less than 4000 kW)
                    'MA':16525, # New York State Elec & Gas Corp, All Regions - SERVICE CLASSIFICATION NO. 7-1 Large General Service TOU - Secondary -ESCO                   
                    'MTN':17101, # Public Service Co of Colorado, Secondary General Service (Schedule SG)                   
                    'ENC':15526, # Wisconsin Power & Light Co, Industrial Power Cp-1 (Secondary)                   
                    'NE':16635, # Delmarva Power, General Service - Primary                   
                    'ESC':15490, # Alabama Power Co, LPM - LIGHT AND POWER SERVICE - MEDIUM                   
                    'WNC':6642 # Northern States Power Co - Wisconsin, Cg-9.1 Large General Time-of-Day Primary Mandatory Customers
                   }
    
    dataframe = dataframe.reset_index()

    # separate agents with incorrect and correct rates
    bad_rates = dataframe.loc[np.in1d(dataframe['tariff_id'], [4145, 7111, 8498, 10953, 10954, 12003])]
    good_rates = dataframe.loc[~np.in1d(dataframe['tariff_id'], [4145, 7111, 8498, 10953, 10954, 12003])]
    
    # if incorrect rates exist, grab the correct ones from the rates table
    if len(bad_rates) > 0:
        
        # set new tariff_id based on location
        bad_rates['tariff_id'] = np.where(bad_rates['sector_abbr']=='res',
                                          bad_rates['state_abbr'].map(res_tariffs),
                                          np.where(bad_rates['sector_abbr']=='com',
                                                   bad_rates['state_abbr'].map(com_tariffs),
                                                   bad_rates['census_division_abbr'].map(ind_tariffs)))
        
        # get json objects for new rates and rename columns in preparation for merge
        new_rates_json_df = get_electric_rates_json(con, bad_rates['tariff_id'].tolist())
        new_rates_json_df = (new_rates_json_df
                             .drop(['rate_name','eia_id'], axis='columns')
                             .rename(columns={'rate_id_alias':'tariff_id','rate_json':'tariff_dict'})
                            )
        
        # drop bad tariff_dict from agent dataframe and merge correct one
        bad_rates = bad_rates.drop(['tariff_dict'], axis='columns')
        bad_rates = bad_rates.merge(new_rates_json_df, how='left', on='tariff_id')
    
    # reconstruct full agent dataframe
    dataframe = pd.concat([good_rates, bad_rates], ignore_index=True, sort=False)
    dataframe = dataframe.set_index('agent_id')

    return dataframe


#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def calc_system_size_wind(dataframe, wind_system_sizing, wind_resource_df):
    
    in_cols = list(dataframe.columns)
    in_cols.remove('agent_id')
    
    dataframe = dataframe.reset_index()    
    
    # get and join in system sizing targets df
    dataframe = pd.merge(dataframe, wind_system_sizing, how='left', on=['sector_abbr'])
    
    # determine whether NEM is available in the state and sector
    dataframe['enable_net_metering'] = dataframe['nem_system_kw_limit'] > 0
    
    # set the target kwh according to NEM availability
    dataframe['target_kwh'] = np.where(dataframe['enable_net_metering'] == False,
        dataframe['load_kwh_per_customer_in_bin'] * dataframe['sys_size_target_no_nem'],
        dataframe['load_kwh_per_customer_in_bin'] * dataframe['sys_size_target_nem'])
    
    # also set the oversize limit according to NEM availability
    dataframe['oversize_limit_kwh'] = np.where(dataframe['enable_net_metering'] == False,
        dataframe['load_kwh_per_customer_in_bin'] * dataframe['sys_oversize_limit_no_nem'],
        dataframe['load_kwh_per_customer_in_bin'] * dataframe['sys_oversize_limit_nem'])

    # join in the resource data
    dataframe = pd.merge(dataframe, wind_resource_df, how = 'left', on = ['sector_abbr', 'county_id', 'bin_id'])

    # calculate the system generation from naep and turbine_size_kw    
    dataframe['annual_energy_production_kwh'] = dataframe['turbine_size_kw'] * dataframe['naep']

    # initialize values for scoe and n_units
    dataframe['scoe'] = np.absolute(dataframe['annual_energy_production_kwh'] - dataframe['target_kwh'])
    dataframe['n_units'] = 1.

    # Handle Special Cases
    
    # Buildings requiring more electricity than can be generated by the largest turbine (1.5 MW)
    # Return very low rank score and the optimal continuous number of turbines
    big_projects = (dataframe['turbine_size_kw'] == 1500) & (dataframe['annual_energy_production_kwh'] < dataframe['target_kwh'])
    dataframe.loc[big_projects, 'scoe'] = 0.
    # handle divide by zero error
    # (only occurs where system size is zero, which is a different slice than big_projects)
    dataframe['annual_energy_production_kwh'] = np.where(dataframe['annual_energy_production_kwh'] == 0., -1., dataframe['annual_energy_production_kwh'])
    dataframe.loc[big_projects, 'n_units'] = np.minimum(4, dataframe['target_kwh'] / dataframe['annual_energy_production_kwh'])
    dataframe['annual_energy_production_kwh'] = np.where(dataframe['annual_energy_production_kwh'] < 0., 0., dataframe['annual_energy_production_kwh'])

    # identify oversized projects
    oversized_turbines = dataframe['annual_energy_production_kwh'] > dataframe['oversize_limit_kwh']
    # also identify zero production turbines
    no_kwh = dataframe['annual_energy_production_kwh'] == 0
    # where either condition is true, set a high score and zero turbines
    dataframe.loc[oversized_turbines | no_kwh, 'scoe'] = np.array([1e8]) + dataframe['turbine_size_kw'] * 100 + dataframe['turbine_height_m']
    dataframe.loc[oversized_turbines | no_kwh, 'n_units'] = 0.0
    # also disable net metering
    dataframe.loc[oversized_turbines | no_kwh, 'enable_net_metering'] = False

    # check that the system is within the net metering size limit
    over_nem_limit = dataframe['turbine_size_kw'] > dataframe['nem_system_kw_limit']
    dataframe.loc[over_nem_limit, 'scoe'] = dataframe['scoe'] * 2.
    dataframe.loc[over_nem_limit, 'enable_net_metering'] = False

    # for each agent, find the optimal turbine
    dataframe['scoe'] = dataframe['scoe'].astype(np.float64)
    dataframe['rank'] = dataframe.groupby(['county_id', 'bin_id', 'sector_abbr'])['scoe'].rank(ascending = True, method = 'first')
    dataframe_sized = dataframe[dataframe['rank'] == 1]
    # add in the system_size_kw field
    dataframe_sized.loc[:, 'system_size_kw'] = dataframe_sized['turbine_size_kw'] * dataframe_sized['n_units']
    # recalculate the aep based on the system size (instead of plain turbine size)
    dataframe_sized.loc[:, 'annual_energy_production_kwh'] = dataframe_sized['system_size_kw'] * dataframe_sized['naep']

    # add capacity factor
    dataframe_sized.loc[:, 'capacity_factor'] = dataframe_sized['naep']/8760.

    # add system size class
    dataframe_sized.loc[:, 'system_size_factors'] = np.where(dataframe_sized['system_size_kw'] > 1500, '1500+', dataframe_sized['system_size_kw'].astype('str'))

    # where system size is zero, adjust other dependent columns:
    no_system = dataframe_sized['system_size_kw'] == 0
    dataframe_sized.loc[:, 'power_curve_1'] = np.where(no_system, -1, dataframe_sized['power_curve_1'])
    dataframe_sized.loc[:, 'power_curve_2'] = np.where(no_system, -1, dataframe_sized['power_curve_2'])
    dataframe_sized.loc[:, 'turbine_size_kw'] = np.where(no_system, 0, dataframe_sized['turbine_size_kw'])
    dataframe_sized.loc[:, 'turbine_height_m'] = np.where(no_system, 0, dataframe_sized['turbine_height_m'])
    dataframe_sized.loc[:, 'n_units'] = np.where(no_system, 0, dataframe_sized['n_units'])
    dataframe_sized.loc[:, 'naep'] = np.where(no_system, 0, dataframe_sized['naep'])
    dataframe_sized.loc[:, 'capacity_factor'] = np.where(no_system, 0, dataframe_sized['capacity_factor'])

    dataframe_sized.loc[:, 'turbine_height_m'] = dataframe_sized['turbine_height_m'].astype(np.float64)

    # add dummy column for inverter lifetime 
    dataframe_sized.loc[:, 'inverter_lifetime_yrs'] = np.nan
    dataframe_sized.loc[:, 'inverter_lifetime_yrs'] = dataframe_sized['inverter_lifetime_yrs'].astype(np.float64)

    return_cols = ['enable_net_metering', 'annual_energy_production_kwh', 'naep', 'capacity_factor', 'system_size_kw', 'system_size_factors', 'n_units', 'inverter_lifetime_yrs',
                   'turbine_height_m', 'turbine_size_kw', 'power_curve_1', 'power_curve_2', 'power_curve_interp_factor', 'wind_derate_factor']
    out_cols = list(pd.unique(in_cols + return_cols))
    
    dataframe_sized = dataframe_sized.set_index('agent_id')

    return dataframe_sized[out_cols]


#%%   
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def process_wind_prices(wind_allowable_turbine_sizes, wind_price_traj):
    
    # join the data
    turbine_prices = pd.merge(wind_allowable_turbine_sizes[wind_allowable_turbine_sizes['allowed'] == True],
                              wind_price_traj, how='left', on=['turbine_size_kw'])
    
    # calculate cost for taller towers
    turbine_prices['tower_cost_adder_dollars_per_kw'] = turbine_prices['cost_for_higher_towers_dollars_per_kw_per_m'] * (
            turbine_prices['turbine_height_m'] - turbine_prices['default_tower_height_m'])
    
    # calculated installed costs (per kW)
    turbine_prices['installed_costs_dollars_per_kw'] = (turbine_prices['capital_cost_dollars_per_kw'] + 
                  turbine_prices['cost_for_higher_towers_dollars_per_kw_per_m'] * (turbine_prices['turbine_height_m'] - turbine_prices['default_tower_height_m']))
    
    return_cols= ['turbine_size_kw', 'turbine_height_m', 'year', 'capital_cost_dollars_per_kw', 'fixed_om_dollars_per_kw_per_yr', 'variable_om_dollars_per_kwh',
                  'cost_for_higher_towers_dollars_per_kw_per_m', 'tower_cost_adder_dollars_per_kw', 'installed_costs_dollars_per_kw']    
    
    return turbine_prices[return_cols]


#%%
@decorators.fn_timer(logger=logger, tab_level=2, prefix='')
def apply_wind_prices(dataframe, turbine_prices):

    in_cols = list(dataframe.columns)    
    in_cols.remove('agent_id')
    
    dataframe = dataframe.reset_index()

    # join the data
    dataframe = pd.merge(dataframe, turbine_prices, how='left', on=['turbine_size_kw', 'turbine_height_m', 'year'])

    # fill nas (these occur where system size is zero)
    dataframe['installed_costs_dollars_per_kw'] = dataframe['installed_costs_dollars_per_kw'].fillna(0)
    dataframe['fixed_om_dollars_per_kw_per_yr'] = dataframe['fixed_om_dollars_per_kw_per_yr'].fillna(0)
    dataframe['variable_om_dollars_per_kwh'] = dataframe['variable_om_dollars_per_kwh'].fillna(0)

    # apply the capital cost multipliers and generalize variable name
    dataframe['system_price_per_kw'] = (dataframe['installed_costs_dollars_per_kw'] * dataframe['cap_cost_multiplier'])
    
    # rename fixed O&M column for later compatibility
    dataframe.rename(columns={'fixed_om_dollars_per_kw_per_yr':'wind_om_per_kw'}, inplace=True)

    return_cols = ['system_price_per_kw', 'wind_om_per_kw', 'variable_om_dollars_per_kwh']
    out_cols = list(pd.unique(in_cols + return_cols))
    
    dataframe = dataframe.set_index('agent_id')

    return dataframe[out_cols]