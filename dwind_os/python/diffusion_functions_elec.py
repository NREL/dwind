"""
Name: diffusion_functions
Purpose: Contains functions to calculate diffusion of distributed wind model

    (1) Determine maximum market size as a function of payback time;
    (2) Parameterize Bass diffusion curve with diffusion rates (p, q) set by 
        payback time;
    (3) Determine current stage (equivaluent time) of diffusion based on existing 
        market and current economics 
    (4) Calculate new market share by stepping forward on diffusion curve.

"""

import numpy as np
import pandas as pd
import config
import utility_functions as utilfunc
import decorators

#==============================================================================
# Load logger
logger = utilfunc.get_logger()
#==============================================================================


@decorators.fn_timer(logger = logger, tab_level = 2, prefix = '')
def calc_diffusion(df, is_first_year, bass_params, year, techs,
                           override_p_value = None, override_q_value = None, override_teq_yr1_value = None):
    """
    Calculates the market share (ms) added in the solve year. Market share must be less
    than max market share (mms) except initial ms is greater than the calculated mms.
    For this circumstance, no diffusion allowed until mms > ms. Also, do not allow ms to
    decrease if economics deterioriate. Using the calculated 
    market share, relevant quantities are updated.

    IN: df - pd dataframe - Main dataframe
    
    OUT: df - pd dataframe - Main dataframe
        market_last_year - pd dataframe - market to inform diffusion in next year
    """
    
    df = df.reset_index()
    bass_params = bass_params[bass_params['tech']=='solar']    
    
    # set p/q/teq_yr1 params    
    df = pd.merge(df, bass_params[['state_abbr', 'bass_param_p', 'bass_param_q', 'teq_yr1', 'sector_abbr']], how = 'left', on  = ['state_abbr','sector_abbr'])
    
    # calc diffusion market share
    df = calc_diffusion_market_share(df, is_first_year)
    
    # market share floor is based on last year's market share
    df['market_share'] = np.maximum(df['diffusion_market_share'], df['market_share_last_year'])
   
    # calculate the "new" market share (old - current)
    df['new_market_share'] = df['market_share'] - df['market_share_last_year']

    # cap the new_market_share where the market share exceeds the max market share
    df['new_market_share'] = np.where(df['market_share'] > df['max_market_share'], 0, df['new_market_share'])

    # calculate new adopters, capacity and market value            
    df['new_adopters'] = df['new_market_share'] * df['developable_agent_weight']
    df['new_market_value'] = df['new_adopters'] * df['system_kw'] * df['system_capex_per_kw']

    df['new_system_kw'] = df['new_adopters'] * df['system_kw']
    df['new_batt_kw'] = df['new_adopters'] * df['batt_kw']
    df['new_batt_kwh'] = df['new_adopters'] * df['batt_kwh']

    # then add these values to values from last year to get cumulative values:
    df['number_of_adopters'] = df['adopters_cum_last_year'] + df['new_adopters']
    df['market_value'] = df['market_value_last_year'] + df['new_market_value']

    df['system_kw_cum'] = df['system_kw_cum_last_year'] + df['new_system_kw']
    df['batt_kw_cum'] = df['batt_kw_cum_last_year'] + df['new_batt_kw']
    df['batt_kwh_cum'] = df['batt_kwh_cum_last_year'] + df['new_batt_kwh']
    
    # constrain state-level capacity totals to known historical SOLAR adoption values
    if (year in (2014, 2016, 2018)) & ('solar' in techs):
        group_cols = ['state_abbr', 'sector_abbr', 'year']
        state_capacity_total = (df[group_cols+['system_kw_cum', 'agent_id']].groupby(group_cols)
                                                                            .agg({'system_kw_cum':'sum', 'agent_id':'count'})
                                                                            .rename(columns={'system_kw_cum':'state_kw_cum', 'agent_id':'agent_count'})
                                                                            .reset_index())
        
        # coerce dtypes
        state_capacity_total.state_kw_cum = state_capacity_total.state_kw_cum.astype(np.float64) 
        df.system_kw_cum = df.system_kw_cum.astype(np.float64) 
        
        # merge state totals back to agent df
        df = pd.merge(df, state_capacity_total, how = 'left', on = ['state_abbr', 'sector_abbr', 'year'])
        
        # read csv of historical capacity values by state and sector
        historical_state_capacity_df = pd.read_csv(config.INSTALLED_CAPACITY_BY_STATE)
        
        # join historical data to agent df
        df = pd.merge(df, historical_state_capacity_df, how='left', on=['state_abbr', 'sector_abbr', 'year'])
        
        # calculate scale factor - weight that is given to each agent based on proportion of state total
        # where state cumulative capacity is 0, proportion evenly to all agents
        df['scale_factor'] =  np.where(df['state_kw_cum'] == 0, 1.0/df['agent_count'], df['system_kw_cum'] / df['state_kw_cum'])
        
        # use scale factor to constrain agent capacity values to historical values
        df['system_kw_cum'] = df['scale_factor'] * df['observed_capacity_mw'] * 1000.
        
        # recalculate number of adopters using anecdotal values
        df['number_of_adopters'] = np.where(df['sector_abbr'] == 'res', df['system_kw_cum']/5.0, df['system_kw_cum']/100.0)
    
        # recalculate market share
        df['market_share'] = np.where(df['developable_agent_weight'] == 0, 0.0, 
                           df['number_of_adopters'] / df['developable_agent_weight'])
        df['market_share'] = df['market_share'].astype(np.float64)
        
        df.drop(['agent_count', 'state_kw_cum', 'state', 'observed_capacity_mw', 'scale_factor'], axis=1, inplace=True)
    
    market_last_year = df[['agent_id',
                            'market_share', 'max_market_share', 'number_of_adopters',
                            'market_value', 'initial_number_of_adopters', 'initial_pv_kw', 'initial_market_share', 'initial_market_value',
                            'system_kw_cum', 'new_system_kw', 'batt_kw_cum', 'batt_kwh_cum']]

    market_last_year.rename(columns={'market_share':'market_share_last_year', 
                               'max_market_share':'max_market_share_last_year',
                               'number_of_adopters':'adopters_cum_last_year',
                               'market_value': 'market_value_last_year',
                               'system_kw_cum':'system_kw_cum_last_year',
                               'batt_kw_cum':'batt_kw_cum_last_year',
                               'batt_kwh_cum':'batt_kwh_cum_last_year'}, inplace=True)

    return df, market_last_year


def calc_diffusion_market_share(df, is_first_year):
    """
    Calculate the fraction of overall population that have adopted the 
    technology in the current period. Note that this does not specify the 
    actual new adoption fraction without knowing adoption in the previous period. 

    IN: payback_period - numpy array - payback in years
        max_market_share - numpy array - maximum market share as decimal
        current_market_share - numpy array - current market share as decimal
                    
    OUT: new_market_share - numpy array - fraction of overall population 
                                            that have adopted the technology
    """

    # The relative economic attractiveness controls the p,q values in Bass diffusion
    # Current assumption is that only payback and MBS are being used, that pp is bounded [0-30] and MBS bounded [0-120]
       
    df = calc_equiv_time(df); # find the 'equivalent time' on the newly scaled diffusion curve
    if is_first_year == True:
        df['teq2'] = df['bass_params_teq'] + df['teq_yr1']
    else:
        df['teq2'] = df['bass_params_teq'] + 2 # now step forward two years from the 'new location'
    
    df = bass_diffusion(df); # calculate the new diffusion by stepping forward 2 years

    df['bass_market_share'] = df.max_market_share * df.new_adopt_fraction # new market adoption    
    df['diffusion_market_share'] = np.where(df.market_share_last_year > df.bass_market_share, df.market_share_last_year, df.bass_market_share)
    
    return df


def calc_equiv_time(df):
    """
    Calculate the "equivalent time" on the diffusion curve. This defines the
    gradient of adoption.

        IN: msly - numpy array - market share last year [at end of the previous solve] as decimal
            mms - numpy array - maximum market share as decimal
            p,q - numpy arrays - Bass diffusion parameters
            
        OUT: t_eq - numpy array - Equivalent number of years after diffusion 
                                  started on the diffusion curve
    """
    
    df['mms_fix_zeros'] = np.where(df['max_market_share'] == 0, 1e-9, df['max_market_share'])
    df['ratio'] = np.where(df['market_share_last_year'] > df['mms_fix_zeros'], 0, df['market_share_last_year']/df['mms_fix_zeros'])
    df['bass_params_teq'] = np.log((1 - df['ratio']) / (1 + df['ratio']*(df['bass_param_q']/df['bass_param_p']))) / (-1*(df['bass_param_p']+df['bass_param_q'])) # solve for equivalent time
   
    return df


def bass_diffusion(df):
    """
    Calculate the fraction of population that diffuse into the max_market_share.
    Note that this is different than the fraction of population that will 
    adopt, which is the max market share

    IN: p,q - numpy arrays - Bass diffusion parameters
        t - numpy array - Number of years since diffusion began
        
        
    OUT: new_adopt_fraction - numpy array - fraction of overall population 
                                            that will adopt the technology
    """
    df['f'] = np.e**(-1*(df['bass_param_p'] + df['bass_param_q']) * df['teq2'])
    df['new_adopt_fraction'] = (1-df['f']) / (1 + (df['bass_param_q']/df['bass_param_p'])*df['f']) # Bass Diffusion - cumulative adoption
    return df