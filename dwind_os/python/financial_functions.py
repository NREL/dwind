import numpy as np
import pandas as pd
import decorators
from scipy import optimize

import settings
import utility_functions as utilfunc
import agent_mutation

import PySAM.Battwatts as battery
import PySAM.BatteryTools as batt_tools
import PySAM.Utilityrate5 as utility
import PySAM.Cashloan as cashloan


#==============================================================================
# Load logger
logger = utilfunc.get_logger()
#==============================================================================


#%%
def calc_system_performance(kw, pv, utilityrate, loan, batt, costs, agent, en_batt=True, batt_simple_dispatch=0):
    """
    Executes Battwatts, Utilityrate5, and Cashloan PySAM modules with system sizes (kw) as input
    
    Parameters
    ----------
    kw: Capacity (in kW)
    pv: Dictionary with generation_hourly and consumption_hourly
    utilityrate: PySAM Utilityrate5 module
    loan: PySAM Cashloan module
    batt: PySAM Battwatts module
    costs: Dictionary with system costs
    agent: pd.Series with agent attributes
    en_batt: Enable battery
    batt_simple_dispatch: batt.Battery.batt_simple_dispatch
        - batt_simple_dispatch = 0 (peak shaving look ahead)
        - batt_simple_dispatch = 1 (peak shaving look behind)

    Returns
    -------
    -loan.Outputs.npv: the negative net present value of system + storage to be optimized for system sizing
    """

    inv_eff = 0.96  # default SAM inverter efficiency for PV
    gen_hourly = pv['generation_hourly']
    load_hourly = pv['consumption_hourly']  # same field as 'load_kwh_per_customer_in_bin_initial' when summed

    dc = [(i * kw) * 1000 for i in gen_hourly] # W
    ac = [i * inv_eff for i in dc] # W
    gen = [i / 1000 for i in ac] # W to kW
    
    # Set up battery, with system generation conditional on the battery generation being included
    if en_batt:

        batt.Battery.dc = dc
        batt.Battery.ac = ac
        batt.Battery.batt_simple_enable = 1
        batt.Battery.batt_simple_chemistry = 1  # default value is 1: li ion for residential
        batt.Battery.batt_simple_dispatch = batt_simple_dispatch
        batt.Battery.batt_simple_meter_position = 0  # default value
        batt.Battery.inverter_efficiency = 100  # recommended by Darice for dc-connected
        batt.Battery.load = load_hourly

        # PV to Battery ratio (kW) - From Ashreeta, 02/08/2020
        pv_to_batt_ratio = 1.31372
        batt_capacity_to_power_ratio = 2 # hours of operation
        
        desired_size = kw / pv_to_batt_ratio # Default SAM value for residential systems is 10 
        desired_power = desired_size / batt_capacity_to_power_ratio

        batt_inputs = {
            'batt_chem': batt.Battery.batt_simple_chemistry,
            'batt_Qfull': 2.5, # default SAM value
            'batt_Vnom_default': 3.6, # default SAM value
            'batt_ac_or_dc': 0,  # dc-connected
            'desired_power': desired_power,
            'desired_capacity': desired_size,
            'desired_voltage': 500,
            'size_by_ac_not_dc': 0,  # dc-connected
            'inverter_eff': batt.Battery.inverter_efficiency
            # 'batt_dc_dc_efficiency': (optional)
        }

        # Default values for lead acid batteries
        if batt.Battery.batt_simple_chemistry == 0:
            batt_inputs['LeadAcid_q10'] = 93.2
            batt_inputs['LeadAcid_q20'] = 100
            batt_inputs['LeadAcid_qn'] = 58.12
            # batt_inputs['LeadAcid_tn']: (optional)

        # PySAM.BatteryTools.size_li_ion_battery is the same as dGen_battery_sizing_battwatts.py
        batt_outputs = batt_tools.size_li_ion_battery(batt_inputs)

        computed_size = batt_outputs['batt_computed_bank_capacity']
        computed_power = batt_outputs['batt_power_discharge_max_kwdc']

        batt.Battery.batt_simple_kwh = computed_size
        batt.Battery.batt_simple_kw = computed_power

        batt.execute()
        
        # declare value for net billing sell rate
        if agent.loc['compensation_style']=='none':
            net_billing_sell_rate = 0.
        else:
            net_billing_sell_rate = agent.loc['wholesale_elec_price_dollars_per_kwh'] * agent.loc['elec_price_multiplier']
       
        utilityrate = process_tariff(utilityrate, agent.loc['tariff_dict'], net_billing_sell_rate)
        utilityrate.SystemOutput.gen = batt.Outputs.gen

        loan.BatterySystem.en_batt = 1
        loan.BatterySystem.batt_computed_bank_capacity = batt.Outputs.batt_bank_installed_capacity
        loan.BatterySystem.batt_bank_replacement = batt.Outputs.batt_bank_replacement
        
        # Battery capacity-based System Costs amount [$/kWhcap]
        loan.BatterySystem.battery_per_kWh = costs['batt_capex_per_kwh']
        
        # specify number of O&M types (1 = PV+batt)
        loan.SystemCosts.add_om_num_types = 1
        # specify O&M variables
        loan.SystemCosts.om_capacity = [costs['system_om_per_kw'] + costs['system_variable_om_per_kw']]
        loan.SystemCosts.om_capacity1 = [costs['batt_om_per_kw']]
        loan.SystemCosts.om_production1 = [costs['batt_om_per_kwh'] * 1000]
        loan.SystemCosts.om_replacement_cost1 = [0.]
        
        # Battery capacity for System Costs values [kW]
        loan.SystemCosts.om_capacity1_nameplate = batt.Battery.batt_simple_kw
        # Battery production for System Costs values [kWh]
        loan.SystemCosts.om_production1_values = [batt.Battery.batt_simple_kwh]

        batt_costs = ((costs['batt_capex_per_kw']*batt.Battery.batt_simple_kw) +
                      (costs['batt_capex_per_kwh'] * batt.Battery.batt_simple_kwh))
        
    else:
        batt.Battery.batt_simple_enable = 0
        loan.BatterySystem.en_batt = 0
        computed_power = computed_size = 0
        
        # declare value for net billing sell rate
        if agent.loc['compensation_style']=='none':
            net_billing_sell_rate = 0.
        else:
            net_billing_sell_rate = agent.loc['wholesale_elec_price_dollars_per_kwh'] * agent.loc['elec_price_multiplier']
        
        utilityrate = process_tariff(utilityrate, agent.loc['tariff_dict'], net_billing_sell_rate)
        utilityrate.SystemOutput.gen = gen
        
        # specify number of O&M types (0 = PV only)
        loan.SystemCosts.add_om_num_types = 0
        # specify O&M variables
        loan.SystemCosts.om_capacity = [costs['system_om_per_kw'] + costs['system_variable_om_per_kw']]
        loan.SystemCosts.om_replacement_cost1 = [0.]
        
        system_costs = costs['system_capex_per_kw'] * kw
        
        batt_costs = 0

    # Execute utility rate module
    utilityrate.Load.load = load_hourly
    
    utilityrate.execute()
    
    # Process payment incentives
    loan = process_incentives(loan, kw, computed_power, computed_size, gen_hourly, agent)
    
    # Specify final Cashloan parameters
    loan.FinancialParameters.system_capacity = kw
    loan.SystemOutput.annual_energy_value = utilityrate.Outputs.annual_energy_value
    loan.SystemOutput.gen = utilityrate.SystemOutput.gen
    loan.ThirdPartyOwnership.elec_cost_with_system = utilityrate.Outputs.elec_cost_with_system
    loan.ThirdPartyOwnership.elec_cost_without_system = utilityrate.Outputs.elec_cost_without_system

    # Calculate system costs
    direct_costs = (system_costs + batt_costs) * costs['cap_cost_multiplier']
    sales_tax = 0
    loan.SystemCosts.total_installed_cost = direct_costs + sales_tax
    
    # Execute financial module
    loan.execute()

    return -loan.Outputs.npv


def calc_system_size_and_performance_pv(agent, sectors, rate_switch_table=None):
    """
    Calculate the optimal system and battery size and generation profile, and resulting bill savings and financial metrics.
    
    Parameters
    ----------
    agent : 'pd.df'
        individual agent object.

    Returns
    -------
    agent: 'pd.df'
        Adds several features to the agent dataframe:

        - **agent_id**
        - **system_kw** - system capacity selected by agent
        - **batt_kw** - battery capacity selected by agent
        - **batt_kwh** - battery energy capacity
        - **npv** - net present value of system + storage
        - **cash_flow**  - array of annual cash flows from system adoption
        - **batt_dispatch_profile** - array of hourly battery dispatch
        - **annual_energy_production_kwh** - annual energy production (kwh) of system
        - **naep** - normalized annual energy production (kwh/kW) of system
        - **capacity_factor** - annual capacity factor
        - **first_year_elec_bill_with_system** - first year electricity bill with adopted system ($/yr)
        - **first_year_elec_bill_savings** - first year electricity bill savings with adopted system ($/yr)
        - **first_year_elec_bill_savings_frac** - fraction of savings on electricity bill in first year of system adoption
        - **max_system_kw** - maximum system size allowed as constrained by roof size or not exceeding annual consumption 
        - **first_year_elec_bill_without_system** - first year electricity bill without adopted system ($/yr)
        - **avg_elec_price_cents_per_kwh** - first year electricity price (c/kwh)
        - **cbi** - ndarray of capacity-based incentives applicable to agent
        - **ibi** - ndarray of investment-based incentives applicable to agent
        - **pbi** - ndarray of performance-based incentives applicable to agent
        - **cash_incentives** - ndarray of cash-based incentives applicable to agent
        - **export_tariff_result** - summary of structure of retail tariff applied to agent
    """

    # Initialize new DB connection    
    model_settings = settings.init_model_settings()
    con, cur = utilfunc.make_con(model_settings.pg_conn_string, model_settings.role)

    # PV
    pv = dict()

    if any('res' in ele for ele in sectors):
        #load_profile_df = agent_mutation.elec.get_and_apply_residential_agent_load_profiles(con, 'res', agent) # *** for full release, don't uncomment ***
        de_ts = pd.read_parquet(model_settings.load_path)
        s = str(agent.loc['bldg_id'])

    elif any('com' in ele for ele in sectors):
        #load_profile_df = agent_mutation.elec.get_and_apply_commercial_agent_load_profiles(con, 'com', agent) # *** for full release, don't uncomment ***
        de_ts = pd.read_parquet(model_settings.load_path)
        de_ts.rename(columns=lambda t: int(t.strip()), inplace=True)
        s = agent.loc['bldg_id']

    pv['consumption_hourly'] = pd.Series(de_ts[s].to_list())
    #pv['consumption_hourly'] = pd.Series(load_profile_df['consumption_hourly']).iloc[0] # *** for full release, don't uncomment ***

    # Using the scale offset factor of 1E6 for capacity factors
    norm_scaled_pv_cf_profiles_df = agent_mutation.elec.get_and_apply_normalized_hourly_resource_solar(con, agent)
    pv['generation_hourly'] = pd.Series(norm_scaled_pv_cf_profiles_df['solar_cf_profile'].iloc[0]) /  1e6
    del norm_scaled_pv_cf_profiles_df
    
    agent.loc['naep'] = float(np.sum(pv['generation_hourly']))

    # Battwatts
    if agent.loc['sector_abbr'] == 'res':
        batt = battery.default("PVWattsBatteryResidential")
    else:
        batt = battery.default("PVWattsBatteryCommercial")

    # Utilityrate5
    if agent.loc['sector_abbr'] == 'res':
        utilityrate = utility.default("PVWattsBatteryResidential")
    else:
        utilityrate = utility.default("PVWattsBatteryCommercial")

    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###--- SYSTEM LIFETIME SETTINGS ---###
    ######################################
    
    # Inflation rate [%]
    utilityrate.Lifetime.inflation_rate = agent.loc['inflation_rate'] * 100
    
    # Number of years in analysis [years]
    utilityrate.Lifetime.analysis_period = agent.loc['economic_lifetime_yrs']
    
    # Lifetime hourly system outputs [0/1]; Options: 0=hourly first year,1=hourly lifetime
    utilityrate.Lifetime.system_use_lifetime_output = 0


    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###---- DEGRADATION/ESCALATION ----###
    ######################################
    
    # Annual energy degradation [%]
    utilityrate.SystemOutput.degradation = [agent.loc['pv_degradation_factor'] * 100] # convert decimal to %
    # Annual electricity rate escalation [%/year]
    utilityrate.ElectricityRates.rate_escalation  = [agent.loc['elec_price_escalator'] * 100] # convert decimal to %
    
    
    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###---- NET METERING SETTINGS -----###
    ######################################
    
    # Dictionary to map dGen compensation styles to PySAM options
    nem_options = {'net metering':0, 'net billing':2, 'buy all sell all':4, 'none':2}
    # Metering options [0=net energy metering,1=net energy metering with $ credits,2=net billing,3=net billing with carryover to next month,4=buy all - sell all]
    utilityrate.ElectricityRates.ur_metering_option = nem_options[agent.loc['compensation_style']]
    # Year end sell rate [$/kWh]
    utilityrate.ElectricityRates.ur_nm_yearend_sell_rate = agent.loc['wholesale_elec_price_dollars_per_kwh'] * agent.loc['elec_price_multiplier']
    
    if agent.loc['compensation_style']=='none':
        net_billing_sell_rate = 0.
    else:
        net_billing_sell_rate = agent.loc['wholesale_elec_price_dollars_per_kwh'] * agent.loc['elec_price_multiplier']


    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###-------- BUY/SELL RATES --------###
    ######################################
    
    # Enable time step sell rates [0/1]
    utilityrate.ElectricityRates.ur_en_ts_sell_rate = 0
    
    # Time step sell rates [0/1]
    utilityrate.ElectricityRates.ur_ts_sell_rate = [0.]
    
    # Set sell rate equal to buy rate [0/1]
    utilityrate.ElectricityRates.ur_sell_eq_buy = 0
    
    
    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###-------- MISC. SETTINGS --------###
    ######################################
    
    # Use single monthly peak for TOU demand charge; options: 0=use TOU peak,1=use flat peak
    utilityrate.ElectricityRates.TOU_demand_single_peak = 0 # ?
    
    # Optionally enable/disable electricity_rate [years]
    utilityrate.ElectricityRates.en_electricity_rates = 1
    

    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###----- TARIFF RESTRUCTURING -----###
    ######################################
    utilityrate = process_tariff(utilityrate, agent.loc['tariff_dict'], net_billing_sell_rate)
    
    
    ######################################
    ###----------- CASHLOAN -----------###
    ###----- FINANCIAL PARAMETERS -----###
    ######################################
    
    # Initiate cashloan model and set market-specific variables
    # Assume res agents do not evaluate depreciation at all
    # Assume non-res agents only evaluate federal depreciation (not state)
    if agent.loc['sector_abbr'] == 'res':
        loan = cashloan.default("PVWattsBatteryResidential")
        loan.FinancialParameters.market = 0
    else:
        loan = cashloan.default("PVWattsBatteryCommercial")
        loan.FinancialParameters.market = 1

    loan.FinancialParameters.analysis_period = agent.loc['economic_lifetime_yrs']
    loan.FinancialParameters.debt_fraction = 100 - (agent.loc['down_payment_fraction'] * 100)
    loan.FinancialParameters.federal_tax_rate = [(agent.loc['tax_rate'] * 100) * 0.7] # SAM default
    loan.FinancialParameters.inflation_rate = agent.loc['inflation_rate'] * 100
    loan.FinancialParameters.insurance_rate = 0
    loan.FinancialParameters.loan_rate = agent.loc['loan_interest_rate'] * 100    
    loan.FinancialParameters.loan_term = agent.loc['loan_term_yrs']
    loan.FinancialParameters.mortgage = 0 # default value - standard loan (no mortgage)
    loan.FinancialParameters.prop_tax_assessed_decline = 5 # PySAM default
    loan.FinancialParameters.prop_tax_cost_assessed_percent = 95 # PySAM default
    loan.FinancialParameters.property_tax_rate = 0 # PySAM default
    loan.FinancialParameters.real_discount_rate = agent.loc['real_discount_rate'] * 100
    loan.FinancialParameters.salvage_percentage = 0    
    loan.FinancialParameters.state_tax_rate = [(agent.loc['tax_rate'] * 100) * 0.3] # SAM default
    loan.FinancialParameters.system_heat_rate = 0

    ######################################
    ###----------- CASHLOAN -----------###
    ###--------- SYSTEM COSTS ---------###
    ######################################

    # System costs that are input to loan.SystemCosts will depend on system configuration (PV, batt, PV+batt)
    # and are therefore specified in calc_system_performance()
    
    system_costs = dict()
    system_costs['system_capex_per_kw'] = agent.loc['system_capex_per_kw']
    system_costs['system_om_per_kw'] = agent.loc['system_om_per_kw']
    system_costs['system_variable_om_per_kw'] = agent.loc['system_variable_om_per_kw']
    system_costs['cap_cost_multiplier'] = agent.loc['cap_cost_multiplier']
    system_costs['batt_capex_per_kw'] = agent.loc['batt_capex_per_kw']
    system_costs['batt_capex_per_kwh'] = agent.loc['batt_capex_per_kwh']
    system_costs['batt_om_per_kw'] = agent.loc['batt_om_per_kw']
    system_costs['batt_om_per_kwh'] = agent.loc['batt_om_per_kwh']
    

    ######################################
    ###----------- CASHLOAN -----------###
    ###---- DEPRECIATION PARAMETERS ---###
    ######################################
    
    if agent.loc['sector_abbr'] == 'res':
        loan.Depreciation.depr_fed_type = 0
        loan.Depreciation.depr_sta_type = 0
    else:
        loan.Depreciation.depr_fed_type = 1
        loan.Depreciation.depr_sta_type = 0


    ######################################
    ###----------- CASHLOAN -----------###
    ###----- TAX CREDIT INCENTIVES ----###
    ######################################
    
    loan.TaxCreditIncentives.itc_fed_percent = agent.loc['itc_fraction_of_capex'] * 100
    
    
    ######################################
    ###----------- CASHLOAN -----------###
    ###-------- BATTERY SYSTEM --------###
    ######################################
    
    loan.BatterySystem.batt_replacement_option = 2 # user schedule
    
    batt_replacement_schedule = [0 for i in range(0, agent.loc['batt_lifetime_yrs'] - 1)] + [1]
    loan.BatterySystem.batt_replacement_schedule = batt_replacement_schedule
    
    
    ######################################
    ###----------- CASHLOAN -----------###
    ###-------- SYSTEM OUTPUT ---------###
    ######################################
    
    loan.SystemOutput.degradation = [agent.loc['pv_degradation_factor'] * 100]
    
    
    ######################################
    ###----------- CASHLOAN -----------###
    ###----------- LIFETIME -----------###
    ######################################
    
    loan.Lifetime.system_use_lifetime_output = 0
    

    
    # From dGen - calc_system_size_and_financial_performance()
    max_size_load = agent.loc['load_kwh_per_customer_in_bin'] / agent.loc['naep']
    max_size_roof = agent.loc['developable_roof_sqft'] * agent.loc['pv_kw_per_sqft']
    max_system_kw = min(max_size_load, max_size_roof)
    
    # set tolerance for minimize_scalar based on max_system_kw value
    tol = min(0.25 * max_system_kw, 0.5)

    # Calculate the PV system size that maximizes the agent's NPV, to a tolerance of 0.5 kW. 
    # Note that the optimization is technically minimizing negative NPV
    # ! As is, because of the tolerance this function would not necessarily return a system size of 0 or max PV size if those are optimal
    res_with_batt = optimize.minimize_scalar(calc_system_performance,
                                             args = (pv, utilityrate, loan, batt, system_costs, True, 0),
                                             bounds = (0, max_system_kw),
                                             method = 'bounded',
                                             tol = tol)

    # PySAM Module outputs with battery
    batt_loan_outputs = loan.Outputs.export()
    batt_util_outputs = utilityrate.Outputs.export()
    batt_annual_energy_kwh = np.sum(utilityrate.SystemOutput.gen)

    batt_kw = batt.Battery.batt_simple_kw
    batt_kwh = batt.Battery.batt_simple_kwh
    batt_dispatch_profile = batt.Outputs.batt_power # ?

    # Run without battery
    res_no_batt = optimize.minimize_scalar(calc_system_performance, 
                                           args = (pv, utilityrate, loan, batt, system_costs, False, 0),
                                           bounds = (0, max_system_kw),
                                           method = 'bounded',
                                           tol = tol)

    # PySAM Module outputs without battery
    no_batt_loan_outputs = loan.Outputs.export()
    no_batt_util_outputs = utilityrate.Outputs.export()
    no_batt_annual_energy_kwh = np.sum(utilityrate.SystemOutput.gen)

    # Retrieve NPVs of system with batt and system without batt
    npv_w_batt = batt_loan_outputs['npv']
    npv_no_batt = no_batt_loan_outputs['npv']

    # Choose the system with the higher NPV
    if npv_w_batt >= npv_no_batt:
        system_kw = res_with_batt.x
        annual_energy_production_kwh = batt_annual_energy_kwh
        first_year_elec_bill_with_system = batt_util_outputs['elec_cost_with_system_year1']
        first_year_elec_bill_without_system = batt_util_outputs['elec_cost_without_system_year1']

        npv = npv_w_batt
        payback = batt_loan_outputs['payback']
        cash_flow = list(batt_loan_outputs['cf_payback_with_expenses']) # ?

        cbi_total = batt_loan_outputs['cbi_total']
        cbi_total_fed = batt_loan_outputs['cbi_total_fed']
        cbi_total_oth = batt_loan_outputs['cbi_total_oth']
        cbi_total_sta = batt_loan_outputs['cbi_total_sta']
        cbi_total_uti = batt_loan_outputs['cbi_total_uti']

        ibi_total = batt_loan_outputs['ibi_total']
        ibi_total_fed = batt_loan_outputs['ibi_total_fed']
        ibi_total_oth = batt_loan_outputs['ibi_total_oth']
        ibi_total_sta = batt_loan_outputs['ibi_total_sta']
        ibi_total_uti = batt_loan_outputs['ibi_total_uti']

        cf_pbi_total = batt_loan_outputs['cf_pbi_total']
        pbi_total_fed = batt_loan_outputs['cf_pbi_total_fed']
        pbi_total_oth = batt_loan_outputs['cf_pbi_total_oth']
        pbi_total_sta = batt_loan_outputs['cf_pbi_total_sta']
        pbi_total_uti = batt_loan_outputs['cf_pbi_total_uti']


    else:
        system_kw = res_no_batt.x
        annual_energy_production_kwh = no_batt_annual_energy_kwh
        first_year_elec_bill_with_system = no_batt_util_outputs['elec_cost_with_system_year1']
        first_year_elec_bill_without_system = no_batt_util_outputs['elec_cost_without_system_year1']

        npv = npv_no_batt
        payback = no_batt_loan_outputs['payback']
        cash_flow = list(no_batt_loan_outputs['cf_payback_with_expenses'])

        batt_kw = 0
        batt_kwh = 0
        batt_dispatch_profile = np.nan

        cbi_total = no_batt_loan_outputs['cbi_total']
        cbi_total_fed = no_batt_loan_outputs['cbi_total_fed']
        cbi_total_oth = no_batt_loan_outputs['cbi_total_oth']
        cbi_total_sta = no_batt_loan_outputs['cbi_total_sta']
        cbi_total_uti = no_batt_loan_outputs['cbi_total_uti']

        ibi_total = no_batt_loan_outputs['ibi_total']
        ibi_total_fed = no_batt_loan_outputs['ibi_total_fed']
        ibi_total_oth = no_batt_loan_outputs['ibi_total_oth']
        ibi_total_sta = no_batt_loan_outputs['ibi_total_sta']
        ibi_total_uti = no_batt_loan_outputs['ibi_total_uti']

        cf_pbi_total = no_batt_loan_outputs['cf_pbi_total']
        pbi_total_fed = no_batt_loan_outputs['cf_pbi_total_fed']
        pbi_total_oth = no_batt_loan_outputs['cf_pbi_total_oth']
        pbi_total_sta = no_batt_loan_outputs['cf_pbi_total_sta']
        pbi_total_uti = no_batt_loan_outputs['cf_pbi_total_uti']
        

    # change 0 value to 1 to avoid divide by zero errors
    if first_year_elec_bill_without_system == 0:
        first_year_elec_bill_without_system = 1.0

    # Add outputs to agent df    
    naep = annual_energy_production_kwh / system_kw
    first_year_elec_bill_savings = first_year_elec_bill_without_system - first_year_elec_bill_with_system
    first_year_elec_bill_savings_frac = first_year_elec_bill_savings / first_year_elec_bill_without_system
    avg_elec_price_cents_per_kwh = first_year_elec_bill_without_system / agent.loc['load_kwh_per_customer_in_bin']

    agent.loc['system_kw'] = system_kw
    agent.loc['npv'] = npv
    agent.loc['payback_period'] = np.round(np.where(np.isnan(payback), 30.1, payback), 1).astype(float)
    agent.loc['cash_flow'] = cash_flow
    agent.loc['annual_energy_production_kwh'] = annual_energy_production_kwh
    agent.loc['naep'] = naep
    agent.loc['capacity_factor'] = agent.loc['naep'] / 8760
    agent.loc['first_year_elec_bill_with_system'] = first_year_elec_bill_with_system
    agent.loc['first_year_elec_bill_savings'] = first_year_elec_bill_savings
    agent.loc['first_year_elec_bill_savings_frac'] = first_year_elec_bill_savings_frac
    agent.loc['max_system_kw'] = max_system_kw
    agent.loc['first_year_elec_bill_without_system'] = first_year_elec_bill_without_system
    agent.loc['avg_elec_price_cents_per_kwh'] = avg_elec_price_cents_per_kwh
    agent.loc['batt_kw'] = batt_kw
    agent.loc['batt_kwh'] = batt_kwh
    agent.loc['batt_dispatch_profile'] = batt_dispatch_profile

    # Financial outputs (find out which ones to include): 
    agent.loc['cbi'] = np.array({'cbi_total': cbi_total,
            'cbi_total_fed': cbi_total_fed,
            'cbi_total_oth': cbi_total_oth,
            'cbi_total_sta': cbi_total_sta,
            'cbi_total_uti': cbi_total_uti
           })
    agent.loc['ibi'] = np.array({'ibi_total': ibi_total,
            'ibi_total_fed': ibi_total_fed,
            'ibi_total_oth': ibi_total_oth,
            'ibi_total_sta': ibi_total_sta,
            'ibi_total_uti': ibi_total_uti
           })
    agent.loc['pbi'] = np.array({'pbi_total': cf_pbi_total,
            'pbi_total_fed': pbi_total_fed,
            'pbi_total_oth': pbi_total_oth,
            'pbi_total_sta': pbi_total_sta,
            'pbi_total_uti': pbi_total_uti
            })
    agent.loc['cash_incentives'] = ''
    agent.loc['export_tariff_results'] = ''

    out_cols = ['agent_id',
                'system_kw',
                'batt_kw',
                'batt_kwh',
                'npv',
                'payback_period',
                'cash_flow',
                'batt_dispatch_profile',
                'annual_energy_production_kwh',
                'naep',
                'capacity_factor',
                'first_year_elec_bill_with_system',
                'first_year_elec_bill_savings',
                'first_year_elec_bill_savings_frac',
                'max_system_kw',
                'first_year_elec_bill_without_system',
                'avg_elec_price_cents_per_kwh',
                'cbi',
                'ibi',
                'pbi',
                'cash_incentives',
                'export_tariff_results'
                ]

    return agent[out_cols]


#%%
def calc_financial_performance_wind(agent, sectors, rate_switch_table=None):
    """
    Calculate bill savings and financial metrics based on pre-selected wind system size.
    
    Parameters
    ----------
    agent : 'pd.df'
        individual agent object.

    Returns
    -------
    agent: 'pd.df'
        Adds several features to the agent dataframe:

        - **agent_id**
        - **system_kw** - system capacity selected by agent
        - **npv** - net present value of system + storage
        - **cash_flow**  - array of annual cash flows from system adoption
        - **batt_dispatch_profile** - array of hourly battery dispatch
        - **annual_energy_production_kwh** - annual energy production (kwh) of system
        - **naep** - normalized annual energy production (kwh/kW) of system
        - **capacity_factor** - annual capacity factor
        - **first_year_elec_bill_with_system** - first year electricity bill with adopted system ($/yr)
        - **first_year_elec_bill_savings** - first year electricity bill savings with adopted system ($/yr)
        - **first_year_elec_bill_savings_frac** - fraction of savings on electricity bill in first year of system adoption
        - **max_system_kw** - maximum system size allowed as constrained by roof size or not exceeding annual consumption 
        - **first_year_elec_bill_without_system** - first year electricity bill without adopted system ($/yr)
        - **avg_elec_price_cents_per_kwh** - first year electricity price (c/kwh)
        - **cbi** - ndarray of capacity-based incentives applicable to agent
        - **ibi** - ndarray of investment-based incentives applicable to agent
        - **pbi** - ndarray of performance-based incentives applicable to agent
        - **cash_incentives** - ndarray of cash-based incentives applicable to agent
        - **export_tariff_result** - summary of structure of retail tariff applied to agent
    """
    
    # Initialize new DB connection    
    model_settings = settings.init_model_settings()
    con, cur = utilfunc.make_con(model_settings.pg_conn_string, model_settings.role)

    if any('res' in ele for ele in sectors):
        #load_profile_df = agent_mutation.elec.get_and_apply_residential_agent_load_profiles(con, 'res', agent) # *** for full release, don't uncomment ***
        de_ts = pd.read_parquet(model_settings.load_path)
        s = str(agent.loc['bldg_id'])

    elif any('com' in ele for ele in sectors):
        #load_profile_df = agent_mutation.elec.get_and_apply_commercial_agent_load_profiles(con, 'com', agent) # *** for full release, don't uncomment ***
        de_ts = pd.read_parquet(model_settings.load_path)
        de_ts.rename(columns=lambda t: int(t.strip()), inplace=True)
        s = agent.loc['bldg_id']

    consumption_hourly = pd.Series(de_ts[s].to_list())
    # consumption_hourly = pd.Series(load_profile_df['consumption_hourly']).iloc[0] # *** for full release, don't uncomment ***

    # Using the scale offset factor of 1E6 for capacity factors
    norm_scaled_wind_profiles_df = agent_mutation.elec.get_and_apply_normalized_hourly_resource_wind(con, agent)
    generation_hourly = pd.Series(norm_scaled_wind_profiles_df['generation_hourly']).iloc[0]
    del norm_scaled_wind_profiles_df
        
    # Instantiate utilityrate5 model based on agent sector
    if agent.loc['sector_abbr'] == 'res':
        utilityrate = utility.default('WindPowerResidential')
    else:
        utilityrate = utility.default('WindPowerCommercial')
        
        
    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###------- ELECTRICITYRATES -------###
    ######################################
    
    # Use single monthly peak for TOU demand charge; options: 0=use TOU peak,1=use flat peak
    utilityrate.ElectricityRates.TOU_demand_single_peak = 0 # ?
    # Optionally enable/disable electricity_rate [years]
    utilityrate.ElectricityRates.en_electricity_rates = 1
    # Annual electricity rate escalation [%/year]
    utilityrate.ElectricityRates.rate_escalation  = [agent.loc['elec_price_escalator'] * 100] # convert decimal to %
    # Enable time step sell rates [0/1]
    utilityrate.ElectricityRates.ur_en_ts_sell_rate = 0
    # Time step sell rates [0/1]
    utilityrate.ElectricityRates.ur_ts_sell_rate = [0.]    
    # Set sell rate equal to buy rate [0/1]
    utilityrate.ElectricityRates.ur_sell_eq_buy = 0
    
    
    # Dictionary to map dGen compensation styles to PySAM options
    nem_options = {'net metering':0, 'net billing':2, 'buy all sell all':4, 'none':2}
    # Metering options [0=net energy metering,1=net energy metering with $ credits,2=net billing,3=net billing with carryover to next month,4=buy all - sell all]
    utilityrate.ElectricityRates.ur_metering_option = nem_options[agent.loc['compensation_style']]
    # Year end sell rate [$/kWh]
    utilityrate.ElectricityRates.ur_nm_yearend_sell_rate = agent.loc['wholesale_elec_price_dollars_per_kwh'] * agent.loc['elec_price_multiplier']
    
    if agent.loc['compensation_style']=='none':
        net_billing_sell_rate = 0.
    else:
        net_billing_sell_rate = agent.loc['wholesale_elec_price_dollars_per_kwh'] * agent.loc['elec_price_multiplier']
    
    
    # Restructure tariff object for PySAM compatibility
    utilityrate = process_tariff(utilityrate, agent.loc['tariff_dict'], net_billing_sell_rate)


    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###----------- LIFETIME -----------###
    ######################################

    # Number of years in analysis [years]
    utilityrate.Lifetime.analysis_period = agent.loc['economic_lifetime_yrs']
    # Inflation rate [%]
    utilityrate.Lifetime.inflation_rate = agent.loc['inflation_rate'] * 100
    # Lifetime hourly system outputs [0/1]; Options: 0=hourly first year,1=hourly lifetime
    utilityrate.Lifetime.system_use_lifetime_output = 0


    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###-------- SYSTEM OUTPUT ---------###
    ######################################
    
    # Annual energy degradation [%] -- Wind degradation already applied via 'derate_factor'
    utilityrate.SystemOutput.degradation = [0.]
    # System power generated [kW]
    utilityrate.SystemOutput.gen = generation_hourly

    
    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###-------- SYSTEM OUTPUT ---------###
    ######################################

    # Electricity load (year 1) [kW]
    utilityrate.Load.load = consumption_hourly
    

    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###------------ EXECUTE -----------###
    ######################################

    utilityrate.execute()
    
    
    ######################################
    ###----------- CASHLOAN -----------###
    ###----- FINANCIAL PARAMETERS -----###
    ######################################
    
    # Initiate cashloan model and set market-specific variables
    if agent.loc['sector_abbr'] == 'res':
        loan = cashloan.default('WindPowerResidential')
        loan.FinancialParameters.market = 0
    else:
        loan = cashloan.default('WindPowerCommercial')
        loan.FinancialParameters.market = 1

    loan.FinancialParameters.analysis_period = agent.loc['economic_lifetime_yrs']
    loan.FinancialParameters.debt_fraction = 100 - (agent.loc['down_payment_fraction'] * 100)
    loan.FinancialParameters.federal_tax_rate = [(agent.loc['tax_rate'] * 100) * 0.7] # SAM default
    loan.FinancialParameters.inflation_rate = agent.loc['inflation_rate'] * 100
    loan.FinancialParameters.insurance_rate = 0
    loan.FinancialParameters.loan_rate = agent.loc['loan_interest_rate'] * 100    
    loan.FinancialParameters.loan_term = agent.loc['loan_term_yrs']
    loan.FinancialParameters.mortgage = 0 # default value - standard loan (no mortgage)
    loan.FinancialParameters.prop_tax_assessed_decline = 5 # PySAM default
    loan.FinancialParameters.prop_tax_cost_assessed_percent = 95 # PySAM default
    loan.FinancialParameters.property_tax_rate = 0 # PySAM default
    loan.FinancialParameters.real_discount_rate = agent.loc['real_discount_rate'] * 100
    loan.FinancialParameters.salvage_percentage = 0    
    loan.FinancialParameters.state_tax_rate = [(agent.loc['tax_rate'] * 100) * 0.3] # SAM default
    loan.FinancialParameters.system_heat_rate = 0
    
    loan.FinancialParameters.system_capacity = agent.loc['system_size_kw']


    ######################################
    ###----------- CASHLOAN -----------###
    ###--------- SYSTEM COSTS ---------###
    ######################################
    
    # specify number of O&M types (0 = system only)
    loan.SystemCosts.add_om_num_types = 0
    # specify O&M variables
    loan.SystemCosts.om_capacity = [agent.loc['system_om_per_kw'] + agent.loc['system_variable_om_per_kw']]

    # Calculate and specify system costs
    system_costs = agent.loc['system_capex_per_kw'] * agent.loc['system_size_kw']
    batt_costs = 0
    sales_tax = 0
    direct_costs = (system_costs + batt_costs) * agent.loc['cap_cost_multiplier']
    loan.SystemCosts.total_installed_cost = direct_costs + sales_tax
    
    
    ######################################
    ###----------- CASHLOAN -----------###
    ###---- DEPRECIATION PARAMETERS ---###
    ######################################
    
    # Federal and State depreciation type
    # Options: 0=none, 1=MACRS half year, 2=straight-line, 3=custom
    if agent.loc['sector_abbr'] == 'res':
        loan.Depreciation.depr_fed_type = 0
        loan.Depreciation.depr_sta_type = 0
    else:
        loan.Depreciation.depr_fed_type = 1
        loan.Depreciation.depr_sta_type = 0


    ######################################
    ###----------- CASHLOAN -----------###
    ###----- TAX CREDIT INCENTIVES ----###
    ######################################
    
    # Federal percentage-based ITC percent [%]
    loan.TaxCreditIncentives.itc_fed_percent = agent.loc['itc_fraction_of_capex'] * 100
    
    
    ######################################
    ###----------- CASHLOAN -----------###
    ###------ PAYMENT INCENTIVES ------###
    ######################################
    
    # Specify payment incentives within Cashloan object
    loan = process_incentives(loan, agent.loc['system_size_kw'], 0, 0, generation_hourly, agent)
    
    
    ######################################
    ###----------- CASHLOAN -----------###
    ###-------- BATTERY SYSTEM --------###
    ######################################
    
    # Enable battery storage model [0/1]
    loan.BatterySystem.en_batt = 0
    
    
    ######################################
    ###----------- CASHLOAN -----------###
    ###-------- SYSTEM OUTPUT ---------###
    ######################################
    
    # Energy value [$] -- i.e. "bill savings"
    loan.SystemOutput.annual_energy_value = utilityrate.Outputs.annual_energy_value
    # Annual energy degradation [%] -- Wind degradation already applied via 'derate_factor'
    loan.SystemOutput.degradation = [0.]
    # Power generated by renewable resource [kW]
    loan.SystemOutput.gen = utilityrate.SystemOutput.gen
    
    
    ######################################
    ###----------- CASHLOAN -----------###
    ###----------- LIFETIME -----------###
    ######################################
    
    loan.Lifetime.system_use_lifetime_output = 0


    ######################################
    ###----------- CASHLOAN -----------###
    ###----- THIRD PARTY OWNERSHIP ----###
    ######################################
    
    # Energy value [$]
    loan.ThirdPartyOwnership.elec_cost_with_system = utilityrate.Outputs.elec_cost_with_system
    # Energy value [$]
    loan.ThirdPartyOwnership.elec_cost_without_system = utilityrate.Outputs.elec_cost_without_system


    ######################################
    ###-------- POSTPROCESSING --------###
    ###------------ RESULTS -----------###
    ######################################
    
    # Get outputs from Utilityrate5 model
    util_outputs = utilityrate.Outputs.export()
    
    # Assign variables from Utilityrate5 outputs, others
    system_kw = agent.loc['system_size_kw']
    first_year_elec_bill_with_system = util_outputs['elec_cost_with_system_year1']
    first_year_elec_bill_without_system = util_outputs['elec_cost_without_system_year1']
    
    # PySAM cannot evaluate system sizes of 0 kW -- check and manually assign values if system_size_kw = 0
    if system_kw > 0:
        
        # Execute Cashloan model
        loan.execute()
        loan_outputs = loan.Outputs.export()
    
        npv = loan_outputs['npv']
        payback = loan_outputs['payback']
        cash_flow = list(loan_outputs['cf_payback_with_expenses'])
    
        cbi_total = loan_outputs['cbi_total']
        cbi_total_fed = loan_outputs['cbi_total_fed']
        cbi_total_oth = loan_outputs['cbi_total_oth']
        cbi_total_sta = loan_outputs['cbi_total_sta']
        cbi_total_uti = loan_outputs['cbi_total_uti']
    
        ibi_total = loan_outputs['ibi_total']
        ibi_total_fed = loan_outputs['ibi_total_fed']
        ibi_total_oth = loan_outputs['ibi_total_oth']
        ibi_total_sta = loan_outputs['ibi_total_sta']
        ibi_total_uti = loan_outputs['ibi_total_uti']
    
        cf_pbi_total = loan_outputs['cf_pbi_total']
        pbi_total_fed = loan_outputs['cf_pbi_total_fed']
        pbi_total_oth = loan_outputs['cf_pbi_total_oth']
        pbi_total_sta = loan_outputs['cf_pbi_total_sta']
        pbi_total_uti = loan_outputs['cf_pbi_total_uti']
        
    else:
        
        npv = 0.
        payback = 30.1
        cash_flow = [0.] * (agent.loc['economic_lifetime_yrs'] + 1)
    
        cbi_total = cbi_total_fed = cbi_total_oth = cbi_total_sta = cbi_total_uti = 0.
        ibi_total = ibi_total_fed = ibi_total_oth = ibi_total_sta = ibi_total_uti = 0.
        cf_pbi_total = pbi_total_fed = pbi_total_oth = pbi_total_sta = pbi_total_uti = 0.
    
    # change 0 value to 1 to avoid divide by zero errors
    if first_year_elec_bill_without_system == 0:
        first_year_elec_bill_without_system = 1.0
    
    # Add outputs to agent df    
    first_year_elec_bill_savings = first_year_elec_bill_without_system - first_year_elec_bill_with_system
    first_year_elec_bill_savings_frac = first_year_elec_bill_savings / first_year_elec_bill_without_system
    avg_elec_price_cents_per_kwh = first_year_elec_bill_without_system / agent.loc['load_kwh_per_customer_in_bin']        

    # Specify variables to write to agent df -- also write placeholder batt values
    agent.loc['system_kw'] = system_kw
    agent.loc['npv'] = npv
    agent.loc['payback_period'] = np.round(np.where(np.isnan(payback), 30.1, payback), 1).astype(float)
    agent.loc['cash_flow'] = cash_flow
    agent.loc['first_year_elec_bill_with_system'] = first_year_elec_bill_with_system
    agent.loc['first_year_elec_bill_savings'] = first_year_elec_bill_savings
    agent.loc['first_year_elec_bill_savings_frac'] = first_year_elec_bill_savings_frac
    agent.loc['first_year_elec_bill_without_system'] = first_year_elec_bill_without_system
    agent.loc['avg_elec_price_cents_per_kwh'] = avg_elec_price_cents_per_kwh
    agent.loc['batt_kw'] = 0.
    agent.loc['batt_kwh'] = 0.
    agent.loc['batt_dispatch_profile'] = np.nan

    # Specify incentive outputs
    agent.loc['cbi'] = np.array({'cbi_total': cbi_total,
            'cbi_total_fed': cbi_total_fed,
            'cbi_total_oth': cbi_total_oth,
            'cbi_total_sta': cbi_total_sta,
            'cbi_total_uti': cbi_total_uti
           })
    agent.loc['ibi'] = np.array({'ibi_total': ibi_total,
            'ibi_total_fed': ibi_total_fed,
            'ibi_total_oth': ibi_total_oth,
            'ibi_total_sta': ibi_total_sta,
            'ibi_total_uti': ibi_total_uti
           })
    agent.loc['pbi'] = np.array({'pbi_total': cf_pbi_total,
            'pbi_total_fed': pbi_total_fed,
            'pbi_total_oth': pbi_total_oth,
            'pbi_total_sta': pbi_total_sta,
            'pbi_total_uti': pbi_total_uti
            })
    agent.loc['cash_incentives'] = ''
    agent.loc['export_tariff_results'] = ''

    out_cols = ['agent_id',
                'system_kw',
                'npv',
                'payback_period',
                'cash_flow',
                'first_year_elec_bill_with_system',
                'first_year_elec_bill_savings',
                'first_year_elec_bill_savings_frac',
                'first_year_elec_bill_without_system',
                'avg_elec_price_cents_per_kwh',
                'cbi',
                'ibi',
                'pbi',
                'cash_incentives',
                'export_tariff_results',
                'batt_kw',
                'batt_kwh',
                'batt_dispatch_profile'
                ]

    return agent[out_cols]


#%%
def process_tariff(utilityrate, tariff_dict, net_billing_sell_rate):
    """
    Instantiate the utilityrate5 PySAM model and process the agent's rate json object to conform with PySAM input formatting.
    
    Parameters
    ----------
    agent : 'pd.Series'
        Individual agent object.
    Returns
    -------
    utilityrate: 'PySAM.Utilityrate5'
    """    
    
    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###--- FIXED AND ANNUAL CHARGES ---###
    ######################################
    
    # Monthly fixed charge [$]
    utilityrate.ElectricityRates.ur_monthly_fixed_charge = tariff_dict['fixed_charge']
    # Annual minimum charge [$]
    utilityrate.ElectricityRates.ur_annual_min_charge = 0. # not currently tracked in URDB rate attribute downloads
    # Monthly minimum charge [$]
    utilityrate.ElectricityRates.ur_monthly_min_charge = 0. # not currently tracked in URDB rate attribute downloads
    
    
    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###-------- DEMAND CHARGES --------###
    ######################################
    
    # Enable demand charge
    utilityrate.ElectricityRates.ur_dc_enable = (tariff_dict['d_flat_exists']) | (tariff_dict['d_tou_exists'])
    
    if utilityrate.ElectricityRates.ur_dc_enable:
    
        if tariff_dict['d_flat_exists']:
            
            # Reformat demand charge table from dGen format
            n_periods = len(tariff_dict['d_flat_levels'][0])
            n_tiers = len(tariff_dict['d_flat_levels'])
            ur_dc_flat_mat = []
            for period in range(n_periods):
                for tier in range(n_tiers):
                    row = [period, tier+1, tariff_dict['d_flat_levels'][tier][period], tariff_dict['d_flat_prices'][tier][period]]
                    ur_dc_flat_mat.append(row)
            
            # Demand rates (flat) table
            utilityrate.ElectricityRates.ur_dc_flat_mat = ur_dc_flat_mat
        
        
        if tariff_dict['d_tou_exists']:
            
            # Reformat demand charge table from dGen format
            n_periods = len(tariff_dict['d_tou_levels'][0])
            n_tiers = len(tariff_dict['d_tou_levels'])
            ur_dc_tou_mat = []
            for period in range(n_periods):
                for tier in range(n_tiers):
                    row = [period+1, tier+1, tariff_dict['d_tou_levels'][tier][period], tariff_dict['d_tou_prices'][tier][period]]
                    ur_dc_tou_mat.append(row)
            
            # Demand rates (TOU) table
            utilityrate.ElectricityRates.ur_dc_tou_mat = ur_dc_tou_mat
    
    
        # Reformat 12x24 tables - original are indexed to 0, PySAM needs index starting at 1
        d_wkday_12by24 = []
        for m in range(len(tariff_dict['d_wkday_12by24'])):
            row = [x+1 for x in tariff_dict['d_wkday_12by24'][m]]
            d_wkday_12by24.append(row)
            
        d_wkend_12by24 = []
        for m in range(len(tariff_dict['d_wkend_12by24'])):
            row = [x+1 for x in tariff_dict['d_wkend_12by24'][m]]
            d_wkend_12by24.append(row)

        # Demand charge weekday schedule
        utilityrate.ElectricityRates.ur_dc_sched_weekday = d_wkday_12by24
        # Demand charge weekend schedule
        utilityrate.ElectricityRates.ur_dc_sched_weekend = d_wkend_12by24
    
    
    ######################################
    ###--------- UTILITYRATE5 ---------###
    ###-------- ENERGY CHARGES --------###
    ######################################
    
    if tariff_dict['e_exists']:
        
        # Dictionary to map dGen max usage units to PySAM options
        max_usage_dict = {'kWh':0, 'kWh/kW':1, 'kWh daily':2, 'kWh/kW daily':3}
        # If max usage units are 'kWh daily', divide max usage by 30 -- rate download procedure converts daily to monthly
        modifier = 30. if tariff_dict['energy_rate_unit'] == 'kWh daily' else 1.
        
        # Reformat energy charge table from dGen format
        n_periods = len(tariff_dict['e_levels'][0])
        n_tiers = len(tariff_dict['e_levels'])
        ur_ec_tou_mat = []
        for period in range(n_periods):
            for tier in range(n_tiers):
                row = [period+1, tier+1, tariff_dict['e_levels'][tier][period]/modifier, max_usage_dict[tariff_dict['energy_rate_unit']], tariff_dict['e_prices'][tier][period], net_billing_sell_rate]
                ur_ec_tou_mat.append(row)
        
        # Energy rates table
        utilityrate.ElectricityRates.ur_ec_tou_mat = ur_ec_tou_mat
        
        # Reformat 12x24 tables - original are indexed to 0, PySAM needs index starting at 1
        e_wkday_12by24 = []
        for m in range(len(tariff_dict['e_wkday_12by24'])):
            row = [x+1 for x in tariff_dict['e_wkday_12by24'][m]]
            e_wkday_12by24.append(row)
            
        e_wkend_12by24 = []
        for m in range(len(tariff_dict['e_wkend_12by24'])):
            row = [x+1 for x in tariff_dict['e_wkend_12by24'][m]]
            e_wkend_12by24.append(row)
        
        # Energy charge weekday schedule
        utilityrate.ElectricityRates.ur_ec_sched_weekday = e_wkday_12by24
        # Energy charge weekend schedule
        utilityrate.ElectricityRates.ur_ec_sched_weekend = e_wkend_12by24
        
    
    return utilityrate


#%%
def process_incentives(loan, kw, batt_kw, batt_kwh, generation_hourly, agent):
    
    ######################################
    ###----------- CASHLOAN -----------###
    ###------ PAYMENT INCENTIVES ------###
    ######################################

    # Read incentive dataframe from agent attributes
    incentive_df = agent.loc['state_incentives']
    
    # Check dtype of incentive_df - process incentives if pd.DataFrame, otherwise do not assign incentive values to cashloan
    if isinstance(incentive_df, pd.DataFrame):
        
        # Fill NaNs in incentive_df - assume max incentive duration of 5 years and max incentive value of $10,000
        incentive_df = incentive_df.fillna(value={'incentive_duration_yrs' : 5, 'max_incentive_usd' : 10000})
        # Filter for CBI's in incentive_df
        cbi_df = (incentive_df.loc[pd.notnull(incentive_df['cbi_usd_p_w'])]                  
                  .sort_values(['cbi_usd_p_w'], axis=0, ascending=False)
                  .reset_index(drop=True)
                 )
        
        # For multiple CBIs that are applicable to the agent, cap at 2 and use PySAM's "state" and "other" option
        if len(cbi_df) == 1:
            
            loan.PaymentIncentives.cbi_sta_amount = cbi_df['cbi_usd_p_w'].iloc[0]
            loan.PaymentIncentives.cbi_sta_deprbas_fed = 0
            loan.PaymentIncentives.cbi_sta_deprbas_sta = 0
            loan.PaymentIncentives.cbi_sta_maxvalue = cbi_df['max_incentive_usd'].iloc[0]
            loan.PaymentIncentives.cbi_sta_tax_fed = 0
            loan.PaymentIncentives.cbi_sta_tax_sta = 0
            
        elif len(cbi_df) >= 2:
            
            loan.PaymentIncentives.cbi_sta_amount = cbi_df['cbi_usd_p_w'].iloc[0]
            loan.PaymentIncentives.cbi_sta_deprbas_fed = 0
            loan.PaymentIncentives.cbi_sta_deprbas_sta = 0
            loan.PaymentIncentives.cbi_sta_maxvalue = cbi_df['max_incentive_usd'].iloc[0]
            loan.PaymentIncentives.cbi_sta_tax_fed = 1
            loan.PaymentIncentives.cbi_sta_tax_sta = 1
            
            loan.PaymentIncentives.cbi_oth_amount = cbi_df['cbi_usd_p_w'].iloc[1]
            loan.PaymentIncentives.cbi_oth_deprbas_fed = 0
            loan.PaymentIncentives.cbi_oth_deprbas_sta = 0
            loan.PaymentIncentives.cbi_oth_maxvalue = cbi_df['max_incentive_usd'].iloc[1]
            loan.PaymentIncentives.cbi_oth_tax_fed = 1
            loan.PaymentIncentives.cbi_oth_tax_sta = 1
            
        else:
            pass
        
        # Filter for PBI's in incentive_df
        pbi_df = (incentive_df.loc[pd.notnull(incentive_df['pbi_usd_p_kwh'])]
                  .sort_values(['pbi_usd_p_kwh'], axis=0, ascending=False)
                  .reset_index(drop=True)
                 )
    
        # For multiple PBIs that are applicable to the agent, cap at 2 and use PySAM's "state" and "other" option
        if len(pbi_df) == 1:
            
            # Aamount input [$/kWh] requires sequence -- repeat pbi_usd_p_kwh using incentive_duration_yrs 
            loan.PaymentIncentives.pbi_sta_amount = [pbi_df['pbi_usd_p_kwh'].iloc[0]] * int(pbi_df['incentive_duration_yrs'].iloc[0])
            loan.PaymentIncentives.pbi_sta_escal = 0.
            loan.PaymentIncentives.pbi_sta_tax_fed = 1
            loan.PaymentIncentives.pbi_sta_tax_sta = 1
            loan.PaymentIncentives.pbi_sta_term = pbi_df['incentive_duration_yrs'].iloc[0]
            
        elif len(pbi_df) >= 2:
            
            # Aamount input [$/kWh] requires sequence -- repeat pbi_usd_p_kwh using incentive_duration_yrs 
            loan.PaymentIncentives.pbi_sta_amount = [pbi_df['pbi_usd_p_kwh'].iloc[0]] * int(pbi_df['incentive_duration_yrs'].iloc[0])
            loan.PaymentIncentives.pbi_sta_escal = 0.
            loan.PaymentIncentives.pbi_sta_tax_fed = 1
            loan.PaymentIncentives.pbi_sta_tax_sta = 1
            loan.PaymentIncentives.pbi_sta_term = pbi_df['incentive_duration_yrs'].iloc[0]
            
            # Aamount input [$/kWh] requires sequence -- repeat pbi_usd_p_kwh using incentive_duration_yrs 
            loan.PaymentIncentives.pbi_oth_amount = [pbi_df['pbi_usd_p_kwh'].iloc[1]] * int(pbi_df['incentive_duration_yrs'].iloc[1])
            loan.PaymentIncentives.pbi_oth_escal = 0.
            loan.PaymentIncentives.pbi_oth_tax_fed = 1
            loan.PaymentIncentives.pbi_oth_tax_sta = 1
            loan.PaymentIncentives.pbi_oth_term = pbi_df['incentive_duration_yrs'].iloc[1]
            
        else:
            pass
        
        # Filter for IBI's in incentive_df
        ibi_df = (incentive_df.loc[pd.notnull(incentive_df['ibi_pct'])]
                  .sort_values(['ibi_pct'], axis=0, ascending=False)
                  .reset_index(drop=True)
                 )
        
        # For multiple IBIs that are applicable to the agent, cap at 2 and use PySAM's "state" and "other" option
        # NOTE: this specifies IBI percentage, instead of IBI absolute amount
        if len(ibi_df) == 1:
    
            loan.PaymentIncentives.ibi_sta_percent = ibi_df['ibi_pct'].iloc[0]
            loan.PaymentIncentives.ibi_sta_percent_deprbas_fed = 0
            loan.PaymentIncentives.ibi_sta_percent_deprbas_sta = 0
            loan.PaymentIncentives.ibi_sta_percent_maxvalue = ibi_df['max_incentive_usd'].iloc[0]
            loan.PaymentIncentives.ibi_sta_percent_tax_fed = 1
            loan.PaymentIncentives.ibi_sta_percent_tax_sta = 1
            
        elif len(ibi_df) >= 2:
            
            loan.PaymentIncentives.ibi_sta_percent = ibi_df['ibi_pct'].iloc[0]
            loan.PaymentIncentives.ibi_sta_percent_deprbas_fed = 0
            loan.PaymentIncentives.ibi_sta_percent_deprbas_sta = 0
            loan.PaymentIncentives.ibi_sta_percent_maxvalue = ibi_df['max_incentive_usd'].iloc[0]
            loan.PaymentIncentives.ibi_sta_percent_tax_fed = 1
            loan.PaymentIncentives.ibi_sta_percent_tax_sta = 1
            
            loan.PaymentIncentives.ibi_oth_percent = ibi_df['ibi_pct'].iloc[1]
            loan.PaymentIncentives.ibi_oth_percent_deprbas_fed = 0
            loan.PaymentIncentives.ibi_oth_percent_deprbas_sta = 0
            loan.PaymentIncentives.ibi_oth_percent_maxvalue = ibi_df['max_incentive_usd'].iloc[1]
            loan.PaymentIncentives.ibi_oth_percent_tax_fed = 1
            loan.PaymentIncentives.ibi_oth_percent_tax_sta = 1
            
        else:
            pass
        
    else:
        pass
    
    return loan


#%%
@decorators.fn_timer(logger = logger, tab_level = 2, prefix = '')
def calc_max_market_share(dataframe, max_market_share_df):

    in_cols = list(dataframe.columns)
    dataframe = dataframe.reset_index()
    
    dataframe['business_model'] = 'host_owned'
    dataframe['metric'] = 'payback_period'
    
    # Convert metric value to integer as a primary key, then bound within max market share ranges
    max_payback = max_market_share_df[max_market_share_df.metric == 'payback_period'].payback_period.max()
    min_payback = max_market_share_df[max_market_share_df.metric == 'payback_period'].payback_period.min()
    max_mbs = max_market_share_df[max_market_share_df.metric == 'percent_monthly_bill_savings'].payback_period.max()
    min_mbs = max_market_share_df[max_market_share_df.metric == 'percent_monthly_bill_savings'].payback_period.min()
    
    # copy the metric valeus to a new column to store an edited version
    payback_period_bounded = dataframe['payback_period'].values.copy()
    
    # where the metric value exceeds the corresponding max market curve bounds, set the value to the corresponding bound
    payback_period_bounded[np.where((dataframe.metric == 'payback_period') & (dataframe['payback_period'] < min_payback))] = min_payback
    payback_period_bounded[np.where((dataframe.metric == 'payback_period') & (dataframe['payback_period'] > max_payback))] = max_payback    
    payback_period_bounded[np.where((dataframe.metric == 'percent_monthly_bill_savings') & (dataframe['payback_period'] < min_mbs))] = min_mbs
    payback_period_bounded[np.where((dataframe.metric == 'percent_monthly_bill_savings') & (dataframe['payback_period'] > max_mbs))] = max_mbs
    dataframe['payback_period_bounded'] = np.round(payback_period_bounded.astype(float), 1)

    # scale and round to nearest int    
    dataframe['payback_period_as_factor'] = (dataframe['payback_period_bounded'] * 100).round().astype('int')
    # add a scaled key to the max_market_share dataframe too
    max_market_share_df['payback_period_as_factor'] = (max_market_share_df['payback_period'] * 100).round().astype('int')

    # Join the max_market_share table and dataframe in order to select the ultimate mms based on the metric value. 
    dataframe = pd.merge(dataframe, max_market_share_df[['sector_abbr', 'max_market_share', 'metric', 'payback_period_as_factor', 'business_model']], 
        how = 'left', on = ['sector_abbr', 'metric','payback_period_as_factor','business_model'])
    
    out_cols = in_cols + ['max_market_share', 'metric']    

    return dataframe[out_cols]