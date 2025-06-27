import os


"""directories."""
# DATA_DIR = os.path.join(PDIR, "data")
# OUTPUT_DIR = os.path.join(PDIR, "output")

PROJECT_DIR = "/projects/dwind/"
PROJECT_DATA_DIR = "/projects/dwind/agents"

# THEORETICAL_GENERATORS_NAME = os.path.join(
#     PROJECT_DATA_DIR,
#     SAMPLE_NAME,
#     f"agents_dwind_{RUN_SECTORS[0].lower()}.parquet"
# )

# PATH_RESULTS = os.path.join(OUTPUT_DIR, THEORETICAL_GENERATORS_NAME)

REV_DIR = "/projects/dwind/data/rev/"
# COST_DIR = "/projects/dwind/configs/costs/atb24"
# WIND_RESOURCE_DIR = "/datasets/WIND/conus/v1.0.0/wtk_led_conus_2018_hourly.h5"
# SOLAR_RESOURCE_DIR = "/datasets/NSRDB/v3/nsrdb_2012.h5"
REV_GENERATION_FILE_PATH = {
    "solar": "/projects/dwind/configs/rev/solar",
    "wind": "/projects/dwind/configs/rev/wind",
}

# os.path.join(DATA_DIR, "cambium_processed")
CAMBIUM_DATA_DIR = "/projects/dwind/data/cambium_processed"
# CAMBIUM_RAWDATA_DIR = os.path.join(DATA_DIR, "cambium_csvs")

# ANALYSIS_DIR = os.path.join(PDIR, "analysis")
# CAMBIUM_PROCESSED_DIR = os.path.join(ANALYSIS_DIR, "cambium")

# ATB
# PATH_COST = {
#     "wind": os.path.join(COST_DIR, "wind_price_FY17.csv"),
#     "solar": os.path.join(COST_DIR, "pv_price_atb20_mid.csv"),
# }

RETAIL_RATE_INPUT_TABLE = os.path.join(COST_DIR, "AEO23_Mid_Case_retail.csv")
WHOLESALE_RATE_INPUT_TABLE = os.path.join(COST_DIR, "AEO23_Mid_Case_wholesale.csv")
FINANCIAL_INPUTS_TABLE = os.path.join(COST_DIR, "ATB24_financing_baseline_2025.csv")
DEPREC_INPUTS_TABLE = os.path.join(COST_DIR, "ATB24_depc_factors.csv")

# wind - prices
WIND_PRICE_INPUT_TABLE = os.path.join(COST_DIR, "ATB24_wind_prices.csv")

# pv - prices
PV_PRICE_INPUT_TABLE = "pv_price_atb20_mid"
PV_PLUS_BATT_PRICE_INPUT_TABLE = "pv_plus_batt_prices_FY20_mid_pv_mid_batt"

# batt - prices
BATT_PRICE_INPUT_TABLE = "batt_prices_FY20_mid"


"""database connections."""
PG_USER = "dwind"
PG_PWD = "distributedwindfutures"

# GIS_PG_CON_STR = f"postgresql://{PG_USER}:{PG_PWD}@1lv11gispg02.nrel.gov:5432/dgen"
PARCELS_PG_CON_STR = f"postgresql://{PG_USER}:{PG_PWD}@1lv11gispg02.nrel.gov:5432/parcels"
ATLAS_PG_CON_STR = f"postgresql://{PG_USER}:{PG_PWD}@plv11dnpg01.nrel.gov:5432/dgen_db_fy17q3_dwind"
LIGHTBOX_PG_CON_STR = f"postgresql://{PG_USER}:{PG_PWD}@gds_publish.nrel.gov:5432/ref_lightbox"
MS_BG_CON_STR = f"postgresql://{PG_USER}:{PG_PWD}@gds_publish.nrel.gov:5432/ref_microsoft"
# SAGE_PG_CON_STR = "postgresql://dwindread:nABWLw8#VfeDYT@sage.hpc.nrel.gov:5432/dwind"
SAGE_PG_CON_STR = "postgresql://dwindread:nABWLw8#VfeDYT@sage.hpc.nrel.gov:5432/dgensfs"


"""flags."""
# SAVE_APP_PICKLES = True
# SAVE_TOTAL_PICKLE = True
# SAVE_TOTAL_SQL = False

# CREATE_AGENTS = False
# ARE YOU SURE YOU WANT THIS TO BE TRUE? DO NOT OVERWRITE THE NON _TEST LKUP
# CREATE_PARCEL_GENERATION_LKUP = False
# CREATE_PARCEL_TARIFF_LKUP = False
SIZE_SYSTEMS = True

# should generation profiles be grabbed from SQL based on parcel id (true)?
# or does a "generation_hourly" column exist (False)?
# PULL_GENERATION_FROM_SQL = False
# PULL_GENERATION_FROM_PKL = False
# should load profiles be grabbed from SQL based on parcel id (true)?
# or does a "consumption_hourly" column exist (False)?
# PULL_CONSUMPTION_FROM_SQL = True
# should BTM tariffs be grabbed from SQL? or does a "tariff_dict" column exist (False)?
# PULL_TARIFF_FROM_SQL = True


"""constants."""
# techs
TECHS = ["wind"]

WIND_TECH_INPUT_TABLE = "wind_tech_performance_1_reference"
WIND_DERATE_INPUT_TABLE = "wind_derate_sched_1_reference"
PV_TECH_INPUT_TABLE = "pv_tech_performance_defaultFY19"
BATT_TECH_INPUT_TABLE = "batt_tech_performance_SunLamp17"

# cambium/nem
CAMBIUM_VALUE = "cambium_grid_value"
# NEM_SCENARIO_TABLE = "nem_scenario_bau_2020"
# NEM_STATE_LIMITS_TABLE = "nem_state_limits_2020"

# scale 2020 solar tech configs to 2030 cfs
# Source: Solar Futures Scenario Assumptions (ATB) 2020 CF: 28% --> 2030 CF: 32%
# CF_2030_SCALE_FACTOR_SOLAR = 1.14

# normalize county roof sqft if sample is larger than this
# ROOF_SQFT_SCALER_THRESH = 2

# upper limit for scale factor of residential load based on floor area; anything above set to limit
# RES_LOAD_SCALE_FACTOR_LIMIT = 10

# minimum number of acres in a parcel to consider groundmount solar
# MIN_GROUND_MOUNT_ACRES = 5
# minimum wind turbine size to also consider groundmount solar
# MIN_GROUND_MOUNT_TURBINE_M = 5

# scalar factor applied to elements of 8760 gen array
GENERATION_SCALE_OFFSET = {"solar": 1000, "wind": 1000}
# CAPACITY_FACTOR_LIMIT = {"wind": 0.1, "solar": 0.1}

PYSAM_OUTPUT_VARIABLES = {
    "btm": [
        "adjusted_installed_cost",
        "discounted_payback",
        "effective_tax_rate",
        "first_cost",
        "lcoe_nom",
        "lcoe_real",
        "loan_amount",
        "npv",
        "payback",
        "present_value_oandm",
        "total_cost",
        "wacc",
    ],
    "fom": [
        "adjusted_installed_cost",
        "cost_debt_upfront",
        "cost_financing",
        "cost_installed",
        "cost_installedperwatt",
        "cost_prefinancing",
        "debt_fraction",
        "effective_tax_rate",
        "lcoe_nom",
        "lcoe_real",
        "nominal_discount_rate",
        "npv_annual_costs",
        "npv_capacity_revenue",
        "npv_curtailment_revenue",
        "npv_energy_market_revenue",
        "npv_energy_nom",
        "npv_energy_real",
        "present_value_oandm",
        "project_return_aftertax_irr",
        "project_return_aftertax_npv",
        "size_of_debt",
        "size_of_equity",
        "wacc",
    ],
}


"""technology definitions."""
# power_curve_lkup = {
#     1: [
#         0.0,
#         0.0,
#         0.022,
#         0.053,
#         0.104,
#         0.18,
#         0.286,
#         0.427,
#         0.607,
#         0.833,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     2: [
#         0.0,
#         0.0,
#         0.032,
#         0.076,
#         0.148,
#         0.256,
#         0.407,
#         0.608,
#         0.865,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     3: [
#         0.0,
#         0.0,
#         0.019,
#         0.045,
#         0.087,
#         0.151,
#         0.239,
#         0.358,
#         0.509,
#         0.698,
#         0.929,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     4: [
#         0.0,
#         0.0,
#         0.031,
#         0.074,
#         0.144,
#         0.25,
#         0.396,
#         0.592,
#         0.842,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     5: [
#         0.0,
#         0.0,
#         0.033,
#         0.078,
#         0.153,
#         0.264,
#         0.42,
#         0.627,
#         0.892,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     6: [
#         0.0,
#         0.0,
#         0.04,
#         0.094,
#         0.184,
#         0.318,
#         0.504,
#         0.753,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     7: [
#         0.0,
#         0.0,
#         0.038,
#         0.089,
#         0.174,
#         0.301,
#         0.478,
#         0.714,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
#     8: [
#         0.0,
#         0.0,
#         0.045,
#         0.107,
#         0.209,
#         0.362,
#         0.575,
#         0.858,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         1.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ],
# }

turbine_class_dict = {
    0: "none",
    2.5: "res",
    5.0: "res",
    10.0: "res",
    20.0: "res",
    50.0: "com",
    100.0: "com",
    250.0: "mid",
    500.0: "mid",
    750.0: "mid",
    1000.0: "large",
    1500.0: "large",
}

azimuth_direction_to_degree = {"E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270}


"""tariff information."""
# how to select tariff for BTM agents; "random": random choice of applicable tariffs,
# "bill_calculation":run full bill calculation and select tariff with lowest bill
# TARIFF_SELECT_OPTION = "random"

# NOTE: these are specific to 20200721 vintage of URDB rates
# res_tariffs = {
#     "AL": {"rate_id_alias": 17279, "rate_type_tou": True},  # Family Dwelling Service
#     "AR": {
#         "rate_id_alias": 16671,
#         "rate_type_tou": True,
#     },  # Optional Residential Time-Of-Use (RT) Single Phase
#     "AZ": {
#         "rate_id_alias": 15704,
#         "rate_type_tou": True,
#     },  # Residential Time of Use (Saver Choice) TOU-E
#     "CA": {
#         "rate_id_alias": 15747,
#         "rate_type_tou": True,
#     },  # E-1 -Residential Service Baseline Region P
#     "CO": {"rate_id_alias": 17078, "rate_type_tou": True},  # Residential Service (Schedule R)
#     "CT": {"rate_id_alias": 16678, "rate_type_tou": False},  # Rate 1 - Residential Electric Service
#     "DC": {"rate_id_alias": 16806, "rate_type_tou": True},  # Residential - Schedule R
#     "DE": {"rate_id_alias": 11569, "rate_type_tou": True},  # Residential Service
#     "FL": {"rate_id_alias": 16986, "rate_type_tou": False},  # RS-1 Residential Service
#     "GA": {"rate_id_alias": 16649, "rate_type_tou": True},  # SCHEDULE R-22 RESIDENTIAL SERVICE
#     "IA": {"rate_id_alias": 11693, "rate_type_tou": True},  # Optional Residential Service
#     "ID": {"rate_id_alias": 16227, "rate_type_tou": False},  # Schedule 1: Residential Rates
#     "IL": {"rate_id_alias": 16045, "rate_type_tou": True},  # DS-1 Residential Zone 1
#     "IN": {"rate_id_alias": 15491, "rate_type_tou": False},  # RS - Residential Service
#     "KS": {"rate_id_alias": 8178, "rate_type_tou": True},  # M System Residential Service
#     "KY": {"rate_id_alias": 16566, "rate_type_tou": False},  # Residential Service
#     "LA": {
#         "rate_id_alias": 16352,
#         "rate_type_tou": True,
#     },  # Residential and Farm Service - Single Phase (RS-L)
#     "MA": {"rate_id_alias": 15953, "rate_type_tou": False},  # Greater Boston Residential R-1 (A1)
#     "MD": {"rate_id_alias": 14779, "rate_type_tou": False},  # Residential Service (R)
#     "ME": {
#         "rate_id_alias": 15984,
#         "rate_type_tou": False,
#     },  # A Residential Standard Offer Service (Bundled)
#     "MI": {
#         "rate_id_alias": 16265,
#         "rate_type_tou": True,
#     },  # Residential Service - Secondary (Rate RS)
#     "MN": {
#         "rate_id_alias": 15556,
#         "rate_type_tou": True,
#     },  # Residential Service - Overhead Standard (A01)
#     "MO": {"rate_id_alias": 17207, "rate_type_tou": True},  # 1(M) Residential Service Rate
#     "MS": {
#         "rate_id_alias": 16788,
#         "rate_type_tou": False,
#     },  # Residential Service Single Phase (RS-38C)
#     "MT": {"rate_id_alias": 5216, "rate_type_tou": False},  # Single Phase
#     "NC": {
#         "rate_id_alias": 16938,
#         "rate_type_tou": True,
#     },  # Residential Service (RES-41) Single Phase
#     "ND": {"rate_id_alias": 14016, "rate_type_tou": True},  # Residential Service Rate 10
#     "NE": {"rate_id_alias": 13817, "rate_type_tou": True},  # Residential Service
#     "NH": {"rate_id_alias": 16605, "rate_type_tou": False},  # Residential Service
#     "NJ": {"rate_id_alias": 16229, "rate_type_tou": True},  # RS - Residential Service
#     "NM": {"rate_id_alias": 8692, "rate_type_tou": True},  # 1A (Residential Service)
#     "NV": {"rate_id_alias": 16701, "rate_type_tou": False},  # D-1 (Residential Service)
#     "NY": {"rate_id_alias": 16902, "rate_type_tou": False},  # SC1- Zone A
#     "OH": {"rate_id_alias": 16892, "rate_type_tou": True},  # RS (Residential Service)
#     "OK": {"rate_id_alias": 15258, "rate_type_tou": True},  # Residential Service (R-1)
#     "OR": {
#         "rate_id_alias": 15847,
#         "rate_type_tou": False,
#     },  # Schedule 4 - Residential (Single Phase)
#     "PA": {"rate_id_alias": 17237, "rate_type_tou": False},  # RS (Residential Service)
#     "RI": {"rate_id_alias": 16598, "rate_type_tou": False},  # A-16 (Residential Service)
#     "SC": {"rate_id_alias": 15744, "rate_type_tou": False},  # Residential - RS (SC)
#     "SD": {"rate_id_alias": 1216, "rate_type_tou": False},  # Town and Rural Residential Rate
#     "TN": {"rate_id_alias": 15149, "rate_type_tou": False},  # Residential Electric Service
#     "TX": {"rate_id_alias": 16710, "rate_type_tou": True},  # Residential Service - Time Of Day
#     "UT": {
#         "rate_id_alias": 15847,
#         "rate_type_tou": False,
#     },  # Schedule 4 - Residential (Single Phase)
#     "VA": {"rate_id_alias": 17067, "rate_type_tou": True},  # Residential Schedule 1
#     "VT": {"rate_id_alias": 16544, "rate_type_tou": False},  # Rate 01 Residential Service
#     "WA": {
#         "rate_id_alias": 16305,
#         "rate_type_tou": False,
#     },  # 10 (Residential and Farm Primary General Service)
#     "WI": {"rate_id_alias": 15543, "rate_type_tou": False},  # Residential Rg-1
#     "WV": {"rate_id_alias": 15515, "rate_type_tou": False},  # Residential Service A
#     "WY": {
#         "rate_id_alias": 15847,
#         "rate_type_tou": False,
#     },  # Schedule 4 - Residential (Single Phase)
# }

# com_tariffs = {
#     "AL": {
#         "rate_id_alias": 15494,
#         "rate_type_tou": True,
#     },  # BTA - BUSINESS TIME ADVANTAGE (OPTIONAL) - Primary
#     "AR": {"rate_id_alias": 16674, "rate_type_tou": False},  # Small General Service (SGS)
#     "AZ": {
#         "rate_id_alias": 10742,
#         "rate_type_tou": True,
#     },  # LGS-TOU- N - Large General Service Time-of-Use
#     "CA": {
#         "rate_id_alias": 17057,
#         "rate_type_tou": True,
#     },  # A-10 Medium General Demand Service (Secondary Voltage)
#     "CO": {"rate_id_alias": 17102, "rate_type_tou": True},  # Commercial Service (Schedule C)
#     "CT": {
#         "rate_id_alias": 16684,
#         "rate_type_tou": False,
#     },  # Rate 35 Intermediate General Electric Service
#     "DC": {"rate_id_alias": 15336, "rate_type_tou": True},  # General Service (Schedule GS)
#     "DE": {"rate_id_alias": 1199, "rate_type_tou": False},  # Schedule LC-P Large Commercial Primary
#     "FL": {"rate_id_alias": 13790, "rate_type_tou": True},  # SDTR-1 (Option A)
#     "GA": {
#         "rate_id_alias": 1905,
#         "rate_type_tou": True,
#     },  # SCHEDULE TOU-MB-4 TIME OF USE - MULTIPLE BUSINESS
#     "IA": {"rate_id_alias": 11705, "rate_type_tou": True},  # Three Phase Farm
#     "ID": {
#         "rate_id_alias": 14782,
#         "rate_type_tou": False,
#     },  # Large General Service (3 Phase)-Schedule 21
#     "IL": {"rate_id_alias": 1567, "rate_type_tou": False},  # General Service Three Phase standard
#     "IN": {"rate_id_alias": 15492, "rate_type_tou": False},  # CS - Commercial Service
#     "KS": {"rate_id_alias": 14736, "rate_type_tou": False},  # Generation Substitution Service
#     "KY": {"rate_id_alias": 17179, "rate_type_tou": True},  # General Service (Single Phase)
#     "LA": {"rate_id_alias": 17220, "rate_type_tou": False},  # Large General Service (LGS-L)
#     "MA": {
#         "rate_id_alias": 16005,
#         "rate_type_tou": False,
#     },  # Western Massachusetts Primary General Service G-2
#     "MD": {"rate_id_alias": 2659, "rate_type_tou": False},  # Commercial
#     "ME": {"rate_id_alias": 16125, "rate_type_tou": False},  # General Service Rate
#     "MI": {"rate_id_alias": 5355, "rate_type_tou": False},  # Large Power Service (LP4)
#     "MN": {
#         "rate_id_alias": 15566,
#         "rate_type_tou": False,
#     },  # General Service (A14) Secondary Voltage
#     "MO": {
#         "rate_id_alias": 17208,
#         "rate_type_tou": True,
#     },  # 2(M) Small General Service - Single phase
#     "MS": {
#         "rate_id_alias": 13427,
#         "rate_type_tou": True,
#     },  # General Service - Low Voltage Single-Phase (GS-LVS-14)
#     "MT": {"rate_id_alias": 10707, "rate_type_tou": False},  # Three Phase
#     "NC": {"rate_id_alias": 16947, "rate_type_tou": False},  # General Service (GS-41)
#     "ND": {
#         "rate_id_alias": 14035,
#         "rate_type_tou": False,
#     },  # Small General Electric Service rate 20 (Demand Metered; Non-Demand)
#     "NE": {"rate_id_alias": 13818, "rate_type_tou": True},  # General Service Single-Phase
#     "NH": {"rate_id_alias": 16620, "rate_type_tou": False},  # GV Commercial and Industrial Service
#     "NJ": {"rate_id_alias": 17095, "rate_type_tou": True},  # AGS Secondary- BGS-RSCP
#     "NM": {"rate_id_alias": 15769, "rate_type_tou": True},  # 2A (Small Power Service)
#     "NV": {"rate_id_alias": 13724, "rate_type_tou": True},  # OGS-2-TOU
#     "NY": {
#         "rate_id_alias": 15940,
#         "rate_type_tou": False,
#     },  # SC-9 - General Large High Tension Service [Westchester]
#     "OH": {"rate_id_alias": 16873, "rate_type_tou": True},  # GS (General Service-Secondary)
#     "OK": {"rate_id_alias": 17144, "rate_type_tou": True},  # GS-TOU (General Service Time-Of-Use)
#     "OR": {
#         "rate_id_alias": 15829,
#         "rate_type_tou": False,
#     },  # Small Non-Residential Direct Access Service, Single Phase (Rate 532)
#     "PA": {"rate_id_alias": 7066, "rate_type_tou": False},  # Large Power 2 (LP2)
#     "RI": {"rate_id_alias": 16600, "rate_type_tou": False},  # G-02 (General C & I Rate)
#     "SC": {"rate_id_alias": 16207, "rate_type_tou": False},  # 3 (Municipal  Power Service)
#     "SD": {"rate_id_alias": 3650, "rate_type_tou": False},  # Small Commercial
#     "TN": {"rate_id_alias": 15154, "rate_type_tou": False},  # Medium General Service (Primary)
#     "TX": {"rate_id_alias": 6001, "rate_type_tou": False},  # Medium Non-Residential LSP POLR
#     "UT": {"rate_id_alias": 3478, "rate_type_tou": False},  # SCHEDULE GS - 3 Phase General Service
#     "VA": {"rate_id_alias": 16557, "rate_type_tou": True},  # Small General Service Schedule 5
#     "VT": {"rate_id_alias": 16543, "rate_type_tou": False},  # Rate 06: General Service
#     "WA": {
#         "rate_id_alias": 16306,
#         "rate_type_tou": False,
#     },  # 40 (Large Demand General Service over 3MW - Primary)
#     "WI": {
#         "rate_id_alias": 6620,
#         "rate_type_tou": True,
#     },  # Cg-7 General Service Time-of-Day (Primary)
#     "WV": {"rate_id_alias": 15518, "rate_type_tou": False},  # General Service C
#     "WY": {"rate_id_alias": 3878, "rate_type_tou": False},  # General Service (GS)-Three phase
# }

# map industrial tariffs based on census division
# ind_tariffs = {
#     "SA": {
#         "rate_id_alias": 16657,
#         "rate_type_tou": True,
#     },  # Georgia Power Co, Schedule TOU-GSD-10 Time Of Use - General Service Demand
#     "WSC": {
#         "rate_id_alias": 15919,
#         "rate_type_tou": False,
#     },  # Southwestern Public Service Co (Texas), Large General Service - Inside City Limits 115 KV
#     "PAC": {
#         "rate_id_alias": 15864,
#         "rate_type_tou": True,
#     },  # PacifiCorp (Oregon), Schedule 47 - Secondary (Less than 4000 kW)
#     "MA": {
#         "rate_id_alias": 16525,
#         "rate_type_tou": True,
#     },  # New York State Elec & Gas Corp, All Regions - SERVICE CLASSIFICATION NO. 7-1 Large General Service TOU - Secondary -ESCO
#     "MTN": {
#         "rate_id_alias": 17101,
#         "rate_type_tou": True,
#     },  # Public Service Co of Colorado, Secondary General Service (Schedule SG)
#     "ENC": {
#         "rate_id_alias": 15526,
#         "rate_type_tou": True,
#     },  # Wisconsin Power & Light Co, Industrial Power Cp-1 (Secondary)
#     "NE": {
#         "rate_id_alias": 16635,
#         "rate_type_tou": True,
#     },  # Delmarva Power, General Service - Primary
#     "ESC": {
#         "rate_id_alias": 15490,
#         "rate_type_tou": True,
#     },  # Alabama Power Co, LPM - LIGHT AND POWER SERVICE - MEDIUM
#     "WNC": {
#         "rate_id_alias": 6642,
#         "rate_type_tou": True,
#     },  # Northern States Power Co - Wisconsin, Cg-9.1 Large General Time-of-Day Primary Mandatory Customers
# }


"""cambium."""
# CAMBIUM_VARIABLE_LEVEL = ["cambium_co2_rate_lrmer"]

# # for levelizing lrmer from cambium
# CAMBIUM_DISCOUNT_RATES = {
#     "cambium_energy_value": 0.064,
#     "cambium_capacity_value": 0.064,
#     "cambium_co2_rate_lrmer": 0.064,
#     "cambium_grid_value": 0.064,
#     "cambium_portfolio_value": 0.064,
# }

# # for levelizing lrmer from cambium
# CAMBIUM_PERIOD_LEVEL = {
#     "cambium_energy_value": 25,
#     "cambium_capacity_value": 25,
#     "cambium_co2_rate_lrmer": 25,
#     "cambium_grid_value": 25,
#     "cambium_portfolio_value": 25,
# }


"""environmental justice."""
# EJ_PG_USER = "jlockshin"
# EJ_PG_HOST = "gds_edit.nrel.gov"
# EJ_PG_DB = "parcels"
# EJ_PG_SCHEMA = "dwfs"

# EJ_IDX_TABLE = "epa_ej_indexes"
# EJ_REPLICA_TABLE = "nrel_seeds_ii_replica"
# EJ_INPUT_PARCELS = f"random_sample_parcels_{SAMPLE_NAME}"
# EJ_OUTPUT_TABLE = f"dwfs.underserved_parcels_{SAMPLE_NAME}"

# EJ Index Assumptions (EPA)
"""
EJ Indexes (with Brownfields)
    - P_LDPNT_D2: Percentile for EJ Index for % pre-1960 housing (lead paint indicator)
    - P_DSLPM_D2: Percentile for EJ Index for Diesel particulate matter level in air
    - P_CANCR_D2: Percentile for EJ Index for Air toxics cancer risk
    - P_RESP_D2: Percentile for EJ Index for Air toxics respiratory hazard index
    - P_PTRAF_D2: Percentile for EJ Index for Traffic proximity and volume
    - P_PWDIS_D2: Percentile for EJ Index for Indicator for major direct dischargers to water
    - P_PNPL_D2: Percentile for EJ Index for Proximity to National Priorities List (NPL) sites
    - P_PRMP_D2: Percentile for EJ Index for Proximity to Risk Management Plan (RMP) facilities
    - P_PTSDF_D2: Percentile for EJ Index for Proximity to Treatment Storage and Disposal (TSDF) facilities
    - P_OZONE_D2: Percentile for EJ Index for Ozone level in air
    - P_PM25_D2: Percentile for EJ Index for PM2.5 level in air
"""
# EJ_P_LDPNT_D2_THRESH = 75
# EJ_P_DSLPM_D2_THRESH = 75
# EJ_P_CANCR_D2_THRESH = 75
# EJ_P_RESP_D2_THRESH = 75
# EJ_P_PTRAF_D2_THRESH = 75
# EJ_P_PWDIS_D2_THRESH = 75
# EJ_P_PNPL_D2_THRESH = 75
# EJ_P_PRMP_D2_THRESH = 75
# EJ_P_PTSDF_D2_THRESH = 75
# EJ_P_OZONE_D2_THRESH = 75
# EJ_P_PM25_D2_THRESH = 75

# EJ_BROWNFIELDS = True
# EJ_BROWNDIEIDLS_THRESH = 0

# EJ REPLICA Assumptions (NREL)
"""
REPLICA
    - very_low_mf_own_hh: Very Low Income (0-30% AMI) - Multi-Family - Owner-Occupied - Household Count
    - very_low_mf_rent_hh: Very Low Income (0-30% AMI) - Multi-Family - Renter-Occupied - Household Count
    - very_low_sf_own_hh: Very Low Income (0-30% AMI) - Single-Family - Owner-Occupied - Household Count
    - very_low_sf_rent_hh: Very Low Income (0-30% AMI) - Single-Family - Renter-Occupied - Household Count
    - low_mf_own_hh: Low Income (30-50% AMI) - Multi-Family - Owner-Occupied - Household Count
    - low_mf_rent_hh: Low Income (30-50% AMI) - Multi-Family - Renter-Occupied - Household Count
    - low_sf_own_hh: Low Income (30-50% AMI) - Single-Family - Owner-Occupied - Household Count
    - low_sf_rent_hh: Low Income (30-50% AMI) - Single-Family - Renter-Occupied - Household Count
    - hu_own: Total number of owner occupied housing units
    - hu_rent: Total number of renter occupied housing units
    - very_low_mf_own_elep_hh: Very Low Income (0-30% AMI) - Multi-Family - Owner-Occupied - Average Household Electricity Expenditures ($/month)
    - very_low_mf_rent_elep_hh: Very Low Income (0-30% AMI) - Multi-Family - Renter-Occupied - Average Household Electricity Expenditures ($/month)
    - very_low_sf_own_elep_hh: Very Low Income (0-30% AMI) - Single-Family - Owner-Occupied - Average Household Electricity Expenditures ($/month)
    - very_low_sf_rent_elep_hh: Very Low Income (0-30% AMI) - Single-Family - Renter-Occupied - Average Household Electricity Expenditures ($/month)
    - low_mf_own_elep_hh: Low Income (30-50% AMI) - Multi-Family - Owner-Occupied - Average Household Electricity Expenditures ($/month)
    - low_mf_rent_elep_hh: Low Income (30-50% AMI) - Multi-Family - Renter-Occupied - Average Household Electricity Expenditures ($/month)
    - low_sf_own_elep_hh: Low Income (30-50% AMI) - Single-Family - Owner-Occupied - Average Household Electricity Expenditures ($/month)
    - low_sf_rent_elep_hh: Low Income (30-50% AMI) - Single-Family - Renter-Occupied - Average Household Electricity Expenditures ($/month)

    - avg_monthly_bill_dlrs: Average Monthly Bill
    - hh_med_income : Median household income
    - hh_gini_index: Household GINI Index of Income Inequality (0 = complete equality, 1 = complete inequality). US mean (2017): 0.390
    - pop25_no_high_school: Total population with less than a high school diploma (Population 25 years and over)
    - pop_total: Total population
    - pct_eli_hh - Percent Extremely Low Income  (87% null - don"t use this column)
    - lihtc_qualified - Low Income Tax Credit Qualification - where True
"""
# Very Low Income and Low Income
# EJ_REPLICA_VLI = True
# EJ_REPLICA_LI = True

# Ownership
# EJ_REPLICA_RENT = True
# EJ_REPLICA_OWN = True

# Single-Family vs Multi-Family
# EJ_REPLICA_MF = True
# EJ_REPLICA_SF = True

# Incorporaate Electricity Costs as a Fraction of Income
# EJ_REPLICA_ELEC_BURDEN = True

# Other EJ Parameters
# EJ_REPLICA_EDUCATION = True
# EJ_REPLICA_GINI = True
# EJ_REPLICA_LIHTC = True

"""others (no longer necessary)."""
# SITING_SCENARIO_NAME = "baseline"
# SETBACK_FACTOR = 1.10

# SQL_GENERATOR_LKUP_SETTINGS = {
#     "schema": "dwfs_model_results",
#     "owner": "diffusion-writers",
#     "if_exists": "replace",
#     "append_transformations": False,
# }

# SITING_SCENARIO_NAME = "baseline"  # ["baseline", "relaxed_res", "relaxed"]

SQL_GENERATOR_LKUP_TABLE = "diffusion_resource_wind.dwfs_parcel_resource_lkup_test"
# SQL_TARIFF_LKUP_TABLE = "diffusion_resource_wind.dwfs_tariff_lkup_1mill"

# SAMPLE_PARCEL_TABLE_UNPROCESSED = f"dwfs.random_sample_parcels_{SAMPLE_NAME}"
# SAMPLE_PARCEL_TABLE_PREPROCESSED = f"dwfs.processed_parcels_{SAMPLE_NAME}"
# PARCEL_TO_BLOCK_LKUP_TABLE = f"dwfs.lightbox_parcel_block_lkup_{SAMPLE_NAME}"
# PARCEL_TO_BLDG_GEOM_LKUP_TABLE = f"dwfs.lkup_parcels_to_bldgs_{SAMPLE_NAME}_with_geoms"

# SAVE_BASE_NAME_TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

WIND_CONFIG_DIR = os.path.join(os.curdir, "archive", "wind_config_files")
wind_config_dict = {
    "res": os.path.join(WIND_CONFIG_DIR, "wind_res_2030.json"),
    "com": os.path.join(WIND_CONFIG_DIR, "wind_com_2030.json"),
    "mid": os.path.join(WIND_CONFIG_DIR, "wind_mid_2030.json"),
    "large": os.path.join(WIND_CONFIG_DIR, "wind_large_2030.json"),
}

# SOLAR_CONFIG_DIR = os.path.join(os.curdir, "archive", "solar_config_files")
solar_config_dict = {
    "90_15": os.path.join(SOLAR_CONFIG_DIR, "solar_90_15_2020.json"),
    "90_25": os.path.join(SOLAR_CONFIG_DIR, "solar_90_25_2020.json"),
    "90_35": os.path.join(SOLAR_CONFIG_DIR, "solar_90_35_2020.json"),
    "90_45": os.path.join(SOLAR_CONFIG_DIR, "solar_90_45_2020.json"),
    "90_55": os.path.join(SOLAR_CONFIG_DIR, "solar_90_55_2020.json"),
    "135_15": os.path.join(SOLAR_CONFIG_DIR, "solar_135_15_2020.json"),
    "135_25": os.path.join(SOLAR_CONFIG_DIR, "solar_135_25_2020.json"),
    "135_35": os.path.join(SOLAR_CONFIG_DIR, "solar_135_35_2020.json"),
    "135_45": os.path.join(SOLAR_CONFIG_DIR, "solar_135_45_2020.json"),
    "135_55": os.path.join(SOLAR_CONFIG_DIR, "solar_135_55_2020.json"),
    "180_15": os.path.join(SOLAR_CONFIG_DIR, "solar_180_15_2020.json"),
    "180_25": os.path.join(SOLAR_CONFIG_DIR, "solar_180_25_2020.json"),
    "180_35": os.path.join(SOLAR_CONFIG_DIR, "solar_180_35_2020.json"),
    "180_45": os.path.join(SOLAR_CONFIG_DIR, "solar_180_45_2020.json"),
    "180_55": os.path.join(SOLAR_CONFIG_DIR, "solar_180_55_2020.json"),
    "225_15": os.path.join(SOLAR_CONFIG_DIR, "solar_225_15_2020.json"),
    "225_25": os.path.join(SOLAR_CONFIG_DIR, "solar_225_25_2020.json"),
    "225_35": os.path.join(SOLAR_CONFIG_DIR, "solar_225_35_2020.json"),
    "225_45": os.path.join(SOLAR_CONFIG_DIR, "solar_225_45_2020.json"),
    "225_55": os.path.join(SOLAR_CONFIG_DIR, "solar_225_55_2020.json"),
    "270_15": os.path.join(SOLAR_CONFIG_DIR, "solar_270_15_2020.json"),
    "270_25": os.path.join(SOLAR_CONFIG_DIR, "solar_270_25_2020.json"),
    "270_35": os.path.join(SOLAR_CONFIG_DIR, "solar_270_35_2020.json"),
    "270_45": os.path.join(SOLAR_CONFIG_DIR, "solar_270_45_2020.json"),
    "270_55": os.path.join(SOLAR_CONFIG_DIR, "solar_270_55_2020.json"),
}

# RUN_REV_IN_MODEL = False

# pct frequency to output stautus during multiprocess execution (i.e 0.5 would be two statuses)
VERBOSITY = 0.1
# PARCEL_QUERY_LIMIT = None
# WORKERS = 1
THREAD_WORKERS = 1
# NODES = 1
CORES = 104

# SETBACK_FACTOR = 1.10
# ALLOW_MULTIPLE_TURBINES = False
# MAX_TURBINE_COUNT = 3
# not sure how best to handle this, limits turbine sizes eligible to have multiple instances
# LIMIT_SIZE_MULTIPLE_TURBINES = [1500]
# ~12 [kW/acre]. Original = (3 MW/km2) from Mooney et al, unpublished. "Exploring the Technical Potential Opportunities of Wind on Parceled Lands in CO, MN, and NY." NREL. https://app.box.com/s/nescj4obrq72o56jhmop0tnthrmw0u1y.
# POWER_DENSITY_LIMIT = 3000 / 247.105
SYS_SIZE_TARGET_NO_NEM = 0.5
SYS_OVERSIZE_LIMIT_NO_NEM = 1.0

# EXCLUDE_SLOPE = True
# EXCLUDE_SLOPE_20_PCT = True
# EXCLUDE_SLOPE_05_PCT = False

# PARCELS_MINUS_SLOPE_20_PCT_TABLE = f"dwfs.parcels_minus_exclusions_{SAMPLE_NAME}"
# PARCELS_MINUS_SLOPE_05_PCT_TABLE = f"dwfs.parcels_minus_exclusions_{SAMPLE_NAME}_solar"
# PARCELS_MINUS_EXCLUSIONS_NO_SLOPE_TABLE = f"dwfs.parcels_minus_exclusions_{SAMPLE_NAME}_no_slope"

# PARCELS_WITH_TURBINE_AREA_WITH_SLOPE_TABLE = f"dwfs.parcels_with_turbine_area_{SAMPLE_NAME}"
# PARCELS_WITH_TURBINE_AREA_NO_SLOPE_TABLE = f"dwfs.parcels_with_turbine_area_{SAMPLE_NAME}_no_slope"

# PARCELS_FINAL_PREPROCESS_TABLE = f"dwfs.processed_parcels_{SAMPLE_NAME}"
# PARCELS_FINAL_PREPROCESS_NO_SLOPE_TABLE = f"dwfs.processed_parcels_{SAMPLE_NAME}_no_slope"

# SLOPE_EXCLUSION_WIND_RASTER = "/projects/dwind/dwfs_data/exclusions/srtm_slope_gt_20_pct_4326.tif"
# SLOPE_EXCLUSION_SOLAR_RASTER = "/projects/dwind/dwfs_data/exclusions/srtm_slope_gt_05_pct_4326.tif"

# CLIPPED_SLOPE_RASTER_DIR = "/projects/dwind/dwfs_data/exclusions/clipped_rasters"

# USE_CODE_STD_DESC_LPS_EXCLUSIONS = [
#     "TRANSPORTATION",
#     "NATURAL RESOURCES",
#     "CEMETERY (EXEMPT)",
#     "FOREST (PARK; RESERVE; RECREATION, CONSERVATION)",
#     "MARINE FACILITY/BOAT REPAIRS (SMALL CRAFT OR SAILBOAT)",
#     "CULTURAL, HISTORICAL (MONUMENTS; HOMES; MUSEUMS; OTHER)",
#     "WILDLIFE (REFUGE)",
#     "OUTDOOR RECREATION: BEACH, MOUNTAIN, DESERT",
#     "WATER AREA (LAKES; RIVER; SHORE)-VACANT LAND",
#     "WASTE LAND, MARSH, SWAMP, SUBMERGED-VACANT LAND",
#     "IRRIGATION, FLOOD CONTROL",
#     "PUBLIC SWIMMING POOL",
#     "HISTORICAL-PRIVATE (GENERAL)",
#     "CHEMICAL",
#     "RESERVOIR, WATER SUPPLY",
#     "FISH CAMPS, GAME CLUB, TARGET SHOOTING",
#     "BOAT SLIPS, MARINA, YACHT CLUB (RECREATION/PLEASURE), BOAT LANDING",
#     "RAIL (RIGHT-OF-WAY & TRACK)",
#     "ZOO",
#     "ROADS, STREETS, BRIDGES",
#     "RECREATIONAL VEHICLES / TRAVEL TRAILERS",
#     "RAILROAD & RELATED",
#     "ROAD (RIGHT-OF-WAY)",
#     "TIMBERLAND, FOREST, TREES (AGRICULTURAL)",
#     "AIRPORT & RELATED",
#     "PARK, PLAYGROUND, PICNIC AREA",
#     "FEDERAL PROPERTY (EXEMPT)",
#     "MILITARY (OFFICE; BASE; POST; PORT; RESERVE; WEAPON RANGE; TEST SITES)",
#     "PRIVATE PRESERVE, OPEN SPACE-VACANT LAND (FOREST LAND, CONSERVATION)",
#     "WATERCRAFT (SHIPS, BOATS, PWCS, ETC.)",
# ]

# OPEN_EI_API_KEY = "0B1qqoXeQfbLsJIzIpZrMsbOLfXh0Zch34nnA5xH"  # signup here https://openei.org/services/api/signup/
# OPEN_EI_API_EMAIL = "thomas.bowen@nrel.gov"

# SAMPLING_PG_USER = "jlockshin"
# SAMPLING_PG_HOST = "gds_edit.nrel.gov"
# SAMPLING_PG_DB = "parcels"
# SAMPLING_PG_SCHEMA = "dwfs"

# PARCEL_SAMPLING_TABLE = f"dwfs.random_sample_parcels_{SAMPLE_NAME}"
# PARCEL_TO_BLDG_LKUP = "bldg_lkup.lkup_parcels_to_bldgs"
# PARCEL_TO_BLDG_LKUP_SAMPLE_ONLY = f"dwfs.lkup_parcels_to_bldgs_{SAMPLE_NAME}_sample_only"
# PARCELS_MINUS_BLDGS_TABLE = f"dwfs.lkup_parcels_to_bldgs_{SAMPLE_NAME}_with_geoms"
# DUPLICATE_PARCEL_GEOMS_TABLE = f"dwfs.duplicate_parcel_geoms_{SAMPLE_NAME}"

SITING_INPUTS = {
    "solar": {
        "capacity_density_kw_per_sqft": 0.0033444,  # 36 MW/km^2 to kW/ft^2
        "max_btm_size_kw": 5000.0,
        "min_fom_size_kw": 500.0,
        "max_fom_size_kw": 10000.0,
    },
    "wind": {
        "canopy_pct_requiring_clearance": 10.0,
        "canopy_clearance_static_adder_m": 12,
        "required_parcel_size_cap_acres": 1e6,
        "blade_height_setback_factor": 1.10,
        "blade_height_setback_factor_res": 1.10,
        "max_btm_size_kw": 5000.0,
        "min_fom_size_kw": 500.0,
        "max_fom_size_kw": 10000.0,
    },
}

# WEIGHT_SAMPLING_TABLE = f"dwfs.random_sample_parcels_{SAMPLE_NAME}"
# WEIGHT_PCT_NULL = 1.05
# WEIGHT_COL = "use_code_std_ctgr_desc_lps"
# WEIGHT_COL_FILE = os.path.join(DATA_DIR, f"{WEIGHT_COL}_state_count.csv")
# WEIGHT_SAVE_TO_FILE = True
# WEIGHT_OUTPUT_FILE = f"lkup_gid_to_weights_{SAMPLE_NAME}.csv"
# WEIGHT_SAVE_TO_DB = False
# WEIGHT_OUTPUT_TABLE = f"dwfs.lkup_gid_to_weights_{SAMPLE_NAME}"

# AGRICULTURAL_PARCEL_LKUP_TABLE = f"dwfs.lkup_parcels_to_agr_{SAMPLE_NAME}"
