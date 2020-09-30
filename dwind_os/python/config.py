import os
import multiprocessing

#==============================================================================
#   get postgres connection parameters
#==============================================================================
# get the path of the current file
model_path = os.path.dirname(os.path.abspath(__file__))

# set the name of the pg_params_file
pg_params_file = 'pg_params_atlas.json'

#==============================================================================
#   model start year
#==============================================================================
start_year = 2014

#==============================================================================
#   set number of parallel processes to run postgres queries (this is ignored if parallelize = F)
#==============================================================================
pg_procs = 2

#==============================================================================
#   local cores
#==============================================================================
local_cores = multiprocessing.cpu_count()//2

#==============================================================================
#  Should the output schema be deleted after the model run
#==============================================================================
delete_output_schema = True

#==============================================================================
#  Set switch to determine if model should output ReEDS data (datfunc.aggregate_outputs_solar)
#==============================================================================
dynamic_system_sizing = True

#==============================================================================
#  Runtime Tests
#==============================================================================
NULL_COLUMN_EXCEPTIONS = ['state_incentives', 'pct_state_incentives', 'batt_dispatch_profile', 'export_tariff_results', 'inverter_lifetime_yrs']

CHANGED_DTYPES_EXCEPTIONS = []
MISSING_COLUMN_EXCEPTIONS = []

#==============================================================================
#  Detailed Output
#==============================================================================
VERBOSE = False

#==============================================================================
#  Define CSVS
#==============================================================================

cwd = os.getcwd() #should be /python
pdir = os.path.abspath('..') #should be /dgen or whatever it is called
INSTALLED_CAPACITY_BY_STATE = os.path.join(pdir, 'input_data','installed_capacity_mw_by_state_sector.csv')