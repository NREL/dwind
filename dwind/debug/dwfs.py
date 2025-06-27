import numpy as np
import pandas as pd 


# f = '/projects/dwind/archive/dwfs_2022/fom_baseline_2022/dwfs/output/run_FOM_baseline_2022.pkl'
# f_apr = '/projects/dwind/archive/dwfs_2022/fom_baseline_2022/dwfs/output/run_FOM_baseline_2022_13042022.pkl'
# f_jan = '/projects/dwind/archive/dwfs_2022/fom_baseline_2022/dwfs/output/run_FOM_baseline_2022_07012022.pkl'

# df = pd.read_pickle(f)
# df_apr = pd.read_pickle(f_apr)
# df_jan = pd.read_pickle(f_jan)

# cost = np.mean(df.wind_breakeven_cost_fom.values)
# cost_apr = np.mean(df_apr.wind_breakeven_cost_fom.values)
# cost_jan = np.mean(df_jan.wind_breakeven_cost_fom.values)

# print(cost)
# print(cost_apr)
# print(cost_jan)

f_btm = '/projects/dwind/archive/dwfs_2022/btm_baseline_2022/dwfs/output/run_BTM_baseline_2022.pkl'
f_btm_jan = '/projects/dwind/archive/dwfs_2022/btm_baseline_2022/dwfs/output/run_BTM_baseline_2022_07012022.pkl'

df_btm = pd.read_pickle(f_btm)
df_btm_jan = pd.read_pickle(f_btm_jan)

cost_btm = np.nanmean(df_btm.wind_breakeven_cost_btm.values)
cost_btm_jan = np.nanmean(df_btm_jan.wind_breakeven_cost_btm.values)

print(cost_btm)
print(cost_btm_jan)
