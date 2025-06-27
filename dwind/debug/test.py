import pandas as pd
import numpy as np

# f = '/projects/dwind/runs_2023/minnesota/chunk_files_minnesota_fom_baseline_2022/minnesota_fom_baseline_2022_0.pkl'
# f = '/projects/dwind/runs_2023/indiana/btm/chunk_files_indiana_btm_baseline_2022/indiana_btm_baseline_2022_0.pkl'
# f = '/projects/dwind/archive/dwfs_2022/fom_baseline_2022/dwfs/output/run_FOM_baseline_2022.pkl'
f = '/projects/dwind/runs_2023/priority6/btm/baseline_2022/priority6_btm_baseline_2022.pkl'
df = pd.read_pickle(f)
print(df)
for c in df.columns: print(c)

# f = '/projects/dwind/archive/dwfs_2022/fom_baseline_2022/dwfs/analysis/weights/lkup_gid_to_weights_100k.csv'
# weights = pd.read_csv(f)
# df.drop(columns=['weight'], inplace=True)
# df = df[df['wind_breakeven_cost_fom'] > 0]
# df = df.merge(weights, how='left', on='gid')

# print(np.sum(df.weight.values) / 1000000)