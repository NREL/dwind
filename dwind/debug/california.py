import pandas as pd

f = '/projects/dwind/agents/california/agents_dwind_btm.pickle'
df = pd.read_pickle(f)
# df = pd.read_parquet(f)
print(df)
df = df[~df['rate_id_alias'].isna()]
print(df)
# for c in df.columns: print(c)

def collect_results(n=4):
    dfs = []
    for i in range(n):
        f = f'/projects/dwind/runs_2023/california/fom/chunk_files_california_fom_baseline_2022/california_fom_baseline_2022_{i}.pkl'
        df = pd.read_pickle(f)
        df = df[['gid', 'lat', 'lon', 'wind_breakeven_cost_fom']]
        dfs.append(df)
    
    df = pd.concat(dfs)
    df.to_csv('test.csv', index=False)
    
# collect_results()
