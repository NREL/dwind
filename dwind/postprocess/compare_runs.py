import numpy as np
import pandas as pd


def export_comparison_summary(scenario):
    app = scenario[:3]
    scen = scenario[4:]
    
    cols = [
        'parcel_size_acres',
        'wind_size_kw',
        'wind_size_kw_techpot',
        f'wind_breakeven_cost_{app}'
    ]
    
    dfs = []
    i_start = 1 if scenario != 'btm_baseline_2022' else 2
    i_end = 9
    
    for i in range(i_start, i_end):
        f_old = f'/projects/dwind/runs_2023/priority{i}/old/{app}/{scen}/priority{i}_{scenario}.pkl'
        f_cur = f'/projects/dwind/runs_2023/priority{i}/{app}/{scen}/priority{i}_{scenario}.pkl'
        
        df_old = pd.read_pickle(f_old)
        df_cur = pd.read_pickle(f_cur)
        
        df_dict = {
            'priority': [i],
            'scenario': [scenario],
            'n_agents_old': [len(df_old.index)],
            'n_agents_cur': [len(df_cur.index)]
        }
        
        df = pd.DataFrame.from_dict(df_dict)
        
        df_old = df_old[~df_old[f'wind_breakeven_cost_{app}'].isna()]
        df_old = df_old[df_old[f'wind_breakeven_cost_{app}'] > 0]
        df_cur = df_cur[~df_cur[f'wind_breakeven_cost_{app}'].isna()]
        df_cur = df_cur[df_cur[f'wind_breakeven_cost_{app}'] > 0]
        
        df['n_agents_valid_old'] = len(df_old.index)
        df['n_agents_valid_cur'] = len(df_cur.index)
        
        for col in cols:
            if col == f'wind_breakeven_cost_{app}':
                old = np.mean(df_old[col].values)
                cur = np.mean(df_cur[col].values)
            else:
                old = np.nansum(df_old[col].values)
                cur = np.nansum(df_cur[col].values)

            df[f'{col}_old'] = old
            df[f'{col}_cur'] = cur
            
        print(df)
        dfs.append(df)
        
    df_final = pd.concat(dfs, ignore_index=True)
    print(df_final)
    df_final.to_csv(f'{scenario}_summary.csv', index=False)


def export_summary(scenario):
    app = scenario[:3]
    scen = scenario[4:]
    
    cols = [
        'parcel_size_acres',
        'wind_size_kw',
        'wind_size_kw_techpot',
        f'wind_breakeven_cost_{app}'
    ]
    
    dfs = []
    for i in range(1, 9):
        f_cur = f'/projects/dwind/runs_2023/priority{i}/{app}/{scen}/priority{i}_{scenario}.pkl'
        df_cur = pd.read_pickle(f_cur)
        
        df_dict = {
            'priority': [i],
            'scenario': [scenario],
            'n_agents_cur': [len(df_cur.index)]
        }
        
        df = pd.DataFrame.from_dict(df_dict)
        
        df_cur = df_cur[~df_cur[f'wind_breakeven_cost_{app}'].isna()]
        df_cur = df_cur[df_cur[f'wind_breakeven_cost_{app}'] > 0]
        
        df['n_agents_valid_cur'] = len(df_cur.index)
        
        for col in cols:
            if col == f'wind_breakeven_cost_{app}':
                cur = np.mean(df_cur[col].values)
            else:
                cur = np.nansum(df_cur[col].values)

            df[f'{col}'] = cur
            
        print(df)
        dfs.append(df)
        
    df_final = pd.concat(dfs, ignore_index=True)
    print(df_final)
    df_final.to_csv(f'{scenario}_summary.csv', index=False)


export_summary('fom_baseline_2035')
