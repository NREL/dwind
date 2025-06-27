import numpy as np
import pandas as pd

# issue: BTM baseline 2022 has a much lower econ potential (from 919GW to 25.4GW)


# ['BTM', 'BTM, FOM', 'BTM, FOM, Utility', 'FOM, Utility']
# btm only: ['BTM']
# fom only: ['FOM', 'Utility']
# btm and fom: ['BTM, FOM', 'BTM, FOM, Utility']


# DONE: re-run collect.py on FOM results
# TODO: find tech/econ pot for: BTM only, BTM + FOM, FOM only
# DONE: transfer to kestrel


def summary(scenario):
    app = scenario[:3]
    scen = scenario[4:]
    year = scenario[-4:]
    
    if app == 'btm':
        run_dir = 'runs_2024'
        benchmark = 5675 if year == '2022' else 4354
        # apps = ['BTM', 'BTM, FOM', 'BTM, FOM, Utility']
    else:
        run_dir = 'runs_2023'
        benchmark = 1608 if year == '2022' else 993
        # apps = ['BTM, FOM', 'BTM, FOM, Utility', 'FOM, Utility']

    
    dfs = []
    for i in range(1, 9):
        f = f'/projects/dwind/{run_dir}/priority{i}/{app}/{scen}/priority{i}_{scenario}.pkl'
        df_cur = pd.read_pickle(f)
        
        df_cur = df_cur[~df_cur[f'wind_breakeven_cost_{app}'].isna()]
        df_cur = df_cur[df_cur[f'wind_breakeven_cost_{app}'] > 0]
        # df_cur = df_cur.drop(columns='application')
        
        # # join back load application mapping
        # load_df = pd.read_csv(
        #     '/projects/dwind/data/parcel_landuse_load_application_mapping.csv')
        # load_df = load_df[['land_use', 'application']]

        # df_cur = df_cur.merge(
        #     load_df,
        #     on='land_use',
        #     how='left'
        # )
        
        # df_cur = df_cur[df_cur['application'].isin(apps)]
        
        # remove duplicates
        # df_cur = df_cur.sort_values(f'wind_breakeven_cost_{app}', ascending=False)
        # df_cur = df_cur.drop_duplicates(subset=['gid'], keep='first')
        
        df = pd.DataFrame.from_dict({'priority': [i]})
        for col in [f'wind_size_kw_{app}', 'wind_size_kw_techpot']:
            cur = np.nansum(df_cur[col].values)
            df[f'{col}'] = cur
        
        df_econ = df_cur[df_cur[f'wind_breakeven_cost_{app}'] >= benchmark]
        df['wind_size_kw_econpot'] = np.nansum(df_econ[f'wind_size_kw_{app}'].values)
        df['wind_size_kw_techpot_econpot'] = np.nansum(df_econ['wind_size_kw_techpot'].values)

        dfs.append(df)
        
    df_all = pd.concat(dfs)
    # df_all = pd.concat(dfs, ignore_index=True)

    techpot_gw_btm = np.nansum(df_all[f'wind_size_kw_{app}'].values) / 1000000
    techpot_gw_fom = np.nansum(df_all.wind_size_kw_techpot.values) / 1000000
    econpot_gw_btm = np.nansum(df_all.wind_size_kw_econpot.values) / 1000000
    econpot_gw_fom = np.nansum(df_all.wind_size_kw_techpot_econpot.values) / 1000000
    
    print(scenario)
    print('techpot_gw_btm', techpot_gw_btm)
    print('techpot_gw_fom', techpot_gw_fom)
    print('econpot_gw_btm', econpot_gw_btm)
    print('econpot_gw_fom', econpot_gw_fom)


summary('fom_baseline_2035')

""" 
2024 dwind runs:

BTM
btm_baseline_2022
techpot_gw_btm 87.32178
techpot_gw_fom 2982.5084675
econpot_gw_btm 24.040015
econpot_gw_fom 1295.296825

btm_baseline_2035
techpot_gw_btm 87.90522
techpot_gw_fom 2988.0605925
econpot_gw_btm 41.7560175
econpot_gw_fom 1909.34447

2023 dwind runs:

BTM
btm_baseline_2022
techpot_gw_btm 2990.0946575
techpot_gw_fom 2990.0946575
econpot_gw_btm 41.09299
econpot_gw_fom 41.09299

btm_baseline_2035
techpot_gw_btm 2913.9068075
techpot_gw_fom 2913.9068075
econpot_gw_btm 1200.72678
econpot_gw_fom 1200.72678

FOM

fom_baseline_2022
techpot_gw_fom 8695.120535
econpot_gw_fom 584.4085

fom_baseline_2035
techpot_gw_fom 7795.97
econpot_gw_fom 150.482
"""

# gid = "001c4496-eeb7-460c-8d1b-a7bd8b80da1b"  # california
# gid = "00229ef8-8e44-4364-ba4b-3d558cf9179e"  # florida
# gid = "0026a2d9-eb25-4694-ae5b-7a2688e24a33"  # iowa

# 2022 runs
# f_old = '/projects/dwind/archive/dwfs_2022/btm_baseline_2022/dwfs/output/run_BTM_baseline_2022_07012022.pkl'
# df = pd.read_pickle(f_old)
# # df = df[df["gid"] == gid]
# for c in df.columns:
#     # print(c, df[c].values[0])
#         print(c)

# 2024 runs
# f = '/projects/dwind/runs_2024/priority1/btm/baseline_2022/priority1_btm_baseline_2022.pkl'
# df = pd.read_pickle(f)
# # df = df[df["gid"] == gid]
# for c in df.columns:
#     # print(c, df[c].values[0])
#     print(c)

# f = '/projects/dwind/runs_2023/priority1/fom/baseline_2022/priority1_fom_baseline_2022.pkl'
# df = pd.read_pickle(f)
# # df = df[df["gid"] == gid]
# for c in df.columns:
#     # print(c, df[c].values[0])
#     print(c)

# 2022 agents
# f_agents_old = "/projects/dwind/archive/dwfs_data/1mill_sample/parcels_1mill_baseline.pkl"
# df_agents_old = pd.read_pickle(f_agents_old)
# print(np.unique(df_agents_old.application.values)) # ['BTM' 'BTM, FOM' 'BTM, FOM, Utility' 'FOM, Utility']
# df_agents_old = df_agents_old[df_agents_old["gid"] == gid]
# print(df_agents_old)
# for c in df_agents_old.columns:
#     print(c, df_agents_old[c].values[0])

# 2024 agents
# f_agents = "/projects/dwind/agents/iowa/agents_dwind_btm.pickle"
# df_agents = pd.read_pickle(f_agents)
# print(np.unique(df_agents.application.values))
# df_agents = df_agents[df_agents["gid"] == gid]
# for c in df_agents.columns:
#     print(c, df_agents[c].values[0])


# f_agents = f'/projects/dwind/agents/priority1/agents_dwind_fom_priority1.pickle'
# df = pd.read_pickle(f_agents)
# print(df)  # 8,050,351
# print(np.unique(df.application.values))  # ['BTM' 'BTM, FOM' 'BTM, FOM, Utility' 'FOM, Utility']

# f = f'/projects/dwind/runs_2023/priority1/fom/baseline_2022/priority1_fom_baseline_2022.pkl'
# df = pd.read_pickle(f)
# print(df)  # 8,050,351
# print(np.unique(df.application.values))  # ['FOM']

# f_agents = f'/projects/dwind/agents/priority1/agents_dwind_btm_priority1.pickle'
# df = pd.read_pickle(f_agents)
# print(df)  # 3,786,222
# print(np.unique(df.application.values))  # ['BTM' 'BTM, FOM' 'BTM, FOM, Utility']

# f = f'/projects/dwind/runs_2024/priority1/btm/baseline_2022/priority1_btm_baseline_2022.pkl'
# df = pd.read_pickle(f)
# # print(len(np.unique(df.gid.values)))  # 3,786,222
# # print(np.unique(df.application.values))  # ['BTM']
# gb = df.groupby('gid').agg({'gid': 'count'})
# gb.columns = ['count']
# gb = gb.reset_index(drop=False)

# dups = gb[gb['count'] > 1]
# gids = dups.gid.values  # 552,115 duplicates
# df = df[df['gid'].isin(gids)]
# print(df)
# df.to_csv('btm_2022_duplicates.csv', index=False)