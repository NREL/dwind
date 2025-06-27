import pandas as pd

# combine agents by list of priority

# 3967437 agents fom
# 1394015 agents btm
priority_1 = [
    'minnesota',
    'wisconsin',
    'iowa',
    'illinois',
    'indiana',
    'michigan',
    'ohio',
    'pennsylvania',
    'newyork',
    'kentucky',
    'california'
]

# 2547494 agents fom
# 667336 agents btm
priority_2 = [
    'northdakota',
    'southdakota',
    'nebraska',
    'kansas',
    'oklahoma',
    'texas',
    'arkansas',
    'missouri'
]

# 567490 agents fom
# 124461 agents btm
priority_3 = [
    'montana',
    'wyoming',
    'colorado',
    'newmexico',
    'idaho'
]

# 161943 agents fom
# 75826 agents btm
priority_4 = [
    'washington',
    'oregon'
]

# 91680 agents fom
# 60852 agents btm
priority_5 = [
    'maine',
    'newhampshire',
    'vermont',
    'massachusetts',
    'connecticut',
    'rhodeisland'
]

# 233091 agents fom
# 59011 agents btm
priority_6 = [
    'arizona',
    'nevada',
    'utah'
]

# 167775 agents fom
# 81560 agents btm
priority_7 = [
    'new_jersey',
    'delaware',
    'maryland',
    'virginia',
    'west_virginia',
    'districtofcolumbia'
]

# 663283 agents fom
# 247493 agents btm
priority_8 = [
    'tennessee',
    'north_carolina',
    'south_carolina',
    'georgia',
    'florida',
    'alabama',
    'mississippi',
    'louisiana'
]


def combine_agents(priority, p_int):
    for app in ['fom', 'btm']:
        dfs = []
        for state in priority:
            f = f'/projects/dwind/agents/{state}/agents_dwind_{app}.pickle'
            df = pd.read_pickle(f)
            dfs.append(df)
            
        df_out = pd.concat(dfs)
        df_out = df_out.reset_index(drop=False)
        print(df_out)
        f_out = f'/projects/dwind/agents/priority{p_int}/agents_dwind_{app}_priority{p_int}.pickle'
        df_out.to_pickle(f_out)


combine_agents(priority_8, '8')