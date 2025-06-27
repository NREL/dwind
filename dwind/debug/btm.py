import os
import sys
import pandas as pd

sys.path.append('/home/jlockshi/dwind/python/')
from dwind.model import Agents, Model


def run_model(sample_name, i_start, i_end, node_name, out_path):
    f_agents = os.path.join(
        '/projects/dwind/agents',
        sample_name,
        f'agents_dwind_btm.pickle'
    )

    agent_extractor = Agents(agent_file=f_agents)
    all_agents = agent_extractor.agents

    agents = all_agents.iloc[i_start:i_end]

    model = Model(
        agents, 
        run_name=node_name, 
        out_path=out_path
    )

    model.run()
    

sample_name = 'indiana'
i_start  = 0
i_end = 100
node_name = 'indiana_btm_baseline_2022'
out_path = os.getcwd()
run_model(sample_name, i_start, i_end, node_name, out_path)


f = '/home/jlockshi/dwind/python/debug/indiana_btm_baseline_2022.pkl'
df = pd.read_pickle(f)
df.to_csv('/home/jlockshi/dwind/python/debug/indiana_btm_baseline_2022.csv', index=False)
# print(df)
# for c in df.columns: print(c)
