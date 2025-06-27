import os
import sys
import pandas as pd

sys.path.append("/home/jlockshi/dwind/python/")
from dwind.model import Agents, Model

f_agents = "/projects/dwind/agents/colorado/agents_dwind_btm.pickle"
agent_extractor = Agents(agent_pkl=f_agents)
agents = agent_extractor.agents
agents = agents[agents["county_file"] == "larimer"]
agents = agents.iloc[:5000]

# run_name = f'{sample_name}_{scenario}'
run_name = "coloradolarimer_btm_baseline_2025"

model = Model(
    agents, 
    run_name=run_name, 
    out_path=os.getcwd()
)

model.run()