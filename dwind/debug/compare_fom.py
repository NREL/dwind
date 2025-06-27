import os
import pandas as pd

# fom_baseline_2022 vs fom_baseline_2035

# nem scenario
nem_scenario_2022 = 'nem_baseline_2022.csv'
nem_scenario_2035 = 'nem_baseline_2035.csv'

# cost inputs
cost_inputs_2022 = {
    'system_om_per_kw': 42.79,
    'system_capex_per_kw': 2629
}
cost_inputs_2035 = {
    'system_om_per_kw': 37,
    'system_capex_per_kw': 1841
}

# performance inputs
performance_inputs_2022 = {
    'perf_improvement_factor': {
        2.5: 0.083787756,
        5.0: 0.083787756,
        10.0: 0.083787756,
        20.0: 0.083787756,
        50.0: 0.116657183,
        100.0: 0.116657183,
        250.0: 0.106708234,
        500.0: 0.106708234,
        750.0: 0.106708234,
        1000.0: 0.106708234,
        1500.0: 0.106708234
    }
}
performance_inputs_2035 = {
    'perf_improvement_factor': {
        2.5: 0.23136759,
        5.0: 0.23136759,
        10.0: 0.23136759,
        20.0: 0.23136759,
        50.0: 0.23713196,
        100.0: 0.23713196,
        250.0: 0.23617185,
        500.0: 0.23617185,
        750.0: 0.23617185,
        1000.0: 0.23617185,
        1500.0: 0.23617185
    }
}

# financial inputs
financial_inputs_2022 = {
    'interest_rate': 0.0146,
    'debt_percent': 0.537,
    'ptc_fed_dlrs_per_kwh': 0.015
}

financial_inputs_2035 = {
    'interest_rate': 0.024,
    'debt_percent': 0.670,
    'ptc_fed_dlrs_per_kwh': 0
}

def compare_fom_by_scenario(state):
    cols = [
        'yr',
        'system_om_per_kw',
        'system_capex_per_kw',
        'term_int_rate',
        'debt_percent'
    ]
    
    p = '/projects/dwind/runs_2023'
    
    f_2022 = f'{state}_fom_baseline_2022.pkl'
    f_2022 = os.path.join(p, state, 'fom', f_2022)

    f_2035 = f'{state}_fom_baseline_2035.pkl'
    f_2035 = os.path.join(p, state, 'fom', f_2035)
    
    df_2022 = pd.read_pickle(f_2022)
    df_2035 = pd.read_pickle(f_2035)
    
    for c in cols:
        print(c, '2022:')
        print(df_2022[c])
        
        print(c, '2035:')
        print(df_2035[c])


compare_fom_by_scenario('minnesota')
