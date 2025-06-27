
import os
import pandas as pd
from sqlalchemy import create_engine


atlas_user = 'dwind'
atlas_pwd = 'distributedwindfutures'
atlas_pg_con_str = f'postgresql://{atlas_user}:{atlas_pwd}@1lv11dnpg01.nrel.gov:5432/dgen_db_fy17q3_dwind'

sage_user = 'dwindwrite'
sage_pwd = 'gRdDefVu4FGk!W'
sage_pg_con_str = f'postgresql://{sage_user}:{sage_pwd}@sage.hpc.nrel.gov:5432/dwind'


def export_atlas_table(name, csv):
    sql = f"""select * from diffusion_shared."{name}";"""
    atlas_engine = create_engine(atlas_pg_con_str)
            
    df = pd.read_sql(sql, atlas_engine)
    atlas_engine.dispose()

    sage_engine = create_engine(sage_pg_con_str)
    df.to_sql(name, sage_engine, schema='atb20', if_exists='replace')
    sage_engine.dispose()
    
    if csv:
        p_dir = '/projects/dwind/configs/costs/atb20'
        f = os.path.join(p_dir, f'{name}.csv')
        df.to_csv(f, index=False)
    

tables = [
    'urdb3_rate_jsons_20200721'
    # 'ATB20_Mid_Case_retail',
    # 'ATB20_Mid_Case_wholesale',
    # 'financing_atb_FY20',
    # 'deprec_sch_FY17',
    # 'wind_prices_1_reference',
    # 'wind_tech_performance_1_reference',
    # 'wind_derate_sched_1_reference',
    # 'batt_prices_FY20_mid',
    # 'batt_tech_performance_SunLamp17',
    # 'pv_price_atb20_mid',
    # 'pv_tech_performance_defaultFY19',
    # 'pv_plus_batt_prices_FY20_mid_pv_mid_batt'
]

for name in tables:
    if name == 'urdb3_rate_jsons_20200721':
        export_atlas_table(name, csv=False)
    else:
        export_atlas_table(name, csv=True)