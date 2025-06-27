import pandas as pd
from sqlalchemy import create_engine
       

engine = create_engine(
    'postgresql://dwindread:nABWLw8#VfeDYT@sage.hpc.nrel.gov:5432/dwind')

sql = """
    SELECT
        rate_name, 
        json
    FROM urdb.urdb3_rate_jsons_20200721
    limit 1;
"""
tariff = pd.read_sql(sql, engine)
engine.dispose()

return_val = dict(tariff['json'].iloc[0]) 
return_val.update({'rate_name': tariff['rate_name'].iloc[0]})

print(dict(return_val))
