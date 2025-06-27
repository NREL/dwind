import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def scale_array_precision(row, col, prec_offset_col):
    row[col] = np.array(row[col], dtype='float64') / row[prec_offset_col]

    return row


def scale_array_sum(row, col, scale_col):
    hourly_array = np.array(row[col], dtype='float64')
    row[col] = hourly_array / hourly_array.sum() * np.float64(row[scale_col])

    return row


def get_load():
    engine = create_engine('postgresql://dwindread:nABWLw8#VfeDYT@sage.hpc.nrel.gov:5432/dwind')

    # grab consumption_hourly from sql
    sql = """
            select
                crb_model,
                hdf_index,
                consumption_hourly
            from "load".energy_plus_normalized_load_fixed;
        """
    consumption_hourly = pd.read_sql(sql, engine)
    engine.dispose()

    consumption_hourly['scale_offset'] = 1e8
    consumption_hourly = consumption_hourly.apply(
        scale_array_precision,
        axis=1,
        args=('consumption_hourly', 'scale_offset')
    )

    consumption_hourly['load_kwh'] = 7356.882324
    consumption_hourly = consumption_hourly.apply(
        scale_array_sum,
        axis=1,
        args=('consumption_hourly', 'load_kwh')
    )
    
    print(consumption_hourly)

    consumption = pd.Series(
        consumption_hourly['consumption_hourly']).values[0]
    
    consumption = np.array(consumption, dtype=np.float32)

    print(consumption)
    print(np.sum(consumption))


get_load()