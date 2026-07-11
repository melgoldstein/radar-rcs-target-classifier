"""Generate correlated returns grouped by independent synthetic flight."""
from pathlib import Path
import numpy as np
import pandas as pd
def generate(seed=72):
    generator=np.random.default_rng(seed); rows=[]
    for label, center in [('seabird',(-22,8,9)),('quadcopter',(-16,30,12)),('fixed_wing',(-6,12,65))]:
        for flight in range(40):
            flight_offset=generator.normal(0,2,3)
            for pulse in range(12):
                signature=np.array(center)+flight_offset+generator.normal(0,[5,8,9])
                rows.append([f'{label}-{flight}',label,*signature])
    return pd.DataFrame(rows,columns=['flight_id','target_class','rcs_dbsm','micro_doppler_hz','radial_velocity_mps'])
if __name__=='__main__':
    Path('data').mkdir(exist_ok=True); generate().to_csv('data/returns.csv',index=False)
