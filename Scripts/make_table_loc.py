import pandas as pd
import sys
import os 
from make_datamodule import datamoduleLocal, get_split_dfLocal

module = datamoduleLocal(transforms=None,batch_size=256,echo=False)
df = get_split_dfLocal(module.df,'Test')
table = {'Location':[],'Samples':[],'Patients':[]}
for location in ['frontal','central','parietal','occipital','temporal','general']:
    Samples = len(df[df.location==location])
    Patients = df[df.location==location].patient_id.nunique()
    table['Location'].append(location)
    table['Samples'].append(Samples)
    table['Patients'].append(Patients)

Samples_neg = len(df[(df.Mode=='Test')&(df.fraction_of_yes==0)])
Patients_neg = df[(df.Mode=='Test')&(df.fraction_of_yes==0)].patient_id.nunique()
table['Location'].append('negative')
table['Samples'].append(Samples_neg)
table['Patients'].append(Patients_neg)

Samples = len(df[~df.location.isna()]) + Samples_neg
Patients = df[~df.location.isna()].patient_id.nunique() + Patients_neg 
table['Location'].append('Total')
table['Samples'].append(Samples)
table['Patients'].append(Patients)

result = pd.DataFrame(table)
print(result)
result.to_csv('table_loc.csv',index=False)