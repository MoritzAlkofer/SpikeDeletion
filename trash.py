import pandas as pd
from local_utils import DatamoduleLoc

df = pd.read_csv('/home/moritz/Desktop/programming/SpikeDeletion_clean/tables/split_local_13FEB24.csv')
df.event_file.value_counts()

df[df.event_file=='Bonobo03234_1_12586']


df_center = pd.read_csv('tables/lut_event_23-08-22.csv')
df_member = pd.read_csv('tables/member_17JAN24.csv')
df_center[df_center.event_file=='Bonobo03234_1_12586']
df_member[df_member.eeg_file.str.replace('.mat','')=='Bonobo03234_1_12586']

locations_og = pd.read_csv('tables/locations_13FEB24.csv')

module = DatamoduleLoc(1,1)
event_files = module.get_event_files('Train')
df = pd.DataFrame({'event_files':event_files})
df = pd.DataFrame({'event_files':event_files,'locations':locations})


datamodule 
predictions
visualisation

