import pandas as pd

locations_dict = {'frontal':['Fp1','Fp2','F3','F4','F7','F8','Fz'],
         'parietal':['P3','P4','Pz'],
         'occipital':['O1','O2'],
         'temporal':['T3','T4','T5','T6'],
         'central':['C3','C4','Cz'],
         }

locations_dict = {'frontal':['Fp1','Fp2'],
         'parietal':['P3','P4'],
         'occipital':['O1','O2'],
         'temporal':['T3','T4'],
         'central':['C3','C4'],
         }

all_channels = ['Fp1', 'F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'P3',
                'Fz', 'Cz', 'Pz',
                'Fp2', 'F8', 'T4', 'T6', 'O2', 'F4', 'C4', 'P4']
df = pd.read_excel('/media/moritz/Expansion/Data/Spikes_clemson_10s/tables/segments_labels_channels_montage.xlsx')

#df = df.fillna(0)
#df = df.replace('-',0)
#df[all_channels] = df[all_channels].astype(int)


result = {'event_file':[],'location':[]}
for location in locations_dict.keys():
    # get channels we are looking for
    location_channels = locations_dict[location]
    # if any of the channel columns has 
    #location_df = df[(df[location_channels]!=0).any(axis=1)]
    location_df = df[df[location_channels].apply(lambda r: r.str.contains(',',).any(), axis=1)] 
    result['event_file']+=location_df.event_file.to_list()
    result['location']+=[location]*len(location_df)


result = pd.DataFrame(result)
result.to_csv('/home/moritz/Desktop/programming/SpikeDeletionOld/Data/tables/clemson_all_locs.csv',index=False)
