import os 
import pandas as pd

root = '/home/moritz/Desktop/programming/SpikeDeletion/mount/MoritzAlkofer/SpikeDeletion'
Folders = ['v1_bonobo','v3_bonobo_IFCN6']
locations = ['frontal','central','parietal','occipital','temporal','general']


results = {'event_file':[],'location':[]}

folder = 'v1_bonobo'
for location in locations:
    try:
        path = os.path.join(root,folder,location)
        files = [f.replace('.png','') for f in os.listdir(path)]
        results['event_file']+=files
        results['location']+=[location]*len(files)
        print(f'found {folder} {location}')
    except:
        pass

folder = 'v3_bonobo_IFCN6'
for location in locations:
    try:
        path = os.path.join(root,folder,location+'_DONE')
        files = [f.replace('.png','') for f in os.listdir(path)]
        results['event_file']+=files
        results['location']+=[location]*len(files)
        print(f'found {folder} {location}')
    except:
        pass

result = pd.DataFrame(results)
result.to_csv('spike_location_13FEB24.csv',index=False)