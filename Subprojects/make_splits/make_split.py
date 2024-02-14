import pandas as pd
import numpy as np

# datasets
def DatasetCenter():
    df_center = pd.read_csv('../../Data/tables/lut_event_23-08-22.csv')
    df_center['dataset']='center'
    path_center = '/media/moritz/a80fe7e6-2bb9-4818-8add-17fb9bb673e1/Data/Bonobo/cluster_center/' 
    df_center['path'] = path_center+df_center['event_file']+'.npy'
    return df_center

def DatasetMember():
    df_member = pd.read_csv('../../Data/tables/member_17JAN24.csv')
    df_member['event_file'] = df_member.eeg_file.str.replace('.mat','', regex=True)
    df_member['patient_id']=df_member.eeg_file.str.split('_').str[0]
    df_member['total_votes_received']=8
    df_member['dataset']='member'
    path_member = '/media/moritz/a80fe7e6-2bb9-4818-8add-17fb9bb673e1/Data/Bonobo/cluster_members/' 
    df_member['path'] = path_member+df_member['event_file']+'.npy'
    return df_member

def DatasetControl(N=30000):
    df_control = pd.read_csv('/home/moritz/Desktop/programming/epilepsy_project/tables/bonobo/lut_event_random_controls_23-11-06.csv')
    df_control['patient_id']=df_control.eeg_file.str.split('_').str[0]
    df_control['total_votes_received']=8
    df_control['dataset']='control'
    path_control = '/media/moritz/a80fe7e6-2bb9-4818-8add-17fb9bb673e1/Data/Bonobo/random_snippets/' 
    df_control['path'] = path_control+df_control['event_file']+'.npy'
    return df_control[:N]

# load add datasets and concat dfs
def get_dataframe(N_control):
    df_center = DatasetCenter()
    df_member = DatasetMember()
    df_control = DatasetControl(N_control)

    columns = ['path','event_file','patient_id','total_votes_received','fraction_of_yes','dataset']
    df = pd.concat([df_center[columns],df_member[columns],df_control[columns]])
    return df

# add patientwise stratified split to df
def assign_representative_splits(df):
    p = [0.8, 0.1,0.1] 
    vals = np.random.choice(['Train', 'Val','Test'], size=df.patient_id.nunique(), p=p)

    from tqdm import tqdm
    for patient_id, val in tqdm(zip(df.patient_id.unique(),vals),total=len(vals)):
        df.loc[df.patient_id==patient_id,'Mode']=val

    return df

def assing_location_splits(df,df_info):
    df = df.merge(df_info,how='left',on='event_file')
    df['Mode']=None
    localized_patients = df[~df.location.isna()].patient_id.unique()
    df.loc[df.patient_id.isin(localized_patients),'Mode']='Test'
    unlocalized_patients = df[df.Mode.isna()].patient_id.unique()
    p = [0.8, 0.2] 
    vals = np.random.choice(['Train', 'Val'], size=len(unlocalized_patients), p=p)

    from tqdm import tqdm
    for patient_id, val in tqdm(zip(unlocalized_patients,vals),total=len(vals)):
        df.loc[df.patient_id==patient_id,'Mode']=val

    return df

if __name__=='__main__':
    split = 'localized'
    np.random.seed(1)
    if split == 'representative':
        N_control = 30000
        # get dataframes for center, member and control
        df = get_dataframe(N_control)
        df = assign_representative_splits(df)
        df.to_csv('split_representative_12FEB24.csv',index=False)
    if split == 'localized':
        N_control = 10000
        # get dataframes for center, member and control
        df = get_dataframe(N_control)
        df_info = pd.read_csv('spike_location_13FEB24.csv')
        df_local = assing_location_splits(df,df_info)
        df_local.to_csv('split_local_13FEB24.csv',index=False)