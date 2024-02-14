import pandas as pd
import numpy as np


def load_dataset(dataset='center'):
    if dataset=='center':
        df_samples = pd.read_csv('../../Data/tables/center_26JAN24.csv')
        df_samples = df_samples.rename(columns={'patient_id':'BonoboIndex'})
    if dataset=='member':
        df_samples = pd.read_csv('../../Data/tables/member_17JAN24.csv')
        df_samples['BonoboIndex'] = df_samples.eeg_file.str.split('_').str[0]
    if dataset=='control':
        df_samples = pd.read_csv('../../Data/tables/control_17JAN24.csv')[:30000]
        df_samples['BonoboIndex'] = df_samples.event_file.str.split('_').str[0]

    return df_samples

def load_EEG_and_reports(path_info):
    df_info = pd.read_csv(path_info)
    df_info = df_info.rename({'SexDSC':'Sex'},axis=1)
    df_info = df_info[['BonoboIndex','BDSPPatientID','Sex','ServiceName','AgeAtVisit']].drop_duplicates('BonoboIndex')
    return df_info

def load_lut_master_mgh():
    df = pd.read_csv('/home/moritz/Desktop/programming/SpikeDeletion/Data/tables/archive/OG_sleep_lookup_master_mgh_0515.csv')
    df = df.rename({'SexDSC':'Sex','EthnicGroupDSC':'EthnicGroup'},axis=1)
    df_info = df_info[['BDSPPatientID','AgeAtVisit','Sex','StudyType',]]
    return df

def add_info_to_sample(df_samples,df_info):
    df_samples = df_samples.merge(df_info,how='left',on='BonoboIndex')
    return df_samples

def make_buckets():
    buckets = [[0,5],[5,13],[13,18],[18,30],[30,50],[50,65],[65,75],[75,np.inf]]
    return buckets

def age_filter(df,age_min,age_max):
    df = df[(df.AgeAtVisit>=age_min)&(df.AgeAtVisit<age_max)]
    return df

def filter_known(df,known=True):
    if known:
        return df[~df.Sex.isna()]
    else:
        return df[df.Sex.isna()]

def get_info(df):
    N_patients = df.BonoboIndex.nunique()
    N_patients_samples = df[df.fraction_of_yes>0.5].BonoboIndex.nunique()
    N_samples = len(df)
    N_spikes = len(df[df.fraction_of_yes>0.5])
    N_female = df[df.Sex=='Female'].BonoboIndex.nunique()
    N_Routine = len(df[df.ServiceName=='Routine'])
    N_ltm = len(df[df.ServiceName=='LTM'])
    N_emu = len(df[df.ServiceName=='EMU'])
    N_Other = len(df[~df.ServiceName.isin(['Routine','LTM','EMU'])])
    return N_patients, N_patients_samples, N_samples, N_spikes, N_female,N_Routine,N_ltm,N_emu, N_Other

def get_frac(n,n_total):
    frac = int(np.round(n/n_total,2)*100)
    return frac 

def create_table(df,buckets):
    results = {'Age group':[],'Patients':[],'Female patients':[],'Samples':[],'Spikes':[],'Routine':[], 'LTM':[], 'EMU':[], 'Other':[]}
    for age_min, age_max in buckets:
        sub_df = age_filter(df,age_min,age_max)
    
        N_patients, N_patients_samples, N_samples, N_spikes, N_female,N_Routine,N_ltm,N_emu, N_Other = get_info(sub_df)
        results['Age group'].append(f'{age_min}  to < {age_max}')
        results['Patients'].append(str(N_patients))
        frac_sampled_spikes = int(np.round(N_patients_samples/N_patients,2)*100)
        #results['Patients with sampled spikes'].append(f'{N_patients_samples} ({frac_sampled_spikes})')
        results['Samples'].append(str(N_samples))
        frac_spikes = int(np.round(N_spikes/N_samples,2)*100)
        results['Spikes'].append(f'{N_spikes} ({frac_spikes})')
        frac_female = int(np.round(N_female/N_patients,2)*100)
        results['Female patients'].append(f'{N_female} ({frac_female})')
        results['Routine'].append(f'{N_Routine} ({get_frac(N_Routine,N_samples)})')
        results['LTM'].append(f'{N_ltm} ({get_frac(N_ltm,N_samples)})')
        results['EMU'].append(f'{N_emu} ({get_frac(N_emu,N_samples)})')
        results['Other'].append(f'{N_Other} ({get_frac(N_Other,N_samples)})')
    results['Age group'][-1]=f'75+'
    sub_df = filter_known(df,known=True)
    N_patients, N_patients_samples, N_samples, N_spikes, N_female,N_Routine,N_ltm,N_emu, N_Other = get_info(sub_df)
    results['Age group'].append('Partial total')
    results['Patients'].append(str(N_patients))
    frac_sampled_spikes = int(np.round(N_patients_samples/N_patients,2)*100)
    #results['Patients with sampled spikes'].append(f'{N_patients_samples} ({frac_sampled_spikes})')
    results['Samples'].append(str(N_samples))
    frac_spikes = int(np.round(N_spikes/N_samples,2)*100)
    results['Spikes'].append(f'{N_spikes} ({frac_spikes})')
    frac_female = int(np.round(N_female/N_patients,2)*100)
    results['Female patients'].append(f'{N_female} ({frac_female})')
    results['Routine'].append(f'{N_Routine} ({get_frac(N_Routine,N_samples)})')
    results['LTM'].append(f'{N_ltm} ({get_frac(N_ltm,N_samples)})')
    results['EMU'].append(f'{N_emu} ({get_frac(N_emu,N_samples)})')
    results['Other'].append(f'{N_Other} ({get_frac(N_Other,N_samples)})')

    sub_df = filter_known(df,known=False)
    N_patients, N_patients_samples, N_samples, N_spikes, N_female,N_Routine,N_ltm,N_emu, N_Other = get_info(sub_df)
    results['Age group'].append('Unknown')
    results['Patients'].append(str(N_patients))
    #results['Patients with sampled spikes'].append('?')
    results['Samples'].append(str(N_samples))
    frac_spikes = int(np.round(N_spikes/N_samples,2)*100)
    results['Spikes'].append(f'{N_spikes} ({frac_spikes})')
    frac_female = int(np.round(N_female/N_patients,2)*100)
    results['Female patients'].append('?')
    results['Routine'].append('?')
    results['LTM'].append('?')
    results['EMU'].append('?')
    results['Other'].append('?')

    N_patients, N_patients_samples, N_samples, N_spikes, N_female,N_Routine,N_ltm,N_emu, N_Other = get_info(df)
    results['Age group'].append('Total')
    results['Patients'].append(str(N_patients))
    #frac_sampled_spikes = int(np.round(N_patients_samples/N_patients,2)*100)
    #results['Patients with sampled spikes'].append(None)
    results['Samples'].append(str(N_samples))
    frac_spikes = int(np.round(N_spikes/N_samples,2)*100)
    results['Spikes'].append(f'{N_spikes} ({frac_spikes})')
    frac_female = int(np.round(N_female/N_patients,2)*100)
    results['Female patients'].append(None)
    results['Routine'].append(None)
    results['LTM'].append(None)
    results['EMU'].append(None)
    results['Other'].append(None)

    results= pd.DataFrame(results)
    results= results.append(pd.Series(dtype='float64'), ignore_index=True)
    return results 

if __name__=='__main__':
    path_df_EEG_and_reports = '/media/moritz/Expansion/Data/master/raw/EEGs_And_Reports_20231024.csv'
    df_info = load_EEG_and_reports(path_info=path_df_EEG_and_reports)
    buckets = make_buckets()
    table1s = []
    df_agg = []
    for dataset in ['center','member','control']:
        print(dataset)
        df_samples = load_dataset(dataset=dataset)
        df = add_info_to_sample(df_samples,df_info)
        df_agg.append(df)
        table1 = create_table(df,buckets)
        table1s.append(table1)
        
    df_aggregated = pd.concat(df_agg)
    table_1_aggregated = create_table(df_aggregated,buckets)
    total_table1 = pd.concat(table1s+[table_1_aggregated])
    
    total_table1.to_csv('table1.csv',index=False)

