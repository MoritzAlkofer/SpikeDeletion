import pandas as pd
import numpy as np


def load_dataset(dataset='center'):
    if dataset=='center':
        df = pd.read_csv('../../tables/center_26JAN24.csv')
        df = df.rename(columns={'patient_id':'BonoboIndex'})
        df['eeg_file'] = df.event_file.str.split('_').str[0] + '_'+ df.event_file.str.split('_').str[1]
    if dataset=='member':
        df = pd.read_csv('../../tables/member_17JAN24.csv')
        df = df.rename(columns={'eeg_file':'event_file'})
        df['BonoboIndex'] = df.event_file.str.split('_').str[0]
        df['eeg_file'] = df.event_file.str.split('_').str[0] + '_'+ df.event_file.str.split('_').str[1]
    if dataset=='control':
        df = pd.read_csv('../../tables/control_17JAN24.csv')[:30000]
        df['BonoboIndex'] = df.event_file.str.split('_').str[0]
    if dataset=='external':
        df = pd.read_excel('/media/moritz/Expansion/Data/Spikes_clemson_10s/tables/segments_labels_channels_montage.xlsx')
        df['BonoboIndex'] = df.event_file.str.split('_').str[0]
        df['eeg_file'] = df['BonoboIndex']
        print('AUSSMING THAT EACH PATIENT FROM EXTERNAL DATASET HAS 1 RECORDING!')
        df = df.rename(columns={'Spike':'fraction_of_yes'})
    df['dataset']=dataset
    return df

def load_EEG_and_reports():
    df_info = pd.read_csv('/media/moritz/Expansion/Data/master/raw/EEGs_And_Reports_20231024.csv')
    df_info = df_info.rename({'SexDSC':'Sex'},axis=1)
    df_info = df_info[['BonoboIndex','BDSPPatientID','Sex','ServiceName','AgeAtVisit']].drop_duplicates('BonoboIndex')
    return df_info

def filter_known(df,known=True):
    if known:
        return df[~df.Sex.isna()]
    else:
        return df[df.Sex.isna()]

def get_info(df,spike_threshold):
    # patient level
    N_patients = df.BonoboIndex.nunique()
    N_female = df[df.Sex=='Female'].BonoboIndex.nunique()
    N_male = df[df.Sex=='Male'].BonoboIndex.nunique()+1e-100
    # event level
    N_samples = df.event_file.nunique()
    N_spikes = df[df.fraction_of_yes>spike_threshold].event_file.nunique()
    # eeg level
    N_recording = df.eeg_file.nunique()
    N_Routine = df[df.ServiceName=='Routine'].eeg_file.nunique()
    N_ltm = df[df.ServiceName=='LTM'].eeg_file.nunique()
    N_emu = df[df.ServiceName=='EMU'].eeg_file.nunique()
    N_Other = df[~df.ServiceName.isin(['Routine','LTM','EMU'])].eeg_file.nunique()
    return N_patients,N_recording, N_female,N_male, N_samples, N_spikes, N_Routine, N_ltm, N_emu, N_Other

def get_frac(n,n_total):
    frac = int(np.round(n/n_total,2)*100)
    return frac 

def write_to_results(results,group_name,N_patients,N_recording, N_female,N_male, N_samples, N_spikes, N_Routine, N_ltm, N_emu, N_Other):
        results['Group name'].append(group_name)
        # patient level
        results['Patients'].append(str(N_patients))
        results['Female'].append(f'{N_female} ({get_frac(N_female,N_female+N_male)})')
        # sample level
        results['Samples'].append(str(N_samples))
        results['Spikes'].append(f'{N_spikes} ({get_frac(N_spikes,N_samples)})')
        # recording level
        results['Recordings'].append(str(N_recording))
        results['Routine'].append(f'{N_Routine} ({get_frac(N_Routine,N_recording)})')
        results['LTM'].append(f'{N_ltm} ({get_frac(N_ltm,N_recording)})')
        results['EMU'].append(f'{N_emu} ({get_frac(N_emu,N_recording)})')
        results['Other'].append(f'{N_Other} ({get_frac(N_Other,N_recording)})')
        return results

def build_table1(df,buckets,spike_threshold):
    results = {'Group name':[],'Patients':[],'Recordings':[],'Female':[],
               'Samples':[],'Spikes':[],'Routine':[], 'LTM':[], 'EMU':[], 'Other':[]}
    
    group_name = 'Total'
    N_patients, N_recording, N_female, N_male, N_samples, N_spikes, N_Routine, N_ltm, N_emu, N_Other = get_info(df,spike_threshold)
    results = write_to_results(results,group_name,N_patients, N_recording, N_female, N_male,
                                N_samples, N_spikes, N_Routine, N_ltm, N_emu, N_Other)

    group_name = 'Internal'
    sub_df = df[df.dataset!='external']
    N_patients, N_recording, N_female, N_male, N_samples, N_spikes, N_Routine, N_ltm, N_emu, N_Other = get_info(sub_df,spike_threshold)
    results = write_to_results(results,group_name,N_patients, N_recording, N_female, N_male,
                                N_samples, N_spikes, N_Routine, N_ltm, N_emu, N_Other)

    group_name = 'External'
    sub_df = df[(df.dataset=='external')]
    N_patients, N_recording, N_female, N_male, N_samples, N_spikes, N_Routine, N_ltm, N_emu, N_Other = get_info(sub_df,spike_threshold)
    results = write_to_results(results,group_name,N_patients, N_recording, N_female, N_male,
                                N_samples, N_spikes, N_Routine, N_ltm, N_emu, N_Other)

    for age_min, age_max in buckets:
        # get sub df for age bucket
        sub_df = df[(df.AgeAtVisit>=age_min)&(df.AgeAtVisit<age_max)]
        group_name = f'{age_min}  to < {age_max}'
        N_patients, N_recording, N_female, N_male, N_samples, N_spikes, N_Routine, N_ltm, N_emu, N_Other = get_info(sub_df,spike_threshold)
        results = write_to_results(results,group_name,N_patients, N_recording, N_female, N_male,
                                N_samples, N_spikes, N_Routine, N_ltm, N_emu, N_Other)

    group_name = 'Unknown'
    sub_df = df[df.Sex.isna()]
    N_patients, N_recording, N_female, N_male, N_samples, N_spikes, N_Routine, N_ltm, N_emu, N_Other = get_info(sub_df,spike_threshold)
    results = write_to_results(results,group_name,N_patients, N_recording, N_female, N_male,
                                N_samples, N_spikes, N_Routine, N_ltm, N_emu, N_Other)

    table1= pd.DataFrame(results)    
    return table1

if __name__=='__main__':
    buckets = buckets = [[0,5],[5,13],[13,18],[18,30],[30,50],[50,65],[65,75],[75,np.inf]]
    spike_threshold = 0.5

    df_info = load_EEG_and_reports()

    dfs = []
    for dataset in ['center','member','control','external']:
        df = load_dataset(dataset)
        dfs.append(df[['event_file','eeg_file','BonoboIndex','dataset','fraction_of_yes']])
    
    df = pd.concat(dfs)
    df = df.drop_duplicates('event_file')
    df = df.merge(df_info,how='left',on='BonoboIndex')
    table1 = build_table1(df,buckets,spike_threshold)
    table1.to_csv('table1.csv',index=False)

