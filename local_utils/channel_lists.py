all_referential = ['Fp1','F3','C3','P3','F7','T3','T5','O1', 'Fz','Cz','Pz', 'Fp2','F4','C4','P4','F8','T4','T6','O2']

all_bipolar = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 
               'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
               'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
               'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
               'Fz-Cz', 'Cz-Pz']

all_average = ['Fp1-avg','F3-avg','C3-avg','P3-avg','F7-avg','T3-avg','T5-avg','O1-avg',
               'Fz-avg','Cz-avg','Pz-avg',
               'Fp2-avg','F4-avg','C4-avg','P4-avg','F8-avg','T4-avg','T6-avg','O2-avg']
        
six_bipolar = ['F3-C3','C3-O1','F4-C4','C4-O2']

two_frontal = ['Fp1','Fp2']
two_central = ['C3','C4']

six_referential = ['F3','C3','O1','F4','C4','O2']

points_of_interest = {'Fp1, Fp2':two_frontal,
                'C3, C4':two_central,
                'T3, F7, T4 F8':['T3','F7','T4','F8'],
                'T3, P3, Pz, T4, P4':['T3','P3','Pz','T4','P4'],
                'F3, C3, O1, F4, C4, O2':six_referential,
                'all 10-20 channels':all_referential,
                }

localized_channels = {'frontal':['Fp1','Fp2'],
                     'parietal':['P3','P4'],
                     'occipital':['O1','O2'],
                     'temporal':['T3','T4'],
                     'central':['C3','C4'],
                     'general':['Fp1','F3','C3','P3','F7','T3','T5','O1', 'Fz','Cz','Pz', 'Fp2','F4','C4','P4','F8','T4','T6','O2']}