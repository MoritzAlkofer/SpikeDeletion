import sys
sys.path.append('../../Scripts')
from make_datamodule import datamodule, get_split_df, datamoduleLocal, get_split_dfLocal

for mode in ['Train','Val','Test']:
    module = datamoduleLocal(transforms=None,batch_size=256,echo=False)
    df = get_split_dfLocal(module.df,mode)
    print(f'{mode} {len(df)}')
      

for mode in ['Train','Val','Test']:
    module = datamodule(transforms=None,batch_size=256,echo=False)
    df = get_split_df(module.df,mode)
    print(f'{mode} {len(df)}')
      