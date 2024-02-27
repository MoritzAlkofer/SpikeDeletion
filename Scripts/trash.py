import pandas as pd

df = pd.read_csv('/home/moritz/Desktop/programming/SpikeDeletion/tables/split_local_13FEB24.csv')
df = df.path.str.replace('a80fe7e6-2bb9-4818-8add-17fb9bb673e1','internal_expansion')
df.to_csv('/home/moritz/Desktop/programming/SpikeDeletion/tables/split_local_13FEB24.csv',index=False)