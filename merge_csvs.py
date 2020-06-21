import os
import pandas as pd

league = 'All'
path = os.getcwd()
work_dir = os.path.join(path, league)
files = [os.path.join(work_dir, file) for file in os.listdir(work_dir) if os.path.isfile(os.path.join(work_dir, file))]
csvs = [c for c in files if c.endswith('.csv') and c != 'merged.csv']
data = [pd.read_csv(c) for c in csvs]
data_len = 0
for d in data:
    data_len += len(d)
print(data_len)
all_data = pd.concat(data)
all_data.drop_duplicates(inplace=True, subset=['Date', 'HomeTeam', 'AwayTeam'])
print(len(all_data))
all_data.to_csv(os.path.join(work_dir, f'{league}-merged.csv'))
