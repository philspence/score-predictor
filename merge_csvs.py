import os
import pandas as pd

league = 'all'
path = os.getcwd()
work_dir = os.path.join(path, league)
files = [os.path.join(work_dir, file) for file in os.listdir(work_dir) if os.path.isfile(os.path.join(work_dir, file))]
csvs = [c for c in files if c.endswith('.csv') and c != os.path.join(work_dir, 'merged.csv')]
# columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST']
data = []
skipped = []
for c in csvs:
    print(c)
    try:
        df = pd.read_csv(c, infer_datetime_format=True)
        df.dropna(inplace=True, subset=['Date'])
        df.dropna(inplace=True, axis=1, how='all')
        df.drop_duplicates(inplace=True)
        df['Date'] = pd.to_datetime(df.Date)
        df.sort_values(by=['Date'])
        start = pd.DatetimeIndex(df['Date']).year.min()
        end = pd.DatetimeIndex(df['Date']).year.max()
        df.insert(1, 'Season', f'{start}_{end}')
        data.append(df)
    except:
        skipped.append(c)
print(skipped)
print(f'Skipped: {len(skipped)} CSVs')
data_len = 0
for d in data:
    data_len += len(d)
all_data = pd.concat(data, ignore_index=True)
print(f'Total matches: {data_len}')
all_data.to_csv(os.path.join('all', f'{league}_merged.csv'))

