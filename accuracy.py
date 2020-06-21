import pandas as pd

data = pd.read_csv('merged-EPL-preds.csv', header=0)
data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
data = data.dropna(subset=['Date'])
date_range = ('08/2018', '06/2019')
data = data.loc[date_range[0] < data.Date]
data = data.loc[data.Date < date_range[1]]
data = data.drop_duplicates()
liv_away = data[data.AwayTeam == 'Liverpool'].to_string()
print(liv_away)
