import pandas as pd

data = pd.read_csv('EPL_form_elo_API.csv', header=0)
corr = data.corr()
print(corr.to_csv('correlations.csv'))
print(corr.to_string())
