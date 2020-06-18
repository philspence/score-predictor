import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('merged_noodds.csv', header=0)

moving_avg = data.groupby('HomeTeam')['FTHG'].rolling(window=10).mean().reset_index()
data['HFAvg'] = moving_avg.set_index('level_1')['FTHG']
moving_avg = data.groupby('HomeTeam')['FTAG'].rolling(window=10).mean().reset_index()
data['HAAvg'] = moving_avg.set_index('level_1')['FTAG']


moving_avg = data.groupby('AwayTeam')['FTAG'].rolling(window=10).mean().reset_index()
data['AFAvg'] = moving_avg.set_index('level_1')['FTAG']
moving_avg = data.groupby('AwayTeam')['FTHG'].rolling(window=10).mean().reset_index()
data['AAAvg'] = moving_avg.set_index('level_1')['FTHG']

data = data.dropna()

Xh = data[['HFAvg', 'AAAvg']].to_numpy()
Xa = data[['AFAvg', 'HAAvg']].to_numpy()
X = np.concatenate((Xh, Xa))

yh = data['FTHG'].to_numpy()
ya = data['HTAG'].to_numpy()
y = np.concatenate((yh, ya))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


model = Sequential()
model.add(Dense(2048, activation='tanh', input_dim=2))
#model.add(Dense(1024, activation='tanh'))
#model.add(Dense(512, activation='tanh'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam',  metrics=['mse'])
model.fit(X_train, y_train, epochs=10, verbose=1, validation_data=(X_test, y_test))



