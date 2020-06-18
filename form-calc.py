import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('merged_noodds.csv', header=0)


def get_mv_avg(df, key, value, win, newcol):
    moving_avg = df.groupby(key)[value].rolling(window=win).mean().reset_index()
    df[newcol] = moving_avg.set_index('level_1')[value]
    return df

w = 3
# HGFAvg = Home Goals For Average, S = Shots
data = get_mv_avg(data, 'HomeTeam', 'FTHG', w, 'HGFAvg')
data = get_mv_avg(data, 'HomeTeam', 'FTAG', w, 'HGAAvg')
data = get_mv_avg(data, 'HomeTeam', 'HS', w, 'HSFAvg')
data = get_mv_avg(data, 'HomeTeam', 'AS', w, 'HSAAvg')

data = get_mv_avg(data, 'AwayTeam', 'FTAG', w, 'AGFAvg')
data = get_mv_avg(data, 'AwayTeam', 'FTHG', w, 'AGAAvg')
data = get_mv_avg(data, 'AwayTeam', 'AS', w, 'ASFAvg')
data = get_mv_avg(data, 'AwayTeam', 'HS', w, 'ASAAvg')

# moving_avg = data.groupby('HomeTeam')['FTHG'].rolling(window=10).mean().reset_index()
# data['HFAvg'] = moving_avg.set_index('level_1')['FTHG']
# moving_avg = data.groupby('HomeTeam')['FTAG'].rolling(window=10).mean().reset_index()
# data['HAAvg'] = moving_avg.set_index('level_1')['FTAG']
#
#
#
# moving_avg = data.groupby('AwayTeam')['FTAG'].rolling(window=10).mean().reset_index()
# data['AFAvg'] = moving_avg.set_index('level_1')['FTAG']
# moving_avg = data.groupby('AwayTeam')['FTHG'].rolling(window=10).mean().reset_index()
# data['AAAvg'] = moving_avg.set_index('level_1')['FTHG']

data = data.dropna()

#Xh = data[['HGFAvg', 'AGAAvg', 'HSFAvg', 'ASAAvg']].to_numpy()
#Xa = data[['AGFAvg', 'HGAAvg', 'ASFAvg', 'HSAAvg']].to_numpy()
Xh = data[['HGFAvg', 'AGAAvg']].to_numpy()
Xa = data[['AGFAvg', 'HGAAvg']].to_numpy()
#Xh = data['HGFAvg'].to_numpy()
#Xa = data['AGFAvg'].to_numpy()
X = np.concatenate((Xh, Xa))

yh = data['FTHG'].to_numpy()
ya = data['HTAG'].to_numpy()
y = np.concatenate((yh, ya))

#scaler = MinMaxScaler()
#scaler = scaler.fit(X)
#X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


model = Sequential()
model.add(Dense(4, activation='relu', input_dim=2))
#model.add(Dropout(0.33))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='selu'))
model.compile(loss='mae', optimizer='rmsprop',  metrics=['mae'])
model.fit(X_train, y_train, epochs=10, verbose=1, validation_data=(X_test, y_test))
y_pred = model.predict(X_test)
pred_real = list(zip(y_pred, y_test))
print(pred_real[0:20])
