import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler


def get_mv_avg(df, key, value, win, newcol):
    moving_avg = df.groupby(key)[value].rolling(window=win).mean().reset_index()
    df[newcol] = moving_avg.set_index('level_1')[value]
    return df


def calc_form(infile, w):
    data = pd.read_csv(f'{infile}.csv', header=0)
    print('Imported CSV')
    data = data[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
    data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
    #data = data.loc[data.Date > '01/01/2000']
    data = data.dropna()
    print('Removed excess data')
    
    # HGFAvg = Home Goals For Average, S = Shots
    data = get_mv_avg(data, 'HomeTeam', 'FTHG', w, 'HGFAvg')
    data = get_mv_avg(data, 'HomeTeam', 'FTAG', w, 'HGAAvg')
    #data = get_mv_avg(data, 'HomeTeam', 'HS', w, 'HSFAvg')
    #data = get_mv_avg(data, 'HomeTeam', 'AS', w, 'HSAAvg')

    data = get_mv_avg(data, 'AwayTeam', 'FTAG', w, 'AGFAvg')
    data = get_mv_avg(data, 'AwayTeam', 'FTHG', w, 'AGAAvg')
    #data = get_mv_avg(data, 'AwayTeam', 'AS', w, 'ASFAvg')
    #data = get_mv_avg(data, 'AwayTeam', 'HS', w, 'ASAAvg')
    data = data.dropna(subset=['HGFAvg', 'HGAAvg', 'AGFAvg', 'AGAAvg'])
    data.to_csv(f'{infile}-avgs.csv')
    return data
    
    
def load_form(infile):
    data = pd.read_csv(f'{infile}.csv', header=0)
    return data
# removes the first 4 games for each team (i.e. no form to calculate with)

def get_training_data(infile, load=True):
    if load is True:
        data = load_form(f'{infile}-avgs')
    else:
        data = calc_form(infile, 3)
    
    #Xh = data[['HGFAvg', 'AGAAvg', 'HSFAvg', 'ASAAvg']].to_numpy()
    #Xa = data[['AGFAvg', 'HGAAvg', 'ASFAvg', 'HSAAvg']].to_numpy()
    Xh = data[['HGFAvg', 'AGAAvg']].to_numpy()
    Xa = data[['AGFAvg', 'HGAAvg']].to_numpy()
    X = np.concatenate((Xh, Xa))

    yh = data['FTHG'].to_numpy()
    ya = data['FTAG'].to_numpy()
    y = np.concatenate((yh, ya))
    #scaler = MinMaxScaler()
    #scaler = scaler.fit(X)
    #X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.33, random_state=42)

def build_model():
    opt = tf.keras.optimizers.RMSprop(momentum=0.15)
    model = Sequential()
    model.add(Dense(4, activation='relu', input_dim=2))
    #model.add(Dropout(0.33))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    #model.add(Dense(2, activation='relu'))
    #model.add(Dense(4, activation='relu'))
    #model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='selu'))
    model.compile(loss='mae', optimizer='rmsprop',  metrics=['mae'])
    return model


def load_data(infile):
    data = pd.read_csv(f'{infile}.csv', header=0)
    Xh = data[['HGFAvg', 'AGAAvg']].to_numpy()
    Xa = data[['AGFAvg', 'HGAAvg']].to_numpy()
    X = np.concatenate((Xh, Xa))

    yh = data['FTHG'].to_numpy()
    ya = data['FTAG'].to_numpy()
    y = np.concatenate((yh, ya))
    
    return X, y


train = False
if train is True:
    X_train, X_test, y_train, y_test = get_training_data('merged-all-avgs')
else:
    data = calc_form('merged-EPL', 3)
    X, y = load_data('merged-EPL-avgs')
if os.path.isfile('avgG-model-06'):
    model = load_model('avgG-model-06')
else:
    model = build_model()
    model.fit(X_train, y_train, epochs=5, verbose=1, validation_data=(X_test, y_test))
    model.save('avgG-model')

    
y_pred = model.predict(X)
yH, ya = np.split(y_pred, 2)[0], np.split(y_pred, 2)[1]
data['PredHG'],  data['PredAG'] = [np.round(yH, 1), np.round(ya, 1)]
data.to_csv('merged-EPL-preds.csv')
exit()
y_test_np = np.array([y_test])
y_test_np = np.reshape(y_test_np, (-1, 1))
pred_real = np.append(y_pred, y_test_np, axis=1)
print(np.round(pred_real, 1)[0:20])
