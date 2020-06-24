import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
import pandas as pd
import argparse
from sklearn.preprocessing import MinMaxScaler, Normalizer


def build_model():
    opt = tf.keras.optimizers.RMSprop(momentum=0.0)
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=1))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='nadam',  metrics=['mae', 'mse'])
    return model


def load_data(infile):
    data = pd.read_csv(f'{infile}.csv', header=0)
    # df = data[['HomeStats', 'AwayStats', 'FTHG', 'FTAG']]
    Xh = data['HStats']
    Xa = data['AStats']
    X = np.concatenate((Xh, Xa))
    X = X.reshape(-1, 1)
    trans_x = MinMaxScaler().fit(X)
    norm_X = trans_x.transform(X)
    print(norm_X)
    yh, ya = data['FTHG'], data['FTAG']
    y = np.concatenate((yh, ya))
    # trans_y = MinMaxScaler().fit(y)
    # norm_y = trans_y.transform(y)
    # norm_y = norm_y.reshape(-1,)
    return norm_X, y, data


def main(inf, m_name):
    X, y, data = load_data(inf)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = build_model()
    model.fit(X_train, y_train, epochs=5, verbose=1, validation_data=(X_test, y_test))
    y_pred = model.predict(X)
    # y_pred = y_pred.reshape(-1, 1)
    # unnorm_y = trans.inverse_transform(y_pred)
    # y_pred = unnorm_y.reshape(-1,)
    yH, yA = np.split(y_pred, 2)[0], np.split(y_pred, 2)[1]
    data['H_Pred'], data['A_Pred'] = yH, yA
    data.to_csv(f'{inf}_preds.csv')
    print(data.head(10))
    print(data.tail(10))
    model.save(m_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='infile', type=str, help='CSV file to load', required=True)
    parser.add_argument('-m', dest='m', type=str, help='Model name', required=False, default='avgG_model')
    args = parser.parse_args()
    infile = args.infile
    m = args.m
    main(infile, m)
