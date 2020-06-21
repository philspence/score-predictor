import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import argparse


def build_model():
    opt = tf.keras.optimizers.RMSprop(momentum=0.15)
    model = Sequential()
    model.add(Dense(4, activation='relu', input_dim=2))
    # model.add(Dropout(0.33))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    # model.add(Dense(2, activation='relu'))
    # model.add(Dense(4, activation='relu'))
    # model.add(Dense(8, activation='relu'))
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


def main(inf, m_name):
    X, y = load_data(inf)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = build_model()
    model.fit(X_train, y_train, epochs=5, verbose=1, validation_data=(X_test, y_test))
    model.save(m_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='infile', type=str, help='CSV file to load', required=True)
    parser.add_argument('-m', dest='m', type=str, help='Model name', required=False, default='avgG_model')
    args = parser.parse_args()
    infile = args.infile
    m = args.m
    main(infile, m)
