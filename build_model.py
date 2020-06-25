import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import argparse
from sklearn.preprocessing import MinMaxScaler, Normalizer


def build_model():
    opt = tf.keras.optimizers.RMSprop(momentum=0.0)
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=4))
    # model.add(Dropout(0.2))
    # model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='selu'))
    model.compile(loss='mse', optimizer='rmsprop',  metrics=['accuracy'])
    return model


def load_data(infile):
    data = pd.read_csv(f'{infile}.csv', header=0)
    # df = data[['HomeStats', 'AwayStats', 'FTHG', 'FTAG']]
    Xh = data[['HGFAvg', 'AGAAvg', 'HSFAvg', 'ASAAvg', 'HPMean', 'APMean']]
    Xa = data[['AGFAvg', 'HGAAvg', 'ASFAvg', 'HSAAvg', 'APMean', 'HPMean']]
    X = np.concatenate((Xh, Xa))
    X = X.reshape(-1, 4)
    trans_x = MinMaxScaler().fit(X)
    norm_X = trans_x.transform(X)
    norm_X = norm_X.reshape(-1, 4)
    # print(norm_X)
    yh, ya = data['FTHG'], data['FTAG']
    y = np.concatenate((yh, ya))
    y = y.reshape(-1, 1)
    trans_y = MinMaxScaler().fit(y)
    norm_y = trans_y.transform(y)
    return norm_X, norm_y, data, trans_y


def main(inf, m_name):
    X, y, data, trans = load_data(inf)
    early_stopping = [EarlyStopping(monitor='val_loss', patience=3),
                      ModelCheckpoint(filepath=os.path.join('best_avgG_model.h5'), monitor='val_loss',
                                      save_best_only=True)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = build_model()
    model.fit(X_train, y_train, epochs=50, verbose=1, validation_data=(X_test, y_test), batch_size=10,
              callbacks=early_stopping)
    y_pred = model.predict(X)
    y_pred = y_pred.reshape(-1, 1)
    unnorm_y = trans.inverse_transform(y_pred)
    y_pred = unnorm_y.reshape(-1,)
    yH, yA = np.split(y_pred, 2)[0], np.split(y_pred, 2)[1]
    data['H_Pred'], data['A_Pred'] = yH, yA
    data.to_csv(f'{inf}_preds.csv')
    print(data.head(10))
    print(data.tail(10))
    model.save(m_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='infile', type=str, help='CSV file to load', required=True)
    parser.add_argument('-m', dest='m', type=str, help='Model name', required=False, default='avgG_model.h5')
    args = parser.parse_args()
    infile = args.infile
    m = args.m
    main(infile, m)
