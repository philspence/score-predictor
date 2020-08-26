import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Concatenate, Input, BatchNormalization
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error as mse
from sklearn.utils.class_weight import compute_class_weight
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pickle


def build_model():
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=2))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model


def onehot_enc_goals(df, goals):
    goals = df[goals].to_numpy()
    goals_shortened = []
    for goal in goals:
        if goal > 3:
            goals_shortened.append(4)
        else:
            goals_shortened.append(goal)
    gs_np = np.array(goals_shortened).reshape(-1, 1)
    goals_cat = to_categorical(gs_np)
    return goals_cat


def ohe_to_goals(y):
    goals = []
    probs = []
    for i in y:
        match = np.where(i == np.amax(i))[0]
        goals.append(match)
        probs.append(np.amax(i))
    return np.array(goals), np.array(probs)


def get_data(infile):
    data = pd.read_csv(f'{infile}.csv', header=0)
    data.dropna(inplace=True)
    X_eloH = data[['HomeElo', 'AwayElo']]
    X_eloA = data[['AwayElo', 'HomeElo']]
    X_elo = np.concatenate((X_eloH, X_eloA), axis=0)
    scaler = preprocessing.MinMaxScaler()
    X_elo_norm = scaler.fit_transform(X_elo)
    yH = onehot_enc_goals(data, 'HomeGoals')
    yA = onehot_enc_goals(data, 'AwayGoals')
    y = np.concatenate((yH, yA), axis=0)
    return X_elo, y, data


def predictions(model, X, data, inf, y):
    y_pred = model.predict(X)
    goals, probs = ohe_to_goals(y_pred)
    data['HGPred'], data['AGPred'] = np.split(goals, 2)[0], np.split(goals, 2)[1]
    data['HGProb'], data['AGProb'] = np.split(probs, 2)[0], np.split(probs, 2)[1]
    data.to_csv(f'{inf}_preds.csv')
    print(data.head(20).to_string())
    print(data.tail(20).to_string())


def get_class_weights(y):
    y_int = [i.argmax() for i in y]
    weights = compute_class_weight('balanced', np.unique(y_int), y_int)
    weights_dict = dict(enumerate(weights))
    return weights_dict


def main(infile, m):
    X, y, data = get_data(infile)
    X_train, X_test = train_test_split(X, test_size=0.33, random_state=42)
    y_train, y_test = train_test_split(y, test_size=0.33, random_state=42)
    early_stopping = [EarlyStopping(monitor='val_loss', patience=5),
                      ModelCheckpoint(filepath=f'best_{m}.h5', monitor='val_loss',
                                      save_best_only=True)]
    class_weights = get_class_weights(y_train)
    # class_weights = {0: 0.57, 1: 0.49, 2: 0.76, 3: 1.66, 4: 4.27, 5: 8.87}
    cw = {0: 0.57, 1: 0.49, 2: 0.76, 3: 0.76, 4: 0.57, 5: 0.57}
    model = build_model()
    history = model.fit(X_train, y_train, epochs=100, verbose=1,
                        validation_data=(X_test, y_test), batch_size=10,
                        callbacks=early_stopping)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    model.save(f'{m}.h5')
    model.save_weights(f'{m}_weights.h5')
    predictions(model, X, data, infile, y)
    return


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', dest='infile', type=str, help='CSV file to load', required=True)
        parser.add_argument('-m', dest='m', type=str, help='Model name, no extension', required=True)
        args = parser.parse_args()  # when running the .py file, pass two parameters -f CSV_FILE (do not include .csv) and
        # -m MODEL_FILENAME, -m doesn't have to exist, this is where the model will be saved to
        infile = args.infile  # infile is now the name of the CSV
        m = args.m  # m is now the model filename
    except:
        infile = 'EPL_form_elo_API'
        m = 'score_cat_model_elo'
    finally:
        main(infile, m)  # run the function main() and give it infile and m
