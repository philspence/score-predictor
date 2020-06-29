import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import pickle


def build_model():
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.01)  # LR default = 0.001, Mom=0.0
    model = Sequential()
    model.add(Dense(8, activation='relu', input_dim=2))
    # model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', 'categorical_accuracy'])
    return model


def build_forest_model(Xtrain, ytrain, Xtest, ytest):
    model = RandomForestClassifier(n_jobs=-1, n_estimators=12)
    print('Fitting model')
    model.fit(Xtrain, ytrain)
    y_pred = model.predict(Xtest)
    print(model.score(Xtest, ytest))


def onehot_enc_goals(df, goals):
    return to_categorical(df[goals])


def ohe_to_goals(y):
    goals = []
    probs = []
    for i in y:
        match = np.where(i == np.amax(i))[0]
        goals.append(match)
        probs.append(np.amax(i))
    return np.array(goals), np.array(probs)


def predict_forest(rf, X, inf, data):
    print('Predicting data')
    y_pred = rf.predict(X)
    data['HGPred'], data['AGPred'] = y_pred[:, 0], y_pred[:, 1]
    data['HGProb', 'AGProb'] = y_prob[:, 0], y_prob[:, 1]
    print(data.head(20).to_string())
    print(data.tail(20).to_string())


def get_data(infile):
    data = pd.read_csv(f'{infile}.csv', header=0, index_col=0)
    XH = data[['HWinPerc', 'AWinPerc']]
    XA = data[['AWinPerc', 'HWinPerc']]
    X = np.concatenate((XH, XA), axis=0)
    # X = data[['HGFAvg', 'AGAAvg', 'AGFAvg', 'HGAAvg', 'HWPred', 'AWPred', 'DPred']]
    yH = onehot_enc_goals(data, 'FTHG')
    yA = onehot_enc_goals(data, 'FTAG')
    y = np.concatenate((yH, yA), axis=0)
    return X, y, data


def predictions(model, X, data, inf, y):
    y_pred = model.predict(X)
    goals, probs = ohe_to_goals(y_pred)
    data['HGPred'], data['AGPred'] = np.split(goals, 2)[0], np.split(goals, 2)[1]
    data['HGProb'], data['AGProb'] = np.split(probs, 2)[0], np.split(probs, 2)[1]
    data.to_csv(f'{inf}_preds.csv')
    print(data.head(20).to_string())
    print(data.tail(20).to_string())


def main(infile, m):
    X, y, data = get_data(infile)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    early_stopping = [EarlyStopping(monitor='val_loss', patience=5),
                      ModelCheckpoint(filepath='best_score_cat_model.h5', monitor='val_loss',
                                      save_best_only=True)]
    # build_forest_model(X_train, y_train, X_test, y_test)
    model = build_model()
    model.fit(X_train, y_train, epochs=50, verbose=1, validation_data=(X_test, y_test), batch_size=10,
              callbacks=early_stopping)
    predictions(model, X, data, infile, y)
    model.save(f'{m}.h5')
    model.save_weights(f'{m}_weights.h5')


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', dest='infile', type=str, help='CSV file to load', required=True)
        parser.add_argument('-m', dest='m', type=str, help='Model name, no extension', required=False,
                            default='score_model')
        args = parser.parse_args()  # when running the .py file, pass two parameters -f CSV_FILE (do not include .csv) and
        # -m MODEL_FILENAME, -m doesn't have to exist, this is where the model will be saved to
        infile = args.infile  # infile is now the name of the CSV
        m = args.m  # m is now the model filename
    except:
        infile = 'top_leagues/EPL_merged_avgs_shots'
        m = 'score_cat_model'
    finally:
        main(infile, m)  # run the function main() and give it infile and m