import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# uncomment the above line if you want to run it on the CPU, otherwise it will run on the GPU
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pickle


def build_model():
    opt = tf.keras.optimizers.RMSprop(momentum=0.01)
    model = Sequential()  # sequential model
    model.add(Dense(4, activation='relu', input_dim=4))  # Dense layer of 16 nodes, activation used is relu
    # input_dim is the size of the array that is being passed, X is an array of width 2 (2 parameters), so this is set
    # to 6 if X changes to an array of 4 parameters, this needs to be change to 4, etc.
    model.add(Dense(6, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # output has a shape of (-1, 2), -1 is just the length of the array i.e.
    # how many results you are predicting and 2 is the width, i.e. [[1, 0] [1, 0], [0, 1], [0, 0]]
    # would be the outcome of 4 results that represents ['H', 'H', 'A', 'D'] and this would have the shape (4, 2),
    # Keras doesn't care about the first value in the shape (how many results you are predicting) it only cares about
    # the second, which is 2
    model.compile(loss='categorical_crossentropy', optimizer=opt,  metrics=['categorical_accuracy'])
    # compile the model, using caterogical_crossentropy as the loss
    # (monitors how accurate the model is, lower is better), adam as the optimiser and then outputs
    # caterogical_accuracy for you to see as it trains (lower is better)
    print('Saving model')
    with open('knn_cat.pickle', 'wb') as p:
        pickle.dump(model, p)
    return model


def load_data(infile):
    data = pd.read_csv(f'{infile}.csv', header=0, index_col=0)
    # load the CSV of the data as a dataframe, uses the first row as the headers for each column
    X = data[['HWinPerc', 'AWinPerc', 'HP5Avg', 'AP5Avg']].to_numpy()  # use the headers to extract the correct columns...
    # average home points, average away points, the "input_dim" for the first layer of the model should reflect how
    # many parameters are entered here, right now this is 2
    y = data[['HW', 'AW', 'D']].to_numpy()  # this is the 2D matrix, a home win = [1, 0], away win = [0, 1] the model
    # will try to fit values to each of these and should come out with a prediction, the values of prediction aren't
    # integers so a draw could be considered [0.5, 0.5] or perhaps from [0.6, 0.4] to [0.4, 0.6]
    return X, y, data


def predict_forest(rf, X, inf, data):
    print('Predicting data')
    # y_pred = rf.predict(X)
    # data['HWPred'], data['AWPred'], data['DPred'] = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    y_pred_prob = rf.predict_proba(X)
    data['HWProb'], data['AWProb'], data['DProb'] = y_pred_prob[0][:, 1], y_pred_prob[1][:, 1], y_pred_prob[2][:, 1]
    data.to_csv(f'{inf}.csv')
    print(data.head(20).to_string())
    print(data.tail(20).to_string())


def main(inf, m_name):
    early_stopping = [EarlyStopping(monitor='val_loss', patience=3),
                      ModelCheckpoint(filepath='cat_best_model.h5', monitor='val_loss',
                                      save_best_only=True)]
    X, y, data = load_data(inf)  # runs the function load_data() and gives it inf, which is the infile name of the csv
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)  # split the data into
    # training and testing, 33% will be testing and 66% for training, random state just ensures a random split of the
    # sample data each time
    model = build_model()  # runs the function build_model and returns the model
    model.fit(X_train, y_train, epochs=50, verbose=1, validation_data=(X_test, y_test), batch_size=10,
              callbacks=early_stopping)
    # fit the model to the training dataset X and y, validate it with testing X and y.
    # Epochs is how many times it iterates over itself this can be trained
    y_pred = model.predict(X)  # put all of X backthrough it and get the predictions
    # split the array of y_pred into H and A, i.e. takes [[1, 0], [0, 1], [1, 0]] and takes the first value of each
    # prediction (1, 0 and 1, in this case) and makes that
    # into one array [1, 0, 1] this is added as a column named HWPred and then does the same with AWPred using the
    # second value of each prediction (0, 1, 0)
    data['HWProb'], data['AWProb'], data['DProb'] = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    print(data.head(20).to_string())  # prints the first 20 and then the last 20 rows of the dataframe
    print(data.tail(20).to_string())
    data.to_csv(f'{inf}.csv')
    model.save(f'{m_name}.h5')  # saves the model using the -m value given when running the script
    model.save_weights(f'{m_name}_weights.h5')


if __name__ == '__main__':  # checks if this was ran using "python build_model_classifier.py" or whether it was imported
    # by another python script, this section will only run if it is run from the commandline i.e. with
    # "python build_model_classifier.py
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', dest='infile', type=str, help='CSV file to load', required=True)
        parser.add_argument('-m', dest='m', type=str, help='Model name', required=False, default='cat_model')
        args = parser.parse_args()  # when running the .py file, pass two parameters -f CSV_FILE (do not include .csv) and
        # -m MODEL_FILENAME, -m doesn't have to exist, this is where the model will be saved to
        infile = args.infile  # infile is now the name of the CSV
        m = args.m  # m is now the model filename
    except:
        infile = 'top_leagues/EPL_merged_avgs'
        m = 'cat_model'
    finally:
        main(infile, m)  # run the function main() and give it infile and m
