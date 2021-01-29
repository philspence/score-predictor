from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
import requests
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf


devices = tf.config.list_physical_devices()
tf.config.experimental.set_visible_devices([devices[0], devices[2]])
tf.config.experimental.set_memory_growth(devices[2], enable=True)

n_fixtures = 5
n_features = 10


class Team:
    def __init__(self, ident, name):
        self.id = ident
        self.name = name
        self.elo_rating = 0
        self.fixtures = list()
        return


class Fixture:
    def __init__(self, ident, date, season, home_team, away_team, home_goals, away_goals, home_elo, away_elo):
        self.id = ident
        self.date = date
        self.season = season
        self.home_team = home_team
        self.away_team = away_team
        self.home_goals = home_goals
        self.away_goals = away_goals
        self.home_elo = home_elo
        self.away_elo = away_elo
        return


class Season:
    def __init__(self, ident, year_start, year_end):
        self.id = ident
        self.year_start = year_start
        self.year_end = year_end
        self.fixtures = list()
        return


class Predictor:
    def __init__(self):
        self.model = many_to_one()
        return

    def train_model(self, x, y):
        weights = get_class_weights(y)
        x, x_val, y, y_val = train_test_split(x, y, test_size=0.10, random_state=42, shuffle=False)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, shuffle=True)
        print('Fitting...')
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=10,
                       class_weight=weights)
        print('Done.')
        return x_val, y_val

    def predict(self, x_val, y_val):
        preds = self.model.predict(x_val)
        preds = ohe_to_int(preds)
        y_val = ohe_to_int(y_val)
        mse = mean_squared_error(y_val.reshape(-1, 1), preds.reshape(-1, 1))
        mae = mean_absolute_error(y_val.reshape(-1, 1), preds.reshape(-1, 1))
        print(f'MAE: {round(mae, 5)}\nMSE: {round(mse, 5)}')
        return preds


def get_class_weights(y):
    int_y = ohe_to_int(y).reshape(-1,)
    weights = compute_class_weight('balanced', np.unique(int_y), int_y)
    class_weights = dict(enumerate(weights))
    return class_weights


def ohe_to_int(preds):
    goals = np.empty(0, )
    for p in preds:
        i = np.where(p == p.max())[0][0]
        goals = np.append(goals, i)
    return goals.reshape(-1, 1)


def create_teams(load=False):
    if load:
        teams = pickle.load(Path('teams.pickle').open(mode='rb'))
    else:
        url = "https://api.teamto.win/v1/listOfTeams.php"
        data = requests.get(url).json()
        teams = dict()
        for team_id in data:
            teams[int(team_id)] = Team(int(data[team_id]['id']), data[team_id]['name'])
        pickle.dump(teams, Path('teams.pickle').open(mode='wb'))
    return teams


def add_fixtures_to_teams(teams):
    seasons = dict()
    for i in range(1, 105):
        url = f'https://api.teamto.win/v1/allFixtures.php?page={i}'
        data = requests.get(url).json()
        for fix_id in data:
            print(fix_id)
            fixture = Fixture(int(fix_id), data[fix_id]['date'], data[fix_id]['season_id'],
                              int(data[fix_id]['home_team_id']), int(data[fix_id]['away_team_id']),
                              float(data[fix_id]['FTHomeGoals']), float(data[fix_id]['FTAwayGoals']),
                              float(data[fix_id]['home_team_elo']), float(data[fix_id]['away_team_elo']))
            teams[int(data[fix_id]['home_team_id'])].fixtures.append(fixture)
            teams[int(data[fix_id]['away_team_id'])].fixtures.append(fixture)
            try:
                seasons[data[fix_id]['season_id']].fixtures.append(fixture)
            except KeyError:
                year_start, year_end = data[fix_id]['season_id'].split('_')
                seasons[data[fix_id]['season_id']] = Season(int(year_start), int(year_end))
                seasons[data[fix_id]['season_id']].fixtures.append(fixture)
    pickle.dump(teams, Path('teams.pickle').open(mode='wb'))
    pickle.dump(seasons, Path('seasons.pickle').open(mode='wb'))
    return seasons


def load_seasons():
    seasons = pickle.load(Path('seasons.pickle').open(mode='rb'))
    return seasons


def pull_training_data(teams):
    print('Getting training data')
    x = np.empty(0, )
    y = np.empty(0, )
    for team_id, team in teams.items():
        # for each team in the teams list
        for fixture in team.fixtures:
            # for each fixture in their fixtures list
            home_scored, home_conceded, home_home_elo, home_away_elo, away_scored, away_conceded, \
            away_away_elo, away_home_elo, home_home, away_home = list(), list(), list(), list(), list(), list(), \
                                                                 list(), list(), list(), list()
            # create empty lists to store data
            if fixture.home_team == team_id:
                # if the team is the home team then carry on, this prevents duplication in fixtures
                home_loc = team.fixtures.index(fixture)
                away_loc = teams[fixture.away_team].fixtures.index(fixture)
                # find location of the fixture in the fixtures index
                home_fixture_counter = 0
                away_fixture_counter = 0
                home_fixtures_to_use = list()
                away_fixtures_to_use = list()
                # create variables to be used
                for historical_fixture in team.fixtures[0:home_loc][::-1]:
                    # go back through the fixtures from the location we are at
                    if home_fixture_counter == 5:
                        break
                    if historical_fixture.home_team == team_id or historical_fixture.away_team == team_id \
                            and historical_fixture.season == fixture.season:
                        # if the home team features in the any of these fixtures (as H or A team) and it is in the same
                        # season then add it to the list and add 1 to the counter
                        home_fixtures_to_use.append(historical_fixture)
                        home_fixture_counter += 1
                if home_fixture_counter < 5:
                    continue
                for historical_fixture in teams[fixture.away_team].fixtures[0:away_loc][::-1]:
                    if away_fixture_counter == 5:
                        break
                    if historical_fixture.home_team == fixture.away_team or \
                            historical_fixture.away_team == fixture.away_team and \
                            historical_fixture.season == fixture.season:
                        # same as above but the away team in the fixture
                        away_fixtures_to_use.append(historical_fixture)
                        away_fixture_counter += 1
                if away_fixture_counter < 5:
                    continue
                # arriving here you should have the previous 5 fixtures of both the H and A team, if not, then they
                # don't exist so need to continue the loop and go to the next fixture, this happens when looking at the
                # first 5 games in a season, these get ignored
                for f in home_fixtures_to_use[::-1]:
                    # need to iterate backwards as they were appended backwards, so they are now forwards
                    if f.home_team == team_id:
                        # if the home team of the fixture is the team we are dealing with
                        home_scored.append(normalise_goals(f.home_goals))
                        home_conceded.append(normalise_goals(f.away_goals))
                        # home_home_elo = the elo of the home team in the pred fixture
                        home_home_elo.append(normalise_elo(f.home_elo))
                        # home_away_elo = the elo of the opposing team in the historical fixture
                        home_away_elo.append(normalise_elo(f.away_elo))
                        home_home.append(1.)
                    elif f.away_team == team_id:
                        # if the team is the away then append the opposite data to home_scored etc.
                        # home_scored etc, means the home team that is playing for the fixture to predict, not the team
                        # in this fixture that is at home
                        home_scored.append(normalise_goals(f.away_goals))
                        home_conceded.append(normalise_goals(f.home_goals))
                        home_home_elo.append(normalise_elo(f.away_elo))
                        home_away_elo.append(normalise_elo(f.home_elo))
                        home_home.append(0.)
                for f in away_fixtures_to_use[::-1]:
                    if f.home_team == fixture.away_team:
                        # if the away team for the predictive fixture is playing at home, then...
                        away_scored.append(normalise_goals(f.home_goals))
                        away_conceded.append(normalise_goals(f.away_goals))
                        # away_away_elo = the elo of the away team in predictive fixture
                        away_away_elo.append(normalise_elo(f.home_elo))
                        # away_home_elo = the opposing teams elo in the historical fixture
                        away_home_elo.append(normalise_elo(f.away_elo))
                        away_home.append(1.)
                    elif f.away_team == fixture.away_team:
                        # if they are the away team then...
                        away_scored.append(normalise_goals(f.away_goals))
                        away_conceded.append(normalise_goals(f.home_goals))
                        away_away_elo.append(normalise_elo(f.away_elo))
                        away_home_elo.append(normalise_elo(f.home_elo))
                        away_home.append(0.)
                # append the predictions for the home team
                x_to_append = np.column_stack([home_home_elo, home_scored, home_conceded, home_away_elo, home_home,
                                               away_away_elo, away_scored, away_conceded, away_home_elo, away_home]).reshape(1, n_fixtures, n_features)
                y_to_append = float(fixture.home_goals)
                x = np.append(x, x_to_append).reshape(-1, n_fixtures, n_features)
                y = np.append(y, goals_to_ohe(y_to_append)).reshape(-1, 10)
                # append the predictions for the away team
                x_to_append = np.column_stack([away_away_elo, away_scored, away_conceded, away_home_elo, away_home,
                                               home_home_elo, home_scored, home_conceded, home_away_elo, home_home]).reshape(1, n_fixtures, n_features)
                y_to_append = float(fixture.away_goals)
                x = np.append(x, x_to_append).reshape(-1, n_fixtures, n_features)
                y = np.append(y, goals_to_ohe(y_to_append)).reshape(-1, 10)
    print('Done.')
    return x, y


def normalise_elo(elo):
    normalised = (elo - 1300) / (2200 - 1300)
    return normalised


def normalise_goals(g):
    normalised = (g - 0) / (9 - 0)
    return normalised


def goals_to_ohe(goals):
    ohe = to_categorical(goals, num_classes=10)
    return ohe


def many_to_one():
    model = Sequential()
    model.add(LSTM(n_features * 2, activation='tanh', return_sequences=True, input_shape=(n_fixtures, n_features)))
    model.add(LSTM(n_features, activation='tanh', return_sequences=True))
    model.add(LSTM(n_features, activation='tanh', return_sequences=False))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model


def main():
    teams = create_teams(load=True)
    seasons = load_seasons()
    x, y = pull_training_data(teams)
    model = Predictor()
    x_val, y_val = model.train_model(x, y)
    predictions = model.predict(x_val, y_val)
    return


if __name__ == '__main__':
    main()
