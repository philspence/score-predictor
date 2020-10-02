import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import GradientBoostingClassifier
import requests
import json
import pickle
from pathlib import Path


def build_model():
    model = GradientBoostingClassifier(n_estimators=10, random_state=42, max_depth=64, verbose=1, learning_rate=0.1,
                                       criterion='mse')
    return model


def get_train_data(infile):
    data = pd.read_csv(f'{infile}.csv', header=0)
    data.dropna(inplace=True)
    X_loc_formH = data[['HomeTeamHomeForm', 'AwayTeamAwayForm']].to_numpy()
    X_formH = data[['HomeTeamForm', 'AwayTeamForm']].to_numpy()
    X_seasonH = data[['HomeTeamSeasonWinPerc', 'AwayTeamSeasonWinPerc']].to_numpy()
    X_loc_formA = data[['AwayTeamAwayForm', 'HomeTeamHomeForm']].to_numpy()
    X_formA = data[['AwayTeamForm', 'HomeTeamForm']].to_numpy()
    X_seasonA = data[['AwayTeamSeasonWinPerc', 'HomeTeamSeasonWinPerc']].to_numpy()
    X_eloH = data[['HomeElo', 'AwayElo']].to_numpy()
    X_eloA = data[['AwayElo', 'HomeElo']].to_numpy()
    X_goalsH = data[['HomeScored', 'AwayConceded']].to_numpy()
    X_goalsA = data[['AwayScored', 'HomeConceded']].to_numpy()
    # concat all three
    X_loc_form = np.concatenate((X_loc_formH, X_loc_formA), axis=0)
    X_form = np.concatenate((X_formH, X_formA), axis=0)
    X_season = np.concatenate((X_seasonH, X_seasonA), axis=0)
    X_elo = np.concatenate((X_eloH, X_eloA), axis=0)
    X_goals = np.concatenate((X_goalsH, X_goalsA), axis=0)
    # yH = onehot_enc_goals(data, 'HomeGoals')
    # yA = onehot_enc_goals(data, 'AwayGoals')
    yH = data[['HomeGoals']].to_numpy()
    yA = data[['AwayGoals']].to_numpy()
    y = np.concatenate((yH, yA), axis=0)
    X = np.hstack((X_form, X_elo, X_season, X_goals))
    return X, y, data


def get_class_weights(y):
    y_int = [i.argmax() for i in y]
    weights = compute_class_weight('balanced', np.unique(y_int), y_int)
    weights_dict = dict(enumerate(weights))
    return weights_dict


def train_model(infile):
    X, y, data = get_train_data(infile)
    X_train, X_test = train_test_split(X, test_size=0.25, random_state=42)
    y_train, y_test = train_test_split(y, test_size=0.25, random_state=42)
    model = build_model()
    # all 4 model used [X_loc_train, X_form_train, X_season_train, X_elo_train]
    print('Fitting Data')
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(score)
    y_pred = model.predict(X_test).reshape(-1, 1)
    validate = np.hstack((y_test, y_pred))
    pickle.dump(model, Path('boost_classifier.pickle').open(mode='wb'))
    return


def pull_data(gwk):
    next_fix = requests.get(f'https://api.teamto.win/v1/gameweekFixtures.php?gameweek={gwk}').json()
    # teams = requests.get('https://api.teamto.win/v1/listOfTeams.php?').json()
    home_teams = []
    away_teams = []
    for fix in next_fix:
        home_teams.append(next_fix[fix]['home_team_id'])
        away_teams.append(next_fix[fix]['away_team_id'])
    return home_teams, away_teams, next_fix


def calc_goals(tms, ha, prop):
    avg_goals = []
    for t in tms:
        form = requests.get(f'https://api.teamto.win/v1/teamForm.php?team_id={t}&fixtures=5').json()
        goals = 0
        for match in form[ha]:
            if match == 'form':
                continue
            else:
                goals += int(form[ha][match][prop])
        avg_goals.append(goals / 3)
    np_arr = np.array([avg_goals])
    np_arr = np_arr.reshape(-1, 1)
    return np_arr


def get_x(home, away):
    home_loc_form, home_form, home_season = calc_form(home, 'home_form')
    away_loc_form, away_form, away_season = calc_form(away, 'away_form')
    home_elo = get_elo(home)
    away_elo = get_elo(away)
    home_goals_for = calc_goals(home, 'home_form', 'goals_scored')
    home_goals_against = calc_goals(home, 'home_form', 'goals_conceded')
    away_goals_for = calc_goals(away, 'away_form', 'goals_scored')
    away_goals_against = calc_goals(away, 'away_form', 'goals_conceded')
    home_goals = np.hstack((home_goals_for, away_goals_against))
    away_goals = np.hstack((away_goals_for, home_goals_against))
    loc_form = stack_concat(home_loc_form, away_loc_form)
    form = stack_concat(home_form, away_form)
    season = stack_concat(home_season, away_season)
    elo = stack_concat(home_elo, away_elo)
    goals = np.concatenate((home_goals, away_goals), axis=0)
    return np.hstack((form, elo, season, goals))


def stack_concat(arr1, arr2):
    arr12 = np.hstack((arr1, arr2))
    arr21 = np.hstack((arr2, arr1))
    arr = np.concatenate((arr12, arr21), axis=0)
    return arr


def get_elo(teams):
    elo_ratings = []
    for team in teams:
        team_info = requests.get(f'https://api.teamto.win/v1/teamInfo.php?team_id={team}').json()
        elo = float(team_info['elo'])
        elo_ratings.append(elo)
    return np.array(elo_ratings).reshape(-1, 1)


def calc_form(tms, ha):
    wld_dict = {'W': 1, 'L': 0, 'D': 0}
    loc_wins = []
    form_wins = []
    season_wins = []
    for t in tms:
        form = requests.get(f'https://api.teamto.win/v1/teamForm.php?team_id={t}&fixtures=3').json()
        loc_win_avg = 0
        for match in form[ha]['form']:
            val = form[ha]['form'][match]
            loc_win_avg += wld_dict[val]
        loc_wins.append(loc_win_avg / 3)
        num = 0
        wins = 0
        while num < 5:
            for match in form['all_form']:
                if form['all_form'][match] == 'W':
                    wins += 1
                num += 1
        form_wins.append(wins / 5)
        win_ratio = (form['season_form']['home']['win_percentage'] + form['season_form']['away']['win_percentage']) / 2
        season_wins.append(win_ratio / 100)
    loc_wins = np.array(loc_wins).reshape(-1, 1)
    form_wins = np.array(form_wins).reshape(-1, 1)
    season_wins = np.array(season_wins).reshape(-1, 1)
    return loc_wins, form_wins, season_wins


def post_data(nf,  s_pred, s_prob):
    # oH, oA, oD = o_pred[:, 0], o_pred[:, 1], o_pred[:, 2]
    sH, sA = np.split(s_pred, 2)[0], np.split(s_pred, 2)[1]
    spH, spA = np.split(s_prob, 2)[0], np.split(s_prob, 2)[1]
    # p_arrH, p_arrA = np.split(p_arr, 2)[0], np.split(p_arr, 2)[1]
    num = 0
    for fix in nf:
        # nf[fix]['home_percentage'] = round(oH[num].item() * 100, 2)
        # nf[fix]['away_percentage'] = round(oA[num].item() * 100, 2)
        # nf[fix]['draw_percentage'] = round(oD[num].item() * 100, 2)
        nf[fix]['predicted_home_goals'] = round(sH[num].item(), 2)
        nf[fix]['predicted_away_goals'] = round(sA[num].item(), 2)
        nf[fix]['probability_home_goals'] = round(spH[num][sH[num]].item(), 2)
        nf[fix]['probability_away_goals'] = round(spA[num][sA[num]].item(), 2)
        nf[fix]['percentage'] = round((spH[num][sH[num]].item() * spA[num][sA[num]].item()) * 100, 2)
        num += 1
        print(f'{nf[fix]["home_team_name"]} {nf[fix]["predicted_home_goals"]} vs '
              f'{nf[fix]["predicted_away_goals"]} {nf[fix]["away_team_name"]}')
    with open('predictions.json', 'w') as f:
        json.dump(nf, f)
    # print(nf)
    # requests.post('https://api.teamto.win/v1/savePrediction.php', json.dumps(nf))


def make_prediction(gwk):
    home_teams, away_teams, next_fix = pull_data(gwk)
    score_X = get_x(home_teams, away_teams)
    score_pred, score_prob = predict_score(score_X)
    post_data(next_fix, score_pred, score_prob)
    return


def predict_score(X):
    model = pickle.load(Path('boost_classifier.pickle').open(mode='rb'))
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    return y_pred, y_prob


def score_predictor(train=False, predict=True):
    infile = 'EPL_form_elo_API'
    gwk = 4
    if train is True:
        train_model(infile)
    if predict is True:
        make_prediction(gwk)


if __name__ == '__main__':
    score_predictor(train=False, predict=True)
