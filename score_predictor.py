import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import GradientBoostingClassifier
from itertools import product
import requests
import json
import pickle
from pathlib import Path


def build_model():
    model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=256, verbose=1, learning_rate=0.1,
                                       criterion='mse')
    return model


def get_train_data(infile, elo_only=False):
    data = pd.read_csv(f'{infile}.csv', header=0)
    data.dropna(inplace=True)
    loc_formH = data[['HomeTeamHomeForm', 'AwayTeamAwayForm']].to_numpy()
    formH = data[['HomeTeamForm', 'AwayTeamForm']].to_numpy()
    seasonH = data[['HomeTeamSeasonWinPerc', 'AwayTeamSeasonWinPerc']].to_numpy()
    loc_formA = data[['AwayTeamAwayForm', 'HomeTeamHomeForm']].to_numpy()
    formA = data[['AwayTeamForm', 'HomeTeamForm']].to_numpy()
    seasonA = data[['AwayTeamSeasonWinPerc', 'HomeTeamSeasonWinPerc']].to_numpy()
    eloH = data[['HomeElo', 'AwayElo']].to_numpy()
    eloA = data[['AwayElo', 'HomeElo']].to_numpy()
    goalsH = data[['HomeScored', 'AwayConceded']].to_numpy()
    goalsA = data[['AwayScored', 'HomeConceded']].to_numpy()
    # concat all three
    loc_form = np.concatenate((loc_formH, loc_formA), axis=0)
    form = np.concatenate((formH, formA), axis=0)
    season = np.concatenate((seasonH, seasonA), axis=0)
    elo = np.concatenate((eloH, eloA), axis=0)
    goals = np.concatenate((goalsH, goalsA), axis=0)
    # yH = onehot_enc_goals(data, 'HomeGoals')
    # yA = onehot_enc_goals(data, 'AwayGoals')
    yH = data[['HomeGoals']].to_numpy()
    yA = data[['AwayGoals']].to_numpy()
    y = np.concatenate((yH, yA), axis=0)
    X = np.hstack((form, elo, season, goals))
    if elo_only is True:
        return elo, y, data
    else:
        return X, y, data


def get_class_weights(y):
    y_int = [i.argmax() for i in y]
    weights = compute_class_weight('balanced', np.unique(y_int), y_int)
    weights_dict = dict(enumerate(weights))
    return weights_dict


def train_model(infile, elo_only=False):
    X, y, data = get_train_data(infile, elo_only=elo_only)
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
    print(validate[0:20])
    if elo_only is True:
        pickle.dump(model, Path('boost_classifier_elo.pickle').open(mode='wb'))
    else:
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
        form = requests.get(f'https://api.teamto.win/v1/teamForm.php?team_id={t}&fixtures=3').json()
        goals = 0
        num = 0
        max_num = 3
        for match in form[ha]:
            if match == 'form' or num >= max_num:
                continue
            else:
                goals += int(form[ha][match][prop])
                num += 1
        if max_num > num:
            avg_goals.append(goals / num)
        else:
            avg_goals.append(goals / max_num)
    np_arr = np.array([avg_goals])
    np_arr = np_arr.reshape(-1, 1)
    return np_arr


def get_x(home, away, elo_only=False):
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
    if elo_only is True:
        return elo
    else:
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
        num = 0
        max_num = 3
        for match in form[ha]['form']:
            if num >= max_num:
                continue
            val = form[ha]['form'][match]
            loc_win_avg += wld_dict[val]
            num += 1
        if max_num > num:
            loc_wins.append(loc_win_avg / num)
        else:
            loc_wins.append(loc_win_avg / max_num)
        num = 0
        wins = 0
        for match in form['all_form']:
            if num >= max_num:
                continue
            if form['all_form'][match] == 'W':
                wins += 1
            num += 1
        if max_num > num:
            form_wins.append(wins / num)
        else:
            form_wins.append(wins / max_num)
        win_ratio = (form['season_form']['home']['win_percentage'] + form['season_form']['away']['win_percentage']) / 2
        season_wins.append(win_ratio / 100)
    loc_wins = np.array(loc_wins).reshape(-1, 1)
    form_wins = np.array(form_wins).reshape(-1, 1)
    season_wins = np.array(season_wins).reshape(-1, 1)
    return loc_wins, form_wins, season_wins


def get_outcome_pred(home, away):
    outcome_prob = list()
    perms = list(product(list(range(0, len(home))), repeat=2))
    h_chance = 0
    a_chance = 0
    draw = 0
    for perm in perms:
        if perm[0] > perm[1]:
            h_chance += (home[perm[0]] * away[perm[1]])
        elif perm[0] < perm[1]:
            a_chance += (home[perm[0]] * away[perm[1]])
        elif perm[0] == perm[1]:
            draw += (home[perm[0]] * away[perm[1]])
    outcome_prob.append([h_chance, a_chance, draw])
    return h_chance, a_chance, draw


def post_data(nf, s_pred, s_prob):
    sH, sA = np.split(s_pred, 2)[0], np.split(s_pred, 2)[1]
    spH, spA = np.split(s_prob, 2)[0], np.split(s_prob, 2)[1]
    num = 0
    for fix in nf:
        home, away, draw = get_outcome_pred(spH[num], spA[num])
        nf[fix]['home_percentage'] = round(home * 100, 2)
        nf[fix]['away_percentage'] = round(away * 100, 2)
        nf[fix]['draw_percentage'] = round(draw * 100, 2)
        nf[fix]['home_goals'] = round(sH[num].item(), 2)
        nf[fix]['away_goals'] = round(sA[num].item(), 2)
        # nf[fix]['probability_home_goals'] = round(spH[num][sH[num]].item(), 2)
        # nf[fix]['probability_away_goals'] = round(spA[num][sA[num]].item(), 2)
        perc = round((spH[num][sH[num]].item() * spA[num][sA[num]].item()) * 100, 2)
        nf[fix]['percentage'] = perc
        num += 1
        print(f'{nf[fix]["home_team_name"]} {nf[fix]["home_goals"]} vs '
              f'{nf[fix]["away_goals"]} {nf[fix]["away_team_name"]} '
              f'({round(home * 100, 2)}, {round(draw * 100, 2)}, {round(away * 100, 2)}) | {perc}')
    with open('predictions.json', 'w') as f:
        json.dump(nf, f)
    # requests.post('https://api.teamto.win/v1/savePrediction.php', json.dumps(nf))


def make_prediction(gwk, elo_only=False):
    home_teams, away_teams, next_fix = pull_data(gwk)
    score_X = get_x(home_teams, away_teams, elo_only=elo_only)
    score_pred, score_prob = predict_score(score_X, elo_only=elo_only)
    post_data(next_fix, score_pred, score_prob)
    return


def predict_score(X, elo_only=False):
    if elo_only is True:
        model = pickle.load(Path('boost_classifier_elo.pickle').open(mode='rb'))
    else:
        model = pickle.load(Path('boost_classifier.pickle').open(mode='rb'))
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    return y_pred, y_prob


def score_predictor(train=False, predict=True, elo_only=False, gwk=None):
    infile = 'EPL_form_elo_API'
    if gwk is None and predict is True:
        print('No gameweek provided, exiting.')
        exit()
    if train is True:
        train_model(infile, elo_only=elo_only)
    if predict is True:
        make_prediction(gwk, elo_only=elo_only)


if __name__ == '__main__':
    score_predictor(train=False, predict=True, elo_only=False, gwk=5)
