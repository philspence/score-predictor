import requests
import numpy as np
from keras.models import load_model
import json
import argparse
import pandas as pd
from itertools import product
from scipy.stats import ttest_ind, ttest_rel, stats


def calc_goals(tms, ha, prop):
    avg_goals = []
    for t in tms:
        form = requests.get(f'https://api.teamto.win/v1/teamForm.php?team_id={t}&fixtures=3').json()
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


def pull_data(gwk):
    next_fix = requests.get(f'https://api.teamto.win/v1/gameweekFixtures.php?gameweek={gwk}').json()
    # teams = requests.get('https://api.teamto.win/v1/listOfTeams.php?').json()
    home_teams = []
    away_teams = []
    for fix in next_fix:
        home_teams.append(next_fix[fix]['home_team_id'])
        away_teams.append(next_fix[fix]['away_team_id'])
    return home_teams, away_teams, next_fix


def get_x(home, away):
    home_avg_goals = calc_goals(home, 'home_form', 'goals_scored')
    away_avg_conc = calc_goals(away, 'away_form', 'goals_conceded')
    away_avg_goals = calc_goals(home, 'away_form', 'goals_scored')
    home_avg_conc = calc_goals(away, 'home_form', 'goals_conceded')

    home_avgs = np.append(home_avg_goals, away_avg_conc, axis=1)
    away_avgs = np.append(away_avg_goals, home_avg_conc, axis=1)
    return np.concatenate((home_avgs, away_avgs))


def predict(X, m):
    model = load_model(m)
    y_pred = model.predict(X)
    yH, ya = np.split(y_pred, 2)[0], np.split(y_pred, 2)[1]
    return yH, ya


def percentages(yH, yA):
    h_range = np.linspace(yH-0.6, yH+0.6, num=12).round(0)
    a_range = np.linspace(yA+0.6, yA-0.6, num=12).round(0)
    home = 0
    away = 0
    draw = 0
    num = 0
    while num < 12:
        if h_range[num] > a_range[num]:
            home += 1
        elif h_range[num] < a_range[num]:
            away += 1
        else:
            draw += 1
        num += 1
    return round((home/12)*100), round((away/12)*100), round((draw/12)*100)


def post_data(nf, yH, yA):
    num = 0
    for fix in nf:
        nf[fix]['predicted_home_goals'] = round(yH[num].item(), 1)
        nf[fix]['predicted_away_goals'] = round(yA[num].item(), 1)
        home, away, draw = percentages(yH[num], yA[num])
        # nf[fix]['home_win_percentage'] = home
        # nf[fix]['away_win_percentage'] = away
        # nf[fix]['draw_percentage'] = draw
        print(nf[fix])
        num += 1
    jsonData = json.dumps(nf)
    print(nf)
    # requests.post(, jsonData)


def load_data_csv(infile):
    data = pd.read_csv(infile, header=0)
    Xh = data[['HGFAvg', 'AGAAvg']].to_numpy()
    Xa = data[['AGFAvg', 'HGAAvg']].to_numpy()
    X = np.concatenate((Xh, Xa))
    print(X)
    return X, data


def add_pred_to_csv(data, yH, yA, o):
    data['PredHG'],  data['PredAG'] = [np.round(yH, 1), np.round(yA, 1)]
    data.to_csv(o)
    # y_test_np = np.array([y_test])
    # y_test_np = np.reshape(y_test_np, (-1, 1))
    # pred_real = np.append(y_pred, y_test_np, axis=1)
    # print(np.round(pred_real, 1)[0:20])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='target', type=str, help='CSV or JSON as source', required=False, default='JSON')
    parser.add_argument('-w', dest='week', type=int, help='Gameweek Number', required=False, default=39)
    parser.add_argument('-m', dest='model', type=str, help='Name of model to use', required=False,
                        default='avgG-model-06')
    args = parser.parse_args()
    target = args.target
    gwk = args.week
    m = args.model

    if target == 'JSON':
        home_teams, away_teams, next_fix = pull_data(gwk)
        X = get_x(home_teams, away_teams)
        yH, yA = predict(X, m)
        post_data(next_fix, yH, yA)
    else:
        infile = f'{target}.csv'
        X, data = load_data_csv(infile)
        yH, yA = predict(X, m)
        add_pred_to_csv(data, yH, yA, f'{target}-predictions.csv')


if __name__ == '__main__':
    main()
