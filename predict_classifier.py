import requests
import numpy as np
from keras.models import load_model
import json
import argparse


def calc_avg(tms, ha):
    wld_dict = {'W': 1, 'L': 0, 'D': 0}
    avg_wins = []
    for t in tms:
        form = requests.get(f'https://api.teamto.win/v1/teamForm.php?team_id={t}&fixtures=3').json()
        win_avg = 0
        for match in form[ha]['form']:
            val = form[ha]['form'][match]
            win_avg += wld_dict[val]
        avg_wins.append(win_avg / 3)
        # print(f'{t}: {avg_wins}')
    np_arr = np.array([avg_wins])
    np_arr = np_arr.reshape(-1, 1)
    return np_arr


def calc_mean(tms, ha):
    mean_wins = []
    for t in tms:
        form = requests.get(f'https://api.teamto.win/v1/teamForm.php?team_id={t}').json()
        win_ratio = form['season_form'][ha]['win_percentage']
        mean_wins.append(win_ratio/100)
    # print(f'{t}: {mean_wins}')
    np_arr = np.array([mean_wins])
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
    home_avg_win = calc_avg(home, 'home_form')
    away_avg_win = calc_avg(away, 'away_form')
    home_mean_win = calc_mean(home, 'home')
    away_mean_win = calc_mean(away, 'away')
    all = np.hstack((home_avg_win, away_avg_win, home_mean_win, away_mean_win))
    # this reflects the HPAvg, APAvg, HPMean, APMean used in building the model
    return all


def predict(X, m):
    print(X)
    model = load_model(m)
    y_pred = model.predict(X)
    return y_pred


def post_data(nf, y_pred):
    yH, yA, yD = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    num = 0
    for fix in nf:
        nf[fix]['predicted_home'] = round(yH[num].item(), 2)
        nf[fix]['predicted_away'] = round(yA[num].item(), 2)
        nf[fix]['predicted_draw'] = round(yD[num].item(), 2)
        print(nf[fix])
        num += 1
    jsonData = json.dumps(nf)
    with open('predictions.json', 'w') as f:
        json.dump(nf, f)
    print(nf)
    # requests.post('https://api.teamto.win/v1/savePrediction.php', jsonData)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', dest='week', type=int, help='Gameweek Number', required=False, default=39)
    parser.add_argument('-m', dest='model', type=str, help='Name of model to use', required=False,
                        default='cat_model.h5')
    args = parser.parse_args()
    gwk = args.week
    m = args.model

    home_teams, away_teams, next_fix = pull_data(gwk)
    X = get_x(home_teams, away_teams)
    y_pred = predict(X, m)
    post_data(next_fix, y_pred)


if __name__ == '__main__':
    main()
