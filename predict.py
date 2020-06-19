import requests
import numpy as np
from keras.models import load_model
import json

gwk = 39
next_fix = requests.get(f'https://api.teamto.win/v1/gameweekFixtures.php?gameweek={gwk}').json()
# teams = requests.get('https://api.teamto.win/v1/listOfTeams.php?').json()

home_teams = []
away_teams = []
for fix in next_fix:
    home_teams.append(next_fix[fix]['home_team_id'])
    away_teams.append(next_fix[fix]['away_team_id'])


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


home_avg_goals = calc_goals(home_teams, 'home_form', 'goals_scored')
away_avg_conc = calc_goals(away_teams, 'away_form', 'goals_conceded')
away_avg_goals = calc_goals(home_teams, 'away_form', 'goals_scored')
home_avg_conc = calc_goals(away_teams, 'home_form', 'goals_conceded')

home_avgs = np.append(home_avg_goals, away_avg_conc, axis=1)
away_avgs = np.append(away_avg_goals, home_avg_conc, axis=1)
X = np.concatenate((home_avgs, away_avgs))
# print(X)
model = load_model('avgG-model-06')
y_pred = model.predict(X)
yH, ya = np.split(y_pred, 2)[0], np.split(y_pred, 2)[1]

num = 0
for fix in next_fix:
    next_fix[fix]['predicted_home_goals'] = round(yH[num].item(), 1)
    next_fix[fix]['predicted_away_goals'] = round(ya[num].item(), 1)
    print(next_fix[fix])
    num += 1

jsonData = json.dumps(next_fix)
# requests.post(, jsonData)
