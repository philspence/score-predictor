import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime as dt


def get_form(team_id, fix_id, location):
    if location == 'home':
        tag = 'H'
        other_loc = 'away'
    else:
        tag = 'A'
        other_loc = 'home'
    form_dict = requests.get(f'https://api.teamto.win/v1/teamHistory.php?team_id={team_id}&fixture_id={fix_id}').json()
    try:
        if form_dict['error']:
            print('No history for fixture')
            return False, False
    except KeyError:
        try:
            if len(form_dict[location]['fixtures']) < 3 or len(form_dict[other_loc]['fixtures']) < 3:
                print('Not enough history for fixture')
                return False, False
        except KeyError:
            print('Not enough history for fixture')
            return False, False
        nums = ['1', '2', '3']
        wins = 0
        dates = []
        for num in nums:
            if form_dict[location]['fixtures'][num]['result'] == tag:
                wins += 1
            dates.append(dt.strptime(form_dict[location]['fixtures'][num]['date'], '%Y-%m-%d'))
            dates.append(dt.strptime(form_dict[other_loc]['fixtures'][num]['date'], '%Y-%m-%d'))
        loc_win_perc = wins / 3
        form_dates = sorted(dates, reverse=True)[0:5]
        wins = 0
        for d in form_dates:
            for fix in form_dict[location]['fixtures']:
                if form_dict[location]['fixtures'][fix]['date'] == d.strftime('%Y-%m-%d'):
                    if form_dict[location]['fixtures'][fix]['result'] == tag:
                        wins += 1
            for fix in form_dict[other_loc]['fixtures']:
                if form_dict[other_loc]['fixtures'][fix]['date'] == d.strftime('%Y-%m-%d'):
                    if form_dict[other_loc]['fixtures'][fix]['result'] == tag:
                        wins += 1
        win_perc = wins / 5
        return loc_win_perc, win_perc


def get_data():
    num = 1
    total_fix_pages = 52
    fixtures_dict = {}
    while num <= total_fix_pages:
        url = requests.get(f'https://api.teamto.win/v1/allFixtures.php?page={num}').json()
        fixtures_dict.update(url)
        num += 1

    data = np.empty([0, 6])
    for fix in fixtures_dict:
        fix_id = fixtures_dict[fix]['fixture_id']
        print(fix_id)
        home_team = fixtures_dict[fix]['home_team_id']
        away_team = fixtures_dict[fix]['away_team_id']
        home_goals = int(float(fixtures_dict[fix]['FTHomeGoals']))
        away_goals = int(float(fixtures_dict[fix]['FTAwayGoals']))
        # home_shots = int(fixtures_dict[fix]['HomeShots'])
        # away_shots = int(fixtures_dict[fix]['AwayShots'])
        home_loc_form, home_general_form = get_form(home_team, fix_id, 'home')
        away_loc_form, away_general_form = get_form(away_team, fix_id, 'away')
        if home_loc_form is False or away_loc_form is False:
            continue
        data = np.append(data, [home_loc_form, home_general_form, away_loc_form, away_general_form, home_goals, away_goals])
    data = data.reshape(-1, 6)
    return data


data = get_data()
cols = ['HomeTeamHomeForm', 'HomeTeamForm', 'AwayTeamAwayForm', 'AwayTeam', 'HomeGoals', 'AwayGoals']
df = pd.DataFrame(data, columns=cols)
df.to_csv('EPL_API_form.csv')
print(len(data))
