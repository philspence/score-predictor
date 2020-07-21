import requests
import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime as dt


def get_form(team_id, fix_id, location, season):
    if location == 'home':
        tag = 'H'
        other_loc = 'away'
        other_tag = 'A'
    else:
        tag = 'A'
        other_loc = 'home'
        other_tag = 'H'
    form_dict = requests.get(f'https://api.teamto.win/v1/teamHistory.php?team_id={team_id}&fixture_id={fix_id}').json()
    try:
        if form_dict['error']:  # if no history...
            print('No history for fixture')
            return False, False
    except KeyError:  # if form_dict['error'] doesn't exist
        try:
            if len(form_dict[location]['fixtures']) < 3 or len(form_dict[other_loc]['fixtures']) < 3:  # if not enough history...
                print('Not enough history for fixture')
                return False, False
        except KeyError:  # if 'home' or 'away' doesn't exist, it should come to here
            print('Not enough history for fixture')
            return False, False
        home_season_count = 0
        away_season_count = 0
        for fix in form_dict[location]['fixtures']:
            if form_dict[location]['fixtures'][fix]['season_id'] == season:
                home_season_count += 1
        for fix in form_dict[other_loc]['fixtures']:
            if form_dict[other_loc]['fixtures'][fix]['season_id'] == season:
                away_season_count += 1
        if home_season_count < 3 or away_season_count < 3:
            print('Not enough history for this season')
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
                    if form_dict[other_loc]['fixtures'][fix]['result'] == other_tag:
                        wins += 1
        win_perc = wins / 5
        return loc_win_perc, win_perc


def get_data():
    num = 1
    total_fix_pages = 52
    fixtures_dict = {}
    print('Pulling API fixture data...')
    while num <= total_fix_pages:
        url = requests.get(f'https://api.teamto.win/v1/allFixtures.php?page={num}').json()
        fixtures_dict.update(url)
        num += 1
    print('Done')
    headers = ['HomeTeamHomeForm', 'HomeTeamForm', 'HomeTeamSeasonWinPerc',
               'AwayTeamAwayForm', 'AwayTeamForm', 'AwayTeamSeasonWinPerc',
               'HomeGoals', 'AwayGoals']
    with open('EPL_form_API.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    for fix in fixtures_dict:
        fix_id = int(fixtures_dict[fix]['fixture_id'])
        print(fix_id)
        season = fixtures_dict[fix]['season_id']
        home_team = fixtures_dict[fix]['home_team_id']
        away_team = fixtures_dict[fix]['away_team_id']
        home_goals = int(float(fixtures_dict[fix]['FTHomeGoals']))
        away_goals = int(float(fixtures_dict[fix]['FTAwayGoals']))
        # home_shots = int(fixtures_dict[fix]['HomeShots'])
        # away_shots = int(fixtures_dict[fix]['AwayShots'])
        home_loc_form, home_general_form = get_form(home_team, fix_id, 'home', season)
        away_loc_form, away_general_form = get_form(away_team, fix_id, 'away', season)
        if home_loc_form is False or away_loc_form is False:
            continue

        home_season_wins = 0
        home_season_fix = 0
        away_season_wins = 0
        away_season_fix = 0
        for i in fixtures_dict:
            if fixtures_dict[i]['season_id'] == season and int(fixtures_dict[i]['fixture_id']) < fix_id and \
                    fixtures_dict[i]['home_team_id'] == home_team:
                home_season_fix += 1
                if fixtures_dict[i]['FTResult'] == 'H':
                    home_season_wins += 1
            if fixtures_dict[i]['season_id'] == season and int(fixtures_dict[i]['fixture_id']) < fix_id and \
                    fixtures_dict[i]['away_team_id'] == away_team:
                away_season_fix += 1
                if fixtures_dict[i]['FTResult'] == 'A':
                    away_season_wins += 1
        home_season_win_perc = home_season_wins / home_season_fix
        away_season_win_perc = away_season_wins / away_season_fix
        data = [home_loc_form, home_general_form, home_season_win_perc,
                away_loc_form, away_general_form, away_season_win_perc,
                home_goals, away_goals]
        with open('EPL_form_API.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)
    return


if __name__ == '__main__':
    get_data()
