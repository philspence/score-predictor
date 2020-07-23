import requests
import os
import csv
import numpy as np
import pandas as pd
import pickle
import json
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
            return np.nan, np.nan, np.nan, np.nan
    except KeyError:  # if form_dict['error'] doesn't exist
        try:
            if len(form_dict[location]['fixtures']) < 3 or len(form_dict[other_loc]['fixtures']) < 3:  # if not enough history...
                print('Not enough history for fixture')
                return np.nan, np.nan, np.nan, np.nan
        except KeyError:  # if 'home' or 'away' doesn't exist, it should come to here
            print('Not enough history for fixture')
            return np.nan, np.nan, np.nan, np.nan
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
            return np.nan, np.nan, np.nan, np.nan
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
        wins_perc = 0
        goals_scored = 0
        goals_conceded = 0
        for d in form_dates:
            for fix in form_dict[location]['fixtures']:
                if form_dict[location]['fixtures'][fix]['date'] == d.strftime('%Y-%m-%d'):
                    if form_dict[location]['fixtures'][fix]['result'] == tag:
                        wins_perc += 1
                    goals_scored += int(float(form_dict[location]['fixtures'][fix][f'{location}_total_goals']))
                    goals_conceded += int(float(form_dict[location]['fixtures'][fix][f'{other_loc}_total_goals']))
            for fix in form_dict[other_loc]['fixtures']:
                if form_dict[other_loc]['fixtures'][fix]['date'] == d.strftime('%Y-%m-%d'):
                    if form_dict[other_loc]['fixtures'][fix]['result'] == other_tag:
                        wins_perc += 1
                    goals_scored += int(float(form_dict[other_loc]['fixtures'][fix][f'{other_loc}_total_goals']))
                    goals_conceded += int(float(form_dict[other_loc]['fixtures'][fix][f'{location}_total_goals']))
        wins_perc /= 5
        goals_scored /= 5
        goals_conceded /= 5
        return loc_win_perc, wins_perc, goals_scored, goals_conceded


def calc_season_win(fixtures_dict, season, home_team, away_team, fix_id):
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
    try:
        home_season_win_perc = home_season_wins / home_season_fix
        away_season_win_perc = away_season_wins / away_season_fix
    except ZeroDivisionError:
        return np.nan, np.nan
    return home_season_win_perc, away_season_win_perc


def calc_elo(home, away, outcome, elo):
    if home not in elo:
        elo[home] = 400
    if away not in elo:
        elo[away] = 400
    k = 20
    eH = 1 / (10 ** (- (elo[home] - elo[away]) / 400) + 1)
    eA = 1 - eH
    if outcome == 'H':
        elo[home] = elo[home] + (1 - eH) * k
        elo[away] = elo[away] + (0 - eA) * k
    if outcome == 'A':
        elo[home] = elo[home] + (0 - eH) * k
        elo[away] = elo[away] + (1 - eA) * k
    if outcome == 'D':
        elo[home] = elo[home] + (0.5 - eH) * k
        elo[away] = elo[away] + (0.5 - eA) * k
    return elo


def multi_process():
    return


def get_data():
    all_data = []
    num = 1
    total_fix_pages = 104
    fixtures_dict = {}
    if os.path.exists('fixtures_API_data.json'):
        print('Loading API from file')
        with open('fixtures_API_data.json', 'r') as f:
            fixtures_dict = json.load(f)
    else:
        print('Pulling API fixture data...')
        while num <= total_fix_pages:
            url = requests.get(f'https://api.teamto.win/v1/allFixtures.php?page={num}').json()
            fixtures_dict.update(url)
            num += 1
        print('Saving to file')
        with open('fixtures_API_data.json', 'w') as f:
            json.dump(fixtures_dict, f)
    print('Done')
    headers = ['HomeTeamHomeForm', 'HomeTeamForm', 'HomeTeamSeasonWinPerc',
               'AwayTeamAwayForm', 'AwayTeamForm', 'AwayTeamSeasonWinPerc',
               'HomeElo', 'AwayElo',
               'HomeScored', 'HomeConceded', 'AwayScored', 'AwayConceded',
               'HomeGoals', 'AwayGoals']
    for fix in fixtures_dict:
        fix_id = int(fixtures_dict[fix]['fixture_id'])
        print(fix_id)
        season = fixtures_dict[fix]['season_id']
        home_team = fixtures_dict[fix]['home_team_id']
        away_team = fixtures_dict[fix]['away_team_id']
        outcome = fixtures_dict[fix]['FTResult']
        home_goals = int(float(fixtures_dict[fix]['FTHomeGoals']))
        away_goals = int(float(fixtures_dict[fix]['FTAwayGoals']))
        home_elo = float(fixtures_dict[fix]['home_team_elo'])
        away_elo = float(fixtures_dict[fix]['away_team_elo'])
        # home_shots = int(fixtures_dict[fix]['HomeShots'])
        # away_shots = int(fixtures_dict[fix]['AwayShots'])
        home_loc_form, home_general_form, home_scored, home_conceded = get_form(home_team, fix_id,
                                                                                'home', season)
        away_loc_form, away_general_form, away_scored, away_conceded = get_form(away_team, fix_id,
                                                                                'away', season)
        home_season_win_perc, away_season_win_perc = calc_season_win(fixtures_dict, season, home_team, away_team,
                                                                     fix_id)
        data = [home_loc_form, home_general_form, home_season_win_perc,
                away_loc_form, away_general_form, away_season_win_perc,
                home_elo, away_elo,
                home_scored, home_conceded, away_scored, away_conceded,
                home_goals, away_goals]
        # with open('EPL_form_elo_API.csv', 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(data)
        all_data.append(data)
    print('Saving CSV')
    with open('EPL_form_elo_API.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in all_data:
            writer.writerow(row)
    print('Done.')
    return all_data


if __name__ == '__main__':
    data = get_data()
