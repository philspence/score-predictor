import requests
import numpy as np
from keras.models import load_model
import json
import pickle
import argparse
from keras.utils import to_categorical
from itertools import permutations, product


def calc_avg(tms, ha):
    wld_dict = {'W': 1, 'L': 0, 'D': 0}
    avg_wins = []
    for t in tms:
        form = requests.get(f'https://api.teamto.win/v1/teamForm.php?team_id={t}&fixtures=3').json()
        win_avg = 0
        num = 0
        while num < 3:
            for match in form[ha]['form']:
                val = form[ha]['form'][match]
                win_avg += wld_dict[val]
                num += 1
        avg_wins.append(win_avg / num)
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


def get_score_x(home, away):
    home_avg_goals = calc_goals(home, 'home_form', 'goals_scored')
    away_avg_conc = calc_goals(away, 'away_form', 'goals_conceded')
    away_avg_goals = calc_goals(home, 'away_form', 'goals_scored')
    home_avg_conc = calc_goals(away, 'home_form', 'goals_conceded')
    all = np.hstack((home_avg_goals, away_avg_conc, away_avg_goals, home_avg_conc))
    return all


def get_x(home, away):
    home_avg_win = calc_avg(home, 'home_form')
    away_avg_win = calc_avg(away, 'away_form')
    home_mean_win = calc_mean(home, 'home')
    away_mean_win = calc_mean(away, 'away')
    home_X = np.hstack((home_mean_win.reshape(-1, 1), away_mean_win.reshape(-1, 1)))
                        # home_avg_win.reshape(-1, 1), away_avg_win.reshape(-1, 1)))
    away_X = np.hstack((away_mean_win.reshape(-1, 1), home_mean_win.reshape(-1, 1)))
    all_X = np.concatenate((home_X, away_X), axis=0)
    # this reflects the HPAvg, APAvg, HPMean, APMean used in building the model
    # outcome_X = np.hstack((home_mean_win.reshape(-1, 1), away_mean_win.reshape(-1, 1), home_avg_win.reshape(-1, 1),
    #                        away_avg_win.reshape(-1, 1)))
    return all_X


def predict(X, m):
    # with open('random_forest_cat.pickle', 'rb') as p:
    #     model = pickle.load(p)
    model = load_model('cat_best_model.h5')
    y_pred = model.predict_proba(X)
    return y_pred


def onehot_enc_goals(df, goals):
    return to_categorical(df[goals])


def ohe_to_goals(y):
    goals = []
    probs = []
    goal_probs = []
    gp_dict = {}
    for i in y:
        gp_dict = {}
        match = np.where(i == np.amax(i))[0]
        goals.append(match)
        probs.append(np.amax(i))
        num = 0
        for goal in i:
            gp_dict[num] = goal
            num += 1
        goal_probs.append(gp_dict)
    outcome_prob = []
    perms = list(product(list(range(0, 10)), repeat=2))
    for team in range(0, 10):
        home = goal_probs[team]
        away = goal_probs[team+10]
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
    return np.array(goals), np.array(probs), np.array(outcome_prob)


def predict_score(X):
    model = load_model('best_score_cat_model.h5')
    y_pred = model.predict(X)
    goals, probs, outcome = ohe_to_goals(y_pred)
    return goals, probs, outcome


def post_data(nf, o_pred, s_pred, s_prob):
    oH, oA, oD = o_pred[:, 0], o_pred[:, 1], o_pred[:, 2]
    sH, sA = np.split(s_pred, 2)[0], np.split(s_pred, 2)[1]
    spH, spA = np.split(s_prob, 2)[0], np.split(s_prob, 2)[1]
    num = 0
    for fix in nf:
        nf[fix]['home_percentage'] = round(oH[num].item() * 100, 2)
        nf[fix]['away_percentage'] = round(oA[num].item() * 100, 2)
        nf[fix]['draw_percentage'] = round(oD[num].item() * 100, 2)
        nf[fix]['predicted_home_goals'] = int(sH[num].item())
        nf[fix]['predicted_away_goals'] = int(sA[num].item())
        nf[fix]['percentage'] = round((spH[num].item() * spA[num].item()) * 100, 2)
        num += 1
    with open('predictions.json', 'w') as f:
        json.dump(nf, f)
    print(nf)
    requests.post('https://api.teamto.win/v1/savePrediction.php', json.dumps(nf))


def main(gwk, mo):
    home_teams, away_teams, next_fix = pull_data(gwk)
    score_X = get_x(home_teams, away_teams)
    # outcome_pred = predict(outcome_X, mo)
    # score_x = get_score_x(home_teams, away_teams)
    # X_outcome = np.hstack((score_x, outcome_pred[:, 0].reshape(-1, 1), outcome_pred[:, 1].reshape(-1, 1),
    #                        outcome_pred[:, 2].reshape(-1, 1)))
    # X_outcome = np.hstack((X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1), outcome_pred[0][:, 1].reshape(-1, 1),
    #                        outcome_pred[1][:, 1].reshape(-1, 1)))
    score_pred, score_prob, outcome_pred = predict_score(score_X)
    post_data(next_fix, outcome_pred, score_pred, score_prob)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-w', dest='week', type=int, help='Gameweek Number', required=True)
        parser.add_argument('-mo', dest='model_outcome', type=str, help='Name of outcome model to use, no extension',
                            required=True)
        parser.add_argument('-ms', dest='model_score', type=str, help='Name of score model, no extension',
                            required=True)
        args = parser.parse_args()
        gwk = args.week
        m_outcome = args.model_outcome
    except:
        gwk = 40
        m_outcome = 'cat_best_model'
        m_score = 'best_score_cat_model'
    finally:
        main(gwk, m_outcome)
