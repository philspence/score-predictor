import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import requests
import numpy as np
from keras.models import load_model
import json
import pickle
import argparse
from keras.utils import to_categorical
from itertools import permutations, product
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd


def pull_data(gwk):
    next_fix = requests.get(f'https://api.teamto.win/v1/gameweekFixtures.php?gameweek={gwk}').json()
    # teams = requests.get('https://api.teamto.win/v1/listOfTeams.php?').json()
    home_teams = []
    away_teams = []
    for fix in next_fix:
        home_teams.append(next_fix[fix]['home_team_id'])
        away_teams.append(next_fix[fix]['away_team_id'])
    return home_teams, away_teams, next_fix


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


def get_x(home, away):
    home_elo = get_elo(home)
    away_elo = get_elo(away)
    elo = stack_concat(home_elo, away_elo)
    scaler = preprocessing.MinMaxScaler()
    elo_norm = scaler.fit_transform(elo)
    return elo


def onehot_enc_goals(df, goals):
    return to_categorical(df[goals])


def plot_goal_probs(h, a):
    h_probs = [v for k, v in h.items()]
    a_probs = [v for k, v in a.items()]
    x = np.arange(len(h_probs))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, h_probs, width, label='Home')
    ax.bar(x + width/2, a_probs, width, label='Away')
    ax.set_ylabel('Probability')
    ax.set_title('Predicted Goals')
    ax.legend()
    fig.tight_layout()
    plt.savefig('Liverpool-Palace.png', dpi=300)
    plt.show()


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
    perms = list(product(list(range(0, 5)), repeat=2))
    for team in range(0, int(len(y) / 2)):
        home = goal_probs[team]
        opposite_team = team + int(len(y) / 2)
        away = goal_probs[opposite_team]
        # plot_goal_probs(home, away)
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


def predict_score(X, m):
    model = load_model(f'best_{m}.h5')
    model.load_weights(f'{m}_weights.h5')
    y_pred = model.predict(X)
    goals, probs, outcome = ohe_to_goals(y_pred)
    probs_arr = np.round_(np.multiply(y_pred, 100), decimals=2)
    return goals, probs, outcome, probs_arr


def post_data(nf, o_pred, s_pred, s_prob, p_arr):
    oH, oA, oD = o_pred[:, 0], o_pred[:, 1], o_pred[:, 2]
    sH, sA = np.split(s_pred, 2)[0], np.split(s_pred, 2)[1]
    spH, spA = np.split(s_prob, 2)[0], np.split(s_prob, 2)[1]
    p_arrH, p_arrA = np.split(p_arr, 2)[0], np.split(p_arr, 2)[1]
    num = 0
    for fix in nf:
        nf[fix]['home_percentage'] = round(oH[num].item() * 100, 2)
        nf[fix]['away_percentage'] = round(oA[num].item() * 100, 2)
        nf[fix]['draw_percentage'] = round(oD[num].item() * 100, 2)
        # nf[fix]['predicted_home_goals'] = int(sH[num].item())
        # nf[fix]['predicted_away_goals'] = int(sA[num].item())
        nf[fix]['predicted_home_goals'] = p_arrH[num].tolist()
        nf[fix]['predicted_away_goals'] = p_arrA[num].tolist()
        nf[fix]['percentage'] = round((spH[num].item() * spA[num].item()) * 100, 2)
        print(int(sH[num].item()), int(sA[num].item()))
        num += 1
        print(nf[fix])

    with open('predictions.json', 'w') as f:
        json.dump(nf, f)
    # print(nf)
    requests.post('https://api.teamto.win/v1/savePrediction.php', json.dumps(nf))
    with open('predictions.json', 'w') as f:
        json.dump(nf, f)


def main(gwk, m):
    # teams = get_3D_XY(pd.read_csv('top_leagues/EPL_merged_avgs.csv', header=0, index_col=0, infer_datetime_format=True))
    home_teams, away_teams, next_fix = pull_data(gwk)
    score_X = get_x(home_teams, away_teams)
    score_pred, score_prob, outcome_pred, probs_arr = predict_score(score_X, m)
    post_data(next_fix, outcome_pred, score_pred, score_prob, probs_arr)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-w', dest='week', type=int, help='Gameweek Number', required=True)
        parser.add_argument('-m', dest='model_score', type=str, help='Name of score model, no extension',
                            required=True)
        args = parser.parse_args()
        gwk = args.week
        m_score = args.model_score
    except:
        gwk = 4
        m_score = 'score_cat_model_elo'
    finally:
        main(gwk, m_score)
