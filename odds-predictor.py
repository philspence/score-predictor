import pandas as pd
import numpy as np
import os
import requests
import operator
import csv
from bs4 import BeautifulSoup
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils import to_categorical
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score

def loadCSV(path):
    df = pd.read_csv(path, header=0)
    df = df.loc[:, ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', bet['home'], bet['draw'], bet['away']]]
    return df

def getDFfromURL(url, h):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')
    tb = soup.find_all('table')
    df = pd.read_html(str(tb), header=h)[0]
    return df

def chng_to_dec(df, col):
    num = 0
    y = []
    while num < len(df.index):
        str = df.iloc[num][col]
        try:
            n, d = map(float, str.split('/'))
            i = n / d
            i += 1
        except:
            i = float(str)
            i += 1
        y.append(i)
        num += 1
    df[col] = y

def getNewOdds(dict):
    df = getDFfromURL('http://odds.bestbetting.com/football/england/premier-league/', 0)
    df = df[df.Home.notnull()]
    df = df[df.Home != 'Home']
    del df['Unnamed: 0'], df['BPP']
    df.reset_index(inplace=True, drop=True)
    df.columns = ['Teams', dict['home'], dict['draw'], dict['away']]
    df[['HomeTeam', 'AwayTeam']] = df.Teams.str.split(' v ', expand=True)
    del df['Teams']
    df[['AwayTeam', 'Delete']] = df.AwayTeam.str.split('(', expand=True)
    df['AwayTeam'] = df.AwayTeam.str.strip()
    del df['Delete']
    chng_to_dec(df, dict['home'])
    chng_to_dec(df, dict['draw'])
    chng_to_dec(df, dict['away'])
    return df

# def encY(y, n):
#     tran_y = le.transform(y)
#     enc_y = to_categorical(tran_y, num_classes=n)
#     return enc_y
#
# def deencY(y):
#     inv_y = []
#     for i in y:
#         deenc = np.argmax(i)
#         inv_y.append(deenc)
#     newy = le.inverse_transform(inv_y)
#     return newy

def isCorrect(y, ny):
    num = 0
    cor = 0
    incor = 0
    while num < len(y):
        if y[num] == ny[num]:
            cor += 1
        else:
            incor += 1
        num += 1
    print(str(cor) + ' out of ' + str(cor + incor))

def isHomeWin(df, n, hg, ag, r):
    if df.iloc[n][hg] > df.iloc[n][ag]:
        df.at[n, r] = 'H'
    elif df.iloc[n][hg] < df.iloc[n][ag]:
        df.at[n, r] = 'A'
    elif df.iloc[n][hg] == df.iloc[n][ag]:
        df.at[n, r] = 'D'

# def define_model():
#     model = Sequential()
#     model.add(Dense(15, input_dim=1, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(8, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal'))
#     model.compile(loss='mse', optimizer='nadam', metrics=['accuracy'])
#     return model

def modelPred(X, model):
    newy_all = []
    num = 0
    while num < 5:
        tempy = model.predict(X)
        newy_all.append(tempy)
        num += 1
    newy = (np.array(newy_all[0]) + np.array(newy_all[1]) + np.array(newy_all[2]) + np.array(newy_all[3])
            + np.array(newy_all[4])) / 5
    return newy

def getHistoricalData():
    all_dfs = {}
    filenames = []
    for root, dirs, files in os.walk('odds'):
        for file in files:
            if file.endswith('.csv'):
                try:
                    path = os.path.join(root, file)
                    key = str(path)
                    all_dfs[key] = loadCSV(path)
                    filenames.append(path)
                except:
                    continue
    all_res = pd.concat(list(all_dfs.values()), keys=list(all_dfs.keys()))
    all_res = all_res[all_res.FTR.notnull()]
    all_res = all_res[all_res.WHH.notnull()]
    return all_res

def preprocessData(df):
    h = np.array(df[bet['home']])
    a = np.array(df[bet['away']])
    X = np.append(h, a)
    h = np.array(df['FTHG'])
    a = np.array(df['FTAG'])
    y = np.append(h, a)
    # #TESTING
    # X = df[[bet['home'], bet['away'], bet['draw']]]
    # y = df[['FTHG', 'FTAG']]
    return X, y

def predictData(df, model):
    # home
    newXH = np.array(df.loc[:, bet['home']])
    newXH = newXH.reshape(-1, 1)
    df['PredHG'] = modelPred(newXH, model)
    df['PredHG'] = df.PredHG.round(decimals=2)
    # away
    newXA = np.array(df.loc[:, bet['away']])
    newXA = newXA.reshape(-1, 1)
    df['PredAG'] = modelPred(newXA, model)
    df['PredAG'] = df.PredAG.round(decimals=2)
    # #TESTING
    # testX = np.array(df.loc[:, [bet['home'], bet['away'], bet['draw']]])
    # df[['PredHG', 'PredAG']] = modelPred(testX, model)

def validData(df, newy, col):
    isCorrect(np.array(df[col]), newy)
    print(mean_squared_error(np.array(df[col]), newy))
    print(r2_score(np.array(df[col]), newy))

def concatArray(y1, y2):
    y1 = np.array(y1)
    y2 = np.array(y2)
    newy = np.append(y1, y2)
    return newy

def sortTeam(df, name, rep_name, col):
    ind = df[df[col] == name].index[0]
    df.at[ind, col] = rep_name

def dictToCSV(dict, pos, name):
    writer = csv.writer(open(name, 'a'))
    for key, val in dict.items():
        writer.writerow([pos, key, val])

bet = {
    'home': 'WHH',
    'away': 'WHA',
    'draw': 'WHD'
}

print('Loading data...')
all_data = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/').json()
teams_df = getDFfromURL('https://fbref.com/en/comps/9/stats/Premier-League-Stats', [0, 1])
players = all_data['elements']
teams = all_data['teams']
all_res = getHistoricalData()

print('Scraping new odds...')
# odds_df = getNewOdds(bet)
# odds_df = odds_df.assign(PredHG=np.nan, PredAG=np.nan, PredR='')
#TESTING
odds_df = pd.read_csv('19-20-odds.csv')
odds_df = odds_df[['HomeTeam', 'AwayTeam', 'WHH', 'WHA', 'WHD', 'FTHG', 'FTAG', 'FTR']]

print('Preprocessing data...')
X, y = preprocessData(all_res)

print('Compiling model...')
# model = define_model()
model = linear_model.SGDRegressor(loss="squared_loss")

print('Fitting model...')
# model.fit(X, y, batch_size=32, epochs=10, verbose=0)
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
model.fit(X, y)

print('Processing data...')
predictData(odds_df, model)
odds_df.PredHG = odds_df.PredHG.round()
odds_df.PredAG = odds_df.PredAG.round()


# #TESTING
odds_df.to_csv('testing-19-20.csv')
correct = odds_df['PredHG'].eq(odds_df['FTHG']) & odds_df['PredAG'].eq(odds_df['FTAG'])
print(correct.eq(True).sum())
print(correct.count())
exit()

teams_arr = concatArray(odds_df['HomeTeam'], odds_df['AwayTeam'])
goals_arr = concatArray(odds_df['PredHG'], odds_df['PredAG'])
cons_arr = concatArray(odds_df['PredAG'], odds_df['PredHG'])

data = {'Teams': teams_arr, 'xG': goals_arr, 'xGC': cons_arr}
pred_res = pd.DataFrame(data)

sortTeam(pred_res, 'Sheffield United', 'Sheffield Utd', 'Teams')
sortTeam(pred_res, 'Brighton & Hove Albion', 'Brighton', 'Teams')
sortTeam(pred_res, 'Leicester City', 'Leicester', 'Teams')
sortTeam(pred_res, 'Manchester United', 'Man Utd', 'Teams')
sortTeam(pred_res, 'Norwich City', 'Norwich', 'Teams')
sortTeam(pred_res, 'Wolverhampton Wanderers', 'Wolves', 'Teams')
sortTeam(pred_res, 'Tottenham Hotspur', 'Spurs', 'Teams')
sortTeam(pred_res, 'AFC Bournemouth', 'Bournemouth', 'Teams')
sortTeam(pred_res, 'Manchester City', 'Man City', 'Teams')
sortTeam(pred_res, 'Newcastle United', 'Newcastle', 'Teams')

print(pred_res.to_string())

print('Saving data...')

pred_res.to_csv('predictions.csv')

# print('Validating data...')
# validData(odds_df, actual_newyH, 'PredHGint')
# validData(odds_df, actual_newyA, 'PredAGint')
# isCorrect(df.loc[:, 'PredR'], df.loc[:, 'FTR'])

#PLAYERS

teams_df.at[8, ('Unnamed: 0_level_0', 'Squad')] = 'Leicester'
teams_df.at[10, ('Unnamed: 0_level_0', 'Squad')] = 'Man City'
teams_df.at[11, ('Unnamed: 0_level_0', 'Squad')] = 'Man Utd'
teams_df.at[12, ('Unnamed: 0_level_0', 'Squad')] = 'Newcastle'
teams_df.at[13, ('Unnamed: 0_level_0', 'Squad')] = 'Norwich'
teams_df.at[16, ('Unnamed: 0_level_0', 'Squad')] = 'Spurs'

for team in teams:
    team['goals'] = pd.to_numeric(
        teams_df[teams_df[('Unnamed: 0_level_0', 'Squad')] == team['name']][('Performance', 'Gls')]).item()
    team['assists'] = pd.to_numeric(
        teams_df[teams_df[('Unnamed: 0_level_0', 'Squad')] == team['name']][('Performance', 'Ast')]).item()
    team['mins'] = pd.to_numeric(
        teams_df[teams_df[('Unnamed: 0_level_0', 'Squad')] == team['name']][('Playing Time', 'MP')]).item()
    team['xG'] = pd.to_numeric(pred_res[pred_res['Teams'] == team['name']]['xG']).item()
    team['xA'] = pd.to_numeric(pred_res[pred_res['Teams'] == team['name']]['xG']).item()
    team['xGC'] = pd.to_numeric(pred_res[pred_res['Teams'] == team['name']]['xGC']).item()

gks = {}
defs = {}
mids = {}
fwds = {}
for player in players:
    if player['chance_of_playing_next_round'] is not None and player['chance_of_playing_next_round'] < 74:
        continue
    for team in teams:
        if player['team_code'] == team['code']:
            player['team'] = team['name']
            player['xGcoef'] = (player['goals_scored'] / team['goals'])
            player['xAcoef'] = (player['assists'] / team['assists'])
            player['xG'] = (team['xG'] * player['xGcoef'])
            player['xA'] = (team['xA'] * player['xAcoef'])
            if float(player['form']) > 1:
                player['xGC'] = team['xGC']
                player['xP'] = 2
            else:
                player['xGC'] = 10
                player['xP'] = 0
    player['xP'] = player['xP'] + (player['xA'] * 3)
    if player['element_type'] == 1:
        player['xP'] = player['xP'] + (player['xG'] * 6)
        if player['xGC'] < 1:
            player['xCS'] = 1 - player['xGC']
            player['xP'] = player['xP'] + (player['xCS'] * 4)
        gks[player['first_name'] + ' ' + player['second_name']] = round(player['xP'], 2)
    elif player['element_type'] == 2:
        player['xP'] = player['xP'] + (player['xG'] * 6)
        if player['xGC'] < 1:
            player['xCS'] = 1 - player['xGC']
            player['xP'] = player['xP'] + (player['xCS'] * 4)
        defs[player['first_name'] + ' ' + player['second_name']] = round(player['xP'], 2)
    elif player['element_type'] == 3:
        player['xP'] = player['xP'] + (player['xG'] * 5)
        if player['xGC'] < 1:
            player['xCS'] = 1 - player['xGC']
            player['xP'] = player['xP'] + (player['xCS'] * 1)
        mids[player['first_name'] + ' ' + player['second_name']] = round(player['xP'], 2)
    elif player['element_type'] == 4:
        player['xP'] = player['xP'] + (player['xG'] * 4)
        fwds[player['first_name'] + ' ' + player['second_name']] = round(player['xP'], 2)

if os.path.isfile('all_players.csv'):
    os.remove('all_players.csv')
elif os.path.isfile(('top_players.csv')):
    os.remove('top_players.csv')

top_gks = dict(sorted(gks.items(), key=operator.itemgetter(1), reverse=True)[:9])
dictToCSV(gks, 'GK', 'all_players.csv')
dictToCSV(top_gks, 'GK', 'top_players.csv')

top_defs = dict(sorted(defs.items(), key=operator.itemgetter(1), reverse=True)[:24])
dictToCSV(defs, 'DEF', 'all_players.csv')
dictToCSV(top_defs, 'DEF', 'top_players.csv')

top_mids = dict(sorted(mids.items(), key=operator.itemgetter(1), reverse=True)[:24])
dictToCSV(mids, 'MID', 'all_players.csv')
dictToCSV(top_mids, 'MID', 'top_players.csv')

top_fwds = dict(sorted(fwds.items(), key=operator.itemgetter(1), reverse=True)[:14])
dictToCSV(fwds, 'FWD', 'all_players.csv')
dictToCSV(top_fwds, 'FWD', 'top_players.csv')
