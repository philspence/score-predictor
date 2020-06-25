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
