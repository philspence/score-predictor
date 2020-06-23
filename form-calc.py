import pandas as pd
import argparse


def get_mv_avg(df, key, value, win, newcol):
    moving_avg = df.groupby(key)[value].rolling(window=win).mean().shift().reset_index()
    df[newcol] = moving_avg.set_index('level_1')[value]
    return df


def calc_form(infile, w):
    data = pd.read_csv(f'{infile}.csv', header=0)
    print('Imported CSV')
    data = data[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST']]
    data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
    # data = data.loc[data.Date > '01/01/2000']
    data.sort_values(by=['Date'])
    data = data.dropna()
    data.drop_duplicates(inplace=True)
    print('Removed excess data')
    
    # HGFAvg = Home Goals For Average, S = Shots
    data = get_mv_avg(data, 'HomeTeam', 'FTHG', w, 'HGFAvg')
    data = get_mv_avg(data, 'HomeTeam', 'FTAG', w, 'HGAAvg')
    data = get_mv_avg(data, 'HomeTeam', 'HS', w, 'HSFAvg')
    data = get_mv_avg(data, 'HomeTeam', 'AS', w, 'HSAAvg')
    data = get_mv_avg(data, 'HomeTeam', 'HST', w, 'HSTFAvg')
    data = get_mv_avg(data, 'HomeTeam', 'AST', w, 'HSTAAvg')

    data = get_mv_avg(data, 'AwayTeam', 'FTAG', w, 'AGFAvg')
    data = get_mv_avg(data, 'AwayTeam', 'FTHG', w, 'AGAAvg')
    data = get_mv_avg(data, 'AwayTeam', 'AS', w, 'ASFAvg')
    data = get_mv_avg(data, 'AwayTeam', 'HS', w, 'ASAAvg')
    data = get_mv_avg(data, 'AwayTeam', 'AST', w, 'ASTFAvg')
    data = get_mv_avg(data, 'AwayTeam', 'HST', w, 'ASTAAvg')

    data['HomeStats'] = data.HGFAvg + data.AGAAvg + data.HSFAvg + data.ASAAvg + data.HSTFAvg + data.ASTAAvg
    data['AwayStats'] = data.AGFAvg + data.HGAAvg + data.ASFAvg + data.HSAAvg + data.ASTFAvg + data.HSTAAvg
    data = data.dropna(subset=['HGFAvg', 'HGAAvg', 'AGFAvg', 'AGAAvg', 'HSFAvg', 'ASAAvg', 'HSTFAvg', 'ASTAAvg',
                               'AGFAvg', 'HGAAvg', 'ASFAvg', 'HSAAvg', 'ASTFAvg', 'HSTAAvg'])
    return data
    

def main(i, o, w):
    data = calc_form(i, w)
    data.to_csv(f'{o}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='infile', type=str, help='CSV file to load', required=True)
    parser.add_argument('-w', dest='window', type=int, default=3, required=False)
    args = parser.parse_args()
    infile = args.infile
    window = args.window
    outfile = f'{infile}-avgs'
    main(infile, outfile, window)
