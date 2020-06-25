import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder


def get_mv_avg(df, key, value, win, newcol):
    print(f'Calculating moving average for {newcol}...')
    mv = df.groupby([key, 'Season'])[value].rolling(win, min_periods=3).mean().shift().reset_index()
    df[newcol] = mv.set_index('level_2')[value]
    return df


def get_avg(df, key, value, win, newcol):
    print(f'Calculating average for {newcol}...')
    avg = df.groupby([key, 'Season'])[value].expanding(3).mean().reset_index()
    df[newcol] = avg.set_index('level_2')[value]
    return df


def calc_form(infile, w):
    data = pd.read_csv(f'{infile}.csv', header=0, index_col=0, infer_datetime_format=True)
    print('Imported CSV')
    columns = ['Season', 'Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    data = data[columns]
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.dropna()
    data.sort_values(by=['Date'], inplace=True)
    print('Removed excess data')
    data['HW'] = [1 if ele == 'H' else 0 for ele in data['FTR']]
    data['AW'] = [1 if ele == 'A' else 0 for ele in data['FTR']]
    data['D'] = [1 if ele == 'D' else 0 for ele in data['FTR']]
    # HGFAvg = Home Goals For Average, S = Shots
    data['HP'] = [1 if ele == 'H' else 0 for ele in data['FTR']]
    data['AP'] = [1 if ele == 'A' else 0 for ele in data['FTR']]

    data = get_avg(data, 'HomeTeam', 'HP', w, 'HPMean')
    data = get_avg(data, 'AwayTeam', 'AP', w, 'APMean')

    data = get_mv_avg(data, 'HomeTeam', 'HP', w, 'HPAvg')
    data = get_mv_avg(data, 'AwayTeam', 'AP', w, 'APAvg')

    data = get_mv_avg(data, 'HomeTeam', 'FTHG', w, 'HGFAvg')
    data = get_mv_avg(data, 'HomeTeam', 'FTAG', w, 'HGAAvg')
    # data = get_mv_avg(data, 'HomeTeam', 'HS', w, 'HSFAvg')
    # data = get_mv_avg(data, 'HomeTeam', 'AS', w, 'HSAAvg')
    # data = get_mv_avg(data, 'HomeTeam', 'HST', w, 'HSTFAvg')
    # data = get_mv_avg(data, 'HomeTeam', 'AST', w, 'HSTAAvg')
    # #
    data = get_mv_avg(data, 'AwayTeam', 'FTAG', w, 'AGFAvg')
    data = get_mv_avg(data, 'AwayTeam', 'FTHG', w, 'AGAAvg')
    # data = get_mv_avg(data, 'AwayTeam', 'AS', w, 'ASFAvg')
    # data = get_mv_avg(data, 'AwayTeam', 'HS', w, 'ASAAvg')
    # data = get_mv_avg(data, 'AwayTeam', 'AST', w, 'ASTFAvg')
    # data = get_mv_avg(data, 'AwayTeam', 'HST', w, 'ASTAAvg')


    # data['HomeStats'] = data.HGFAvg + data.AGAAvg + data.HSFAvg + data.ASAAvg + data.HSTFAvg + data.ASTAAvg
    # data['AwayStats'] = data.AGFAvg + data.HGAAvg + data.ASFAvg + data.HSAAvg + data.ASTFAvg + data.HSTAAvg
    newcols = ['HGFAvg', 'HGAAvg', 'AGFAvg', 'AGAAvg', 'HPAvg', 'APAvg', 'HPMean', 'APMean']
    data = data.dropna(subset=newcols)
    return data
    

def main(i, o, w):
    data = calc_form(i, w)
    data.to_csv(f'{o}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='infile', type=str, help='CSV file to load', required=True)
    parser.add_argument('-w', dest='window', type=int, default=5, required=False)
    args = parser.parse_args()
    infile = args.infile
    window = args.window
    outfile = f'{infile}_avgs'
    main(infile, outfile, window)
