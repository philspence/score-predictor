import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


def get_mv_avg(df, key, value, win, newcol):
    print(f'Calculating moving average for {newcol}...')
    mv = df.groupby([key, 'Season'])[value].rolling(win, min_periods=3).mean().shift().reset_index()
    df[newcol] = mv.set_index('level_2')[value]
    return df


def get_cummean(df, key, value, win, newcol):
    print(f'Calculating cumulative mean for {newcol}...')
    avg = df.groupby([key, 'Season'])[value].expanding(3).mean().reset_index()
    df[newcol] = avg.set_index('level_2')[value]
    return df


def get_ewm(df, key, value, newcol):
    print(f'Calculating EWM for {newcol}...')
    exp_mean = df.groupby([key, 'Season'])[value].apply(lambda x: x.ewm(span=3, min_periods=3).mean()).shift().reset_index()
    df[newcol] = exp_mean.set_index('index')[value]
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

    data = get_cummean(data, 'HomeTeam', 'HP', w, 'HWinPerc')
    data = get_cummean(data, 'AwayTeam', 'AP', w, 'AWinPerc')

    data = get_mv_avg(data, 'HomeTeam', 'HP', w, 'HP5Avg')
    data = get_mv_avg(data, 'AwayTeam', 'AP', w, 'AP5Avg')
    data = get_ewm(data, 'HomeTeam', 'HP', 'HPEWM')
    data = get_ewm(data, 'AwayTeam', 'AP', 'APEWM')

    data = get_mv_avg(data, 'HomeTeam', 'FTHG', w, 'HGF5Avg')
    data = get_mv_avg(data, 'HomeTeam', 'FTAG', w, 'HGA5Avg')
    data = get_ewm(data, 'HomeTeam', 'FTHG', 'HGFEWM')
    data = get_ewm(data, 'HomeTeam', 'FTAG', 'HGAEWM')
    # data = get_mv_avg(data, 'HomeTeam', 'HS', w, 'HSFAvg')
    # data = get_mv_avg(data, 'HomeTeam', 'AS', w, 'HSAAvg')
    # data = get_mv_avg(data, 'HomeTeam', 'HST', w, 'HSTFAvg')
    # data = get_mv_avg(data, 'HomeTeam', 'AST', w, 'HSTAAvg')
    # #
    data = get_mv_avg(data, 'AwayTeam', 'FTAG', w, 'AGF5Avg')
    data = get_mv_avg(data, 'AwayTeam', 'FTHG', w, 'AGA5Avg')
    data = get_ewm(data, 'AwayTeam', 'FTAG', 'AGFEWM')
    data = get_ewm(data, 'AwayTeam', 'FTHG', 'AGAEWM')
    # data = get_mv_avg(data, 'AwayTeam', 'AS', w, 'ASFAvg')
    # data = get_mv_avg(data, 'AwayTeam', 'HS', w, 'ASAAvg')
    # data = get_mv_avg(data, 'AwayTeam', 'AST', w, 'ASTFAvg')
    # data = get_mv_avg(data, 'AwayTeam', 'HST', w, 'ASTAAvg')


    # data['HomeStats'] = data.HGFAvg + data.AGAAvg + data.HSFAvg + data.ASAAvg + data.HSTFAvg + data.ASTAAvg
    # data['AwayStats'] = data.AGFAvg + data.HGAAvg + data.ASFAvg + data.HSAAvg + data.ASTFAvg + data.HSTAAvg
    newcols = ['HGF5Avg', 'AGF5Avg', 'HP5Avg', 'AP5Avg', 'HWinPerc', 'AWinPerc']
               # 'HSFAvg', 'ASFAvg']
    data = data.dropna(subset=newcols)
    return data
    

def main(i, o, w):
    data = calc_form(i, w)
    data.to_csv(f'{o}.csv')


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', dest='infile', type=str, help='CSV file to load', required=True)
        parser.add_argument('-w', dest='window', type=int, default=5, required=False)
        args = parser.parse_args()
        infile = args.infile
        window = args.window
        outfile = f'{infile}_avgs'
    except:
        infile = 'top_leagues/EPL_merged.csv'
        window = 5
        outfile = f'{infile}_avgs'
    finally:
        main(infile, outfile, window)
