import pandas as pd
import random

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def test(st):
    return st+'1'

def data_preprocess(data):

    # drop unwanted features
    data.drop(['winner_seed', 'winner_entry', 'loser_seed', 'loser_entry', 'tourney_id', 'tourney_name', 'tourney_date', 'match_num', 'score'], axis=1, inplace=True)
    
    # drop null
    data.dropna(inplace=True)
    
    # Converting to Categorical features
    for col in ['surface', 'draw_size', 'tourney_level', 'winner_hand', 'winner_ioc', 'loser_hand', 'loser_ioc', 'best_of', 'round', ]:
        data[col] = pd.Categorical(data[col]).codes

    #TODO: bin age and/or height (optional)

    # Replacing 'winner' -> 'player0' and 'loser' -> 'player1'
    ren_col = []
    for col in data.columns:
        if col=='draw_size':
            ren_col.append('draw-size')     # Rename 'draw_size' to 'draw-size'
            continue
        ren_col.append(col.replace('winner_', 'player0_').replace('loser_', 'player1_').replace('w_', 'p0_').replace('l_', 'p1_'))
    data.columns = ren_col

    # Add 'winner' column
    data['winner'] = 0

    # Switch some winners and losers to obtain winner '0's for model to train
    for i in range(data.shape[0]):
        if random.random() > 0.5:
            data.iloc[i, 3:11], data.iloc[i, 11:19] = data.iloc[i, 11:19], data.iloc[i, 3:11]
            data.iloc[i, 22:31], data.iloc[i, 31:40] = data.iloc[i, 31:40], data.iloc[i, 22:31]
            data.iloc[i, -1] = 1
    
    # Find floats to standardize
    floats = []
    for col in data.columns:
        if data[col].dtype == 'float64':
            floats.append(col)
    # Scale data
    scaler = MinMaxScaler()
    data[floats] = scaler.fit_transform(data[floats])

    return data