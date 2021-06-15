import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sksurv.column import encode_categorical

from sklearn import preprocessing
from sklearn_pandas import DataFrameMapper
from stg import STG
import stg.utils as utils

def data_processing(data_df):
    data_df_x = data_df.drop(['LOC', 'UID', 'Hospital_ID', 'SurvivalWeeks', 'admission_date',
                              'discharge_date', 'death_date', 'Mortality', 'CVDeath', 'SurvivalDays', 'CAD'], axis=1)

    data_df_y = data_df[['Mortality', 'SurvivalWeeks']]

    data_df_x = data_df_x.drop(['ICU'], axis=1)

    X_temp = data_df_x[(data_df.LOC == '3') | (data_df.LOC == '2') | (data_df.LOC == '6')]
    y_temp = data_df_y[(data_df.LOC == '3') | (data_df.LOC == '2') | (data_df.LOC == '6')]
    X_df_train, X_df_val, y_df_train, y_df_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=369)

    X_df_test_kao = data_df_x[data_df.LOC == '8']
    y_df_test_kao = data_df_y[data_df.LOC == '8']

    categorical_columns = ['Sex', 'AF', 'DM', 'HTN', 'Hyperlipidemia', 'CHF', 'Smoking',
                           'Cancer.before.adm', 'Foley', 'NG', 'Dyslipidemia']
    numerical_columns = np.setdiff1d(data_df_x.columns, categorical_columns).tolist()

    categorical_ix = [data_df_x.columns.get_loc(col) for col in categorical_columns]
    numerical_ix = np.setdiff1d(list(range(0, len(data_df_x.columns))), categorical_ix).tolist()

    scaler = preprocessing.StandardScaler()

    standardize = [([col], scaler) for col in numerical_columns]
    leave = [(col, None) for col in categorical_columns]

    x_mapper = DataFrameMapper(standardize + leave)

    X_df_train = pd.DataFrame(data=x_mapper.fit_transform(X_df_train),
                              columns=numerical_columns + categorical_columns,
                              index=X_df_train.index)

    X_df_val = pd.DataFrame(data=x_mapper.fit_transform(X_df_val),
                            columns=numerical_columns + categorical_columns,
                            index=X_df_val.index)

    X_df_test_kao = pd.DataFrame(data=x_mapper.fit_transform(X_df_test_kao),
                                 columns=numerical_columns + categorical_columns,
                                 index=X_df_test_kao.index)

    X_df_train = encode_categorical(X_df_train, columns=categorical_columns)
    X_df_val = encode_categorical(X_df_val, columns=categorical_columns)
    X_df_test_kao = encode_categorical(X_df_test_kao, columns=categorical_columns)

    return X_df_train, X_df_val, y_df_train, y_df_val, X_df_test_kao, y_df_test_kao



data = pd.read_csv(os.path.join('..', 'data', '(v2)STROKE_VITAL_SIGN_MICE.csv'))
X_train, X_val, y_train, y_val, X_test_kao, y_test_kao = data_processing(data)

train_X = X_train.values
train_y = {'e': y_train['Mortality'].values, 't': y_train['SurvivalWeeks'].values}
valid_X = X_val.values
valid_y = {'e': y_val['Mortality'].values, 't': y_val['SurvivalWeeks'].values}
test_X = X_test_kao.values
test_y = {'e': y_test_kao['Mortality'].values, 't': y_test_kao['SurvivalWeeks'].values}


train_data={}
train_data['X'], train_data['E'], \
        train_data['T'] = utils.prepare_data(train_X, train_y)
train_data['ties'] = 'noties'

valid_data={}
valid_data['X'], valid_data['E'], \
        valid_data['T'] = utils.prepare_data(valid_X, valid_y)
valid_data['ties'] = 'noties'

test_data = {}
test_data['X'], test_data['E'], \
        test_data['T'] = utils.prepare_data(test_X, test_y)
test_data['ties'] = 'noties'


model = STG(task_type='cox', input_dim=train_data['X'].shape[1], output_dim=1, hidden_dims=[60, 20, 3], activation='selu',
    optimizer='Adam', learning_rate=0.0005, batch_size=train_data['X'].shape[0], feature_selection=True,
    sigma=0.5, lam=0.004, random_state=1, device='cpu')

model.fit(train_data['X'], {'E': train_data['E'], 'T': train_data['T']}, nr_epochs=600,
        valid_X=valid_data['X'], valid_y={'E': valid_data['E'], 'T': valid_data['T']}, print_interval=100)


model.evaluate(test_data['X'], {'E': test_data['E'], 'T': test_data['T']})

gates = model.get_gates(mode='prob')

print()