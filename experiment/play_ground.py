import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn_pandas import DataFrameMapper
import copy
import torch
import torchtuples as tt
import matplotlib.pyplot as plt


data = pd.read_csv(os.path.join('..', 'data', 'tidy_Stroke_Vital_Sign.csv'))
data_x = data.drop(['LOC', 'UID', 'Hospital_ID', 'SurvivalWeeks', 'admission_date',
                    'discharge_date', 'death_date', 'Mortality', 'CVDeath', 'SurvivalDays'], axis=1)

data_y = data[['Mortality', 'SurvivalWeeks']].copy()
# data_y.loc[:, 'Mortality'] = data_y['Mortality'].astype(bool)
# data_y = np.array(list(data_y.to_records(index=False)))

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25, random_state=369)

get_target = lambda df: (df['SurvivalWeeks'].values, df['Mortality'].values)
y_train = get_target(y_train)
y_test = get_target(y_test)



categorical_columns = ['Sex', 'AF', 'DM', 'HTN', 'Hyperlipidemia', 'CHF', 'Smoking',
                       'Cancer before adm', 'Foley', 'NG', 'ICU']
categorical_ix = [data_x.columns.get_loc(col) for col in categorical_columns]

numerical_ix =  np.setdiff1d(list(range(0, len(data_x.columns))), categorical_ix)
numerical_columns = np.setdiff1d(data_x.columns, categorical_columns).tolist()


scaler = preprocessing.StandardScaler()

standardize = [([col], scaler) for col in numerical_columns]
leave = [(col, None) for col in categorical_columns]

x_mapper = DataFrameMapper(standardize + leave)


X_train = pd.DataFrame(data=x_mapper.fit_transform(X_train),
                       columns=numerical_columns+categorical_columns,
                       index=X_train.index)
# x_test = x_mapper.transform(X_test)




in_features = X_train.shape[1]
num_nodes = [32, 16, 8]
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)

from pycox.models import CoxPH
model = CoxPH(net, tt.optim.Adam)
batch_size = 256
lrfinder = model.lr_finder(X_train.values.astype('float32'), y_train, batch_size, tolerance=10)
_ = lrfinder.plot()
tt.callbacks.EarlyStopping(patience=20)
plt.show()

print()