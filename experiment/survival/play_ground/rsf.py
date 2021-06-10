import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
import matplotlib.pyplot as plt

data = pd.read_csv(os.path.join('../..', '..', 'data', 'tidy_Stroke_Vital_Sign.csv'))
data_x = data.drop(['UID', 'Hospital_ID', 'SurvivalWeeks', 'admission_date',
                    'discharge_date', 'death_date', 'Mortality', 'CVDeath'], axis=1)
categorical_ix = [0, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17]
categorical_columns = data_x.columns[categorical_ix].values
data_x_one_hot = pd.get_dummies(data_x, columns=categorical_columns)

data_y = data[['Mortality', 'SurvivalWeeks']]
data_y['Mortality'] = data_y['Mortality'].astype(bool)
data_y = np.array(list(data_y.to_records(index=False)))

X_train, X_test, y_train, y_test = train_test_split(
    data_x_one_hot, data_y, test_size=0.25, random_state=369)

a = X_test[X_test['AF_1.0'] == 1].iloc[0:3, :]
demo_x = pd.concat([X_test[X_test['AF_1.0'] == 1].iloc[0:3, :],
                    X_test[X_test['AF_1.0'] == 0].iloc[0:3, :]])

rsf = RandomSurvivalForest(n_estimators=1000,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           max_features="sqrt",
                           n_jobs=-1,
                           random_state=369)
rsf.fit(X_train, y_train)

print(rsf.score(X_test, y_test))

print('done')