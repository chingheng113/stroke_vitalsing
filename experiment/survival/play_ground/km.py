from sksurv.datasets import load_veterans_lung_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
import os

data = pd.read_csv(os.path.join('../..', '..', 'data', 'tidy_Stroke_Vital_Sign.csv'))
data_x = data.drop(['UID', 'Hospital_ID', 'SurvivalWeeks', 'admission_date', 'discharge_date', 'death_date'], axis=1)
# data_x = data[['Smoking']]
data_y = data[['Mortality', 'SurvivalWeeks']]
data_y['Mortality'] = data_y['Mortality'].astype(bool)


# KM-All survival
time, survival_prob = kaplan_meier_estimator(data_y['Mortality'], data_y['SurvivalWeeks'])
plt.step(time, survival_prob, where="post")
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.show()

# KM-AF survival
data_af = data_y[data_x.AF == 1]
data_non_af = data_y[data_x.AF == 0]
af_time, af_survival_prob = kaplan_meier_estimator(data_af['Mortality'], data_af['SurvivalWeeks'])
plt.step(af_time, af_survival_prob, where="post", label="AF")
non_af_time, non_af_survival_prob = kaplan_meier_estimator(data_non_af['Mortality'], data_non_af['SurvivalWeeks'])
plt.step(non_af_time, non_af_survival_prob, where="post", label="Non AF")
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")
plt.show()





print('done')