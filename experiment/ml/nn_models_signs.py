import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn import preprocessing
from imblearn import over_sampling
from sklearn.metrics import auc, roc_curve
import pickle


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Executing the model on :", device)

ex_data = pd.read_csv(os.path.join('..', '..', 'data', 'tidy_Stroke_Vital_Sign.csv'))[['UID', 'Hospital_ID', 'SurvivalWeeks']]
with open(os.path.join('..', '..', 'data', 'vital_sign_simple.pickle'), 'rb') as f:
    vital_signs = pickle.load(f)
complete_data = pd.merge(ex_data, vital_signs, on=['UID', 'Hospital_ID'])


X_data = complete_data.drop(['UID', 'Hospital_ID', 'SurvivalWeeks'], axis=1)
y_data = complete_data[['SurvivalWeeks']]
y_data_od = (y_data < 4).astype(int)



print('done')