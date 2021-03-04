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
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Executing the model on :",device)

ex_data = pd.read_csv(os.path.join('../..', 'data', 'ex_data.csv'))

ex_data['admin_date'] = ex_data['admin_date'].astype(int).astype(str)
in_date = pd.to_datetime(ex_data['admin_date'], format='%Y/%m/%d', errors='coerce')

ex_data['discharge_date'] = ex_data['discharge_date'].astype(int).astype(str)
out_date = pd.to_datetime(ex_data['discharge_date'], format='%Y/%m/%d', errors='coerce')

day_diff = out_date - in_date
ex_data['duration'] = day_diff.dt.days

y_data = ex_data[['SurvivalWeeks']]
X_data = ex_data.drop(['ID', 'CHT_NO', 'admin_date', 'discharge_date',
                       'AllMortality', 'CVDeath  ', 'Death Date', 'SurvivalWeeks'], axis=1)

categorical_columns = ['Sex', 'AF', 'DM', 'HTN', 'Dyslipidemia', 'CHF', 'Smoking', 'Cancer before adm']
numerical_columns = np.setdiff1d(X_data.columns, categorical_columns)

# one-hot
X_data_one_hot = pd.get_dummies(X_data, columns=categorical_columns)
y_data_od = (y_data < 24).astype(int)


class DNN (nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(X_data_one_hot.shape[1], 15)
        self.fc2 = nn.Linear(15, 7)
        self.fc3 = nn.Linear(7, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = DNN().cuda()

print(model)

# Settings
epochs = 10
batch_size = 128
lr = 5e-4

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = nn.MSELoss()


all_auroc = []
for train_index, test_index in KFold(n_splits=10, random_state=42, shuffle=True).split(X_data_one_hot):
    X_train, X_test = X_data_one_hot.iloc[train_index], X_data_one_hot.iloc[test_index]
    y_train, y_test = y_data_od.iloc[train_index], y_data_od.iloc[test_index]

    # scaling
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # over-sampling
    # print('before', y_train.groupby(['SurvivalWeeks']).size())
    sm = over_sampling.SVMSMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    # print('after', y_train.groupby(['SurvivalWeeks']).size())

    # DataLoader
    train_xt = torch.from_numpy(X_train.astype(np.float32)).cuda(device)
    train_yt = torch.from_numpy(y_train.values.astype(np.float32)).cuda(device)
    train_data = Data.TensorDataset(train_xt, train_yt)
    train_loader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_xt = torch.from_numpy(X_test.astype(np.float32)).cuda(device)
    test_yt = torch.from_numpy(y_test.values.astype(np.float32)).cuda(device)

    train_loss_all = []

    model.train()
    for epoch in range(epochs):
        for step, (inputs, labels) in enumerate(train_loader):
            output = model(inputs)
            train_loss = loss_function(output, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_loss_all.append(train_loss.item())
        print("train epoch %d, loss %s:" % (epoch + 1, train_loss.item()))
    print('training complete')

    model.eval()
    y_pred = model(test_xt).cpu().data.numpy()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auroc = auc(fpr, tpr)
    print('auc', auroc)
    all_auroc.append(auroc)
    # plt.figure()
    # plt.plot(train_loss_all, "g-")
    # plt.title("DNN: Train loss per iteration")
    # plt.show()
print(np.mean(all_auroc), np.std(all_auroc))