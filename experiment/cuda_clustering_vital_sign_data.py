import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import distython
from tsnecuda import TSNE
from sklearn.cluster import DBSCAN


tidy_data = pd.read_csv('vitalsing_data.csv')

print(tidy_data.shape)
X_data = tidy_data.drop(['ID', 'CHT_NO', 'admin_date', 'discharge_date',
                         'AllMortality', 'CVDeath  ', 'Death Date', 'SurvivalWeeks'], axis=1)
y_data = tidy_data[['SurvivalWeeks']]
y_data = (y_data < 24).astype(int)
categorical_ix = [0, 2, 3, 4, 5, 6, 7, 8]

categorical_columns = X_data.columns[categorical_ix].values
numerical_columns = np.setdiff1d(X_data.columns, categorical_columns)


X_data[numerical_columns] = StandardScaler().fit_transform(X_data[numerical_columns])
# one-hot
X_data_one_hot = pd.get_dummies(X_data, columns=categorical_columns)

tsne_embedding = TSNE(n_components=2, perplexity=15, learning_rate=10, random_seed=369).fit_transform(X_data_one_hot)

plt.scatter(
    tsne_embedding[:, 0],
    tsne_embedding[:, 1],
    c=y_data.values.astype(int), s=1, cmap='Spectral')
plt.gca().set_aspect('equal', 'datalim')

plt.savefig(os.path.join('..', 'result', 'tsne_vitalsign.png'))

tsne_labels = DBSCAN(eps=0.3, min_samples=50).fit_predict(tsne_embedding)
# clustered = (tsne_labels >= 0)
# plt.scatter(tsne_embedding[clustered, 0],
#             tsne_embedding[clustered, 1],
#             c=tsne_labels[clustered],
#             s=1,
#             cmap='Spectral')
# plt.savefig(os.path.join('..', 'result', 'tsne_vitalsign_cluster.png'))
tidy_data['tsne_labels'] = tsne_labels
tidy_data.to_csv(os.path.join('..', 'result', 'vital_cluster.csv'), index=False)