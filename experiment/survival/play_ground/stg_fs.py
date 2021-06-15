from stg import STG
import stg.utils as utils
import numpy as np
import torch
import time
from collections import defaultdict
import os
import h5py

def load_cox_gaussian_data():
    dataset_file = os.path.join('gaussian_survival_data.h5')
    datasets = defaultdict(dict)
    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]

    return datasets


datasets = load_cox_gaussian_data()

train_data = datasets['train']
norm_vals = {
        'mean' : datasets['train']['x'].mean(axis=0),
        'std'  : datasets['train']['x'].std(axis=0)
    }
test_data = datasets['test']

# standardize
train_data = utils.standardize_dataset(datasets['train'], norm_vals['mean'],                                           norm_vals['std'])
valid_data = utils.standardize_dataset(datasets['valid'], norm_vals['mean'],                                           norm_vals['std'])
test_data = utils.standardize_dataset(datasets['test'], norm_vals['mean'],                                            norm_vals['std'])

train_X = train_data['x']
train_y = {'e': train_data['e'], 't': train_data['t']}
valid_X = valid_data['x']
valid_y = {'e': valid_data['e'], 't': valid_data['t']}
test_X = test_data['x']
test_y = {'e': test_data['e'], 't': test_data['t']}

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


device = "cpu"
feature_selection = True

print(train_data['X'].shape[0])

model = STG(task_type='cox', input_dim=train_data['X'].shape[1], output_dim=1, hidden_dims=[60, 20, 3], activation='selu',
    optimizer='Adam', learning_rate=0.0005, batch_size=train_data['X'].shape[0], feature_selection=feature_selection,
    sigma=0.5, lam=0.004, random_state=1, device=device)
#model.save_checkpoint(filename='tmp.pth')

now = time.time()
model.fit(train_data['X'], {'E': train_data['E'], 'T': train_data['T']}, nr_epochs=600,
        valid_X=valid_data['X'], valid_y={'E': valid_data['E'], 'T': valid_data['T']}, print_interval=100)
print("Passed time: {}".format(time.time() - now))

model.evaluate(test_data['X'], {'E': test_data['E'], 'T': test_data['T']})

gates = model.get_gates(mode='prob')

print('done')