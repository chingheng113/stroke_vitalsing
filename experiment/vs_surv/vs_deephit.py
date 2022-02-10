import os
import torch
import torch.nn as nn
import random
import numpy as np
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchtuples as tt
from pycox import models
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
from sklearn.model_selection import train_test_split

random.seed(1)
np.random.seed(1)
torch.cuda.manual_seed(1)


class BasicBlock1d(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet1d(nn.Module):
    def __init__(self, block, layers, input_channels=12, inplanes=64, num_classes=2):
        super(ResNet1d, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock1d, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock1d, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1d, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1d, 512, layers[3], stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.adaptiveavgpool(x)
        x2 = self.adaptivemaxpool(x)
        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.size(0), x.size(1))
        return self.fc(x) # remove last layer of classifier


def resnet18(args):
    model = ResNet1d(BasicBlock1d, layers=[2, 2, 2, 2], input_channels=args.input_channels, num_classes=args.durations)
    return model


def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--which_gpu', default='none',
                        help='which_gpu')

    parser.add_argument('--model_name', default='vs_deephit.net',
                        help='model name')

    parser.add_argument('--signal_dataset_path',
                        default='D:\\Google Drive\\CMRP\\Others\\Stroke vital sign\\Data\\unique_vital_signs',
                        help='train dataset name')

    parser.add_argument('--table_path', default='D:\\Google Drive\\CMRP\\Others\\Stroke vital sign\\Data\\unique_vital_signs\\stroke.csv',
                        help='train dataset name')

    parser.add_argument('--save_path',
                        default=os.path.join('models'),
                        help='save path')

    parser.add_argument('--durations', default=96, type=int,
                        help='DeepHit duration')

    parser.add_argument('--sample_ratio', default=1.0, type=float,
                        help='sample ratio')

    parser.add_argument('--input_channels', default=6, type=int,
                        help='input channels')

    parser.add_argument('--epochs', default=40, type=int,
                        help='number of total epochs to run')

    parser.add_argument('--train_batch_size', default=256, type=int,
                        help='mini-batch size')

    parser.add_argument('--val_batch_size', default=256, type=int,
                        help='mini-batch size')

    parser.add_argument('--test_batch_size', default=256, type=int,
                        help='mini-batch size')

    parser.add_argument('--lr', default=1e-4, type=float,
                        help='initial learning rate')

    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay')
    return parser


def read_dataset(singnal_path, table_path, time_name, event_name, sampling_ratio):
    data_all = pd.read_csv(table_path)
    if sampling_ratio < 1.:
        # use less number of data
        data_all = data_all.groupby(event_name).apply(lambda x: x.sample(int(data_all[data_all.Mortality == 1].shape[0]*sampling_ratio)))
    data_path = list()
    times = list(data_all[time_name])
    events = list(data_all[event_name])
    for i in range(len(times)):
        d_path = data_all.loc[data_all.index[[i]], 'UID'].values[0]
        data_path.append(os.path.join(singnal_path, d_path+'.csv'))
    return data_path, times, events


def label_transfer(times, events):
    # times = [period if x == -1 else x for x in times]
    # times = [period if x > period else x for x in times]
    # events = [True if x == 'abnormal' else x for x in events]
    # events = [False if x == 'normal' else x for x in events]
    labels = tt.tuplefy(np.array(times), np.array(events))
    return labels


class VsDatasetBatch(Dataset):
    def __init__(self, data_path, time, event):
        self.data_path = data_path
        self.time, self.event = tt.tuplefy(time, event)

    def __len__(self):
        return len(self.time)

    def __getitem__(self, index):
        if not hasattr(index, '__iter__'):
            index = [index]
        data_array = []
        for i in index:
            data_path = self.data_path[i]
            vs = pd.read_csv(data_path)
            vs = vs.drop(vs.columns[[0, 1, 2]], axis=1)
            data = np.array(vs).astype('float32')
            data = torch.from_numpy(data)
            data_array.append(data)
        data_array = torch.stack(data_array)

        rank_mat = models.data.pair_rank_mat(self.time[index], self.event[index])
        target = tt.tuplefy(self.time[index], self.event[index], rank_mat).to_tensor()
        a = tt.tuplefy(data_array, target)
        return tt.tuplefy(data_array, target)


class VsTestInput(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        data_path = self.data_path[index]
        vs = pd.read_csv(data_path)
        vs = vs.drop(vs.columns[[0, 1, 2]], axis=1)
        data = np.array(vs).astype('float32')
        data = torch.from_numpy(data)
        return data


def save_args(model_save_path, ag):
    with open(os.path.join(model_save_path, 'args.txt'), 'w') as f:
        for arg in vars(ag):
            print('%s: %s' %(arg, getattr(ag, arg)), file=f)


def save_cindex(model_save_path, c):
    with open(os.path.join(model_save_path, 'test_c.txt'), 'w') as f:
        print(c, file=f)


def main():
    parser = setup_parser()
    args = parser.parse_args()

    if args.which_gpu != 'none':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpu

    # save setting
    if not os.path.exists(os.path.join(args.save_path, args.model_name)):
        os.mkdir(os.path.join(args.save_path, args.model_name))

    # label transform
    labtrans = DeepHitSingle.label_transform(args.durations)

    # data reading seeting
    singnal_data_path = args.signal_dataset_path
    table_path = args.table_path
    time_col = 'SurvivalDays'
    event_col = 'Mortality'

    # dataset
    data_pathes, times, events = read_dataset(singnal_data_path, table_path, time_col, event_col, args.sample_ratio)

    data_pathes_train, data_pathes_test, times_train, times_test, events_train, events_test = train_test_split(data_pathes, times, events, test_size=0.3, random_state=369)
    data_pathes_train, data_pathes_val, times_train, times_val, events_train, events_val = train_test_split(data_pathes_train, times_train, events_train, test_size=0.2, random_state=369)

    labels_train = label_transfer(times_train, events_train)
    target_train = labtrans.fit_transform(*labels_train)
    dataset_train = VsDatasetBatch(data_pathes_train, *target_train)
    dl_train = tt.data.DataLoaderBatch(dataset_train, args.train_batch_size, shuffle=True)

    labels_val = label_transfer(times_val, events_val)
    target_val = labtrans.transform(*labels_val)
    dataset_val = VsDatasetBatch(data_pathes_val, *target_val)
    dl_val = tt.data.DataLoaderBatch(dataset_val, args.train_batch_size, shuffle=True)

    labels_test = label_transfer(times_test, events_test)
    dataset_test_x = VsTestInput(data_pathes_test)
    dl_test_x = DataLoader(dataset_test_x, args.test_batch_size, shuffle=False)

    net = resnet18(args)
    model = DeepHitSingle(net, tt.optim.Adam(lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False),
                           duration_index=labtrans.cuts)
    # callbacks = [tt.cb.EarlyStopping(patience=15)]
    callbacks = [tt.cb.BestWeights(file_path=os.path.join(args.save_path, args.model_name, args.model_name+'_bestWeight'), rm_file=False )]
    verbose = True
    model_log = model.fit_dataloader(dl_train, args.epochs, callbacks, verbose, val_dataloader=dl_val)

    save_args(os.path.join(args.save_path, args.model_name), args)
    model_log.to_pandas().to_csv(os.path.join(args.save_path, args.model_name, 'loss.csv'), index=False)
    model.save_net(path=os.path.join(args.save_path, args.model_name, args.model_name+'_final'))
    surv = model.predict_surv_df(dl_test_x)
    surv.to_csv(os.path.join(args.save_path, args.model_name, 'test_sur_df.csv'), index=False)
    ev = EvalSurv(surv, *labels_test, 'km')
    print(ev.concordance_td())
    save_cindex(os.path.join(args.save_path, args.model_name), ev.concordance_td())
    print('done')


if __name__ == "__main__":
    main()