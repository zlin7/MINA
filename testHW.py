from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

import numpy as np
import pickle as dill
from collections import Counter
from time import localtime, strftime
import random

import matplotlib.pyplot as plt
import random
from util import preprocess_physionet, make_data_physionet, make_knowledge_physionet, evaluate
import sys, os
from shutil import copyfile
import ipdb

import model as new_model
import dataloader as dld
import old_model as old_model
from importlib import reload
reload(new_model); reload(old_model); reload(dld)

def train(model, optimizer, loss_func, epoch,
          dataloader,
          log_file):
    """
    X_train: (n_channel, n_sample, n_dim)
    Y_train: (n_sample,)

    K_train_beat: (n_channel, n_sample, n_dim)
    K_train_rhythm: (n_channel, n_sample, n_dim/n_split)
    K_train_freq: (n_channel, n_sample)
    """
    model.train()

    pred_all = []
    loss_all = []
    Y_train = []
    for (X, K_beat, K_rhythm, K_freq), Y in tqdm(dataloader, desc='train', ncols=80):
        X, K_beat, K_rhythm, K_freq = X.cuda(), K_beat.cuda(), K_rhythm.cuda(), K_freq.cuda()
        Y = Y.cuda()

        pred, _ = model.forward_(X, K_beat, K_rhythm, K_freq)

        pred_all.append(pred.cpu().data.numpy())
        Y_train.append(Y.cpu().data.numpy())

        loss = loss_func(pred, Y)
        loss_all.append(loss.cpu().data.numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_res = np.mean(loss_all)
    print('epoch {0} '.format(epoch))
    print('loss ', np.mean(loss_all))
    print('train | ', end='')
    pred_all = np.concatenate(pred_all, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)
    # print(Y_train.shape, pred_all.shape)
    res = evaluate(Y_train, pred_all)
    res.append(loss_res)
    res.append(pred_all)

    with open(log_file, 'a') as fout:
        print('epoch {0} '.format(epoch), 'train | ', res, file=fout)
        print('loss_all ', np.mean(loss_all), file=fout)

    return res


def test(model,
         dataloader,
         log_file):
    model.eval()

    pred_all = []
    att_dic_all = []
    Y_test = []
    for (X, K_beat, K_rhythm, K_freq), Y in tqdm(dataloader, desc='test', ncols=80):
        X, K_beat, K_rhythm, K_freq = X.cuda(), K_beat.cuda(), K_rhythm.cuda(), K_freq.cuda()
        Y = Y.cuda()

        pred, att_dic = model.forward_(X, K_beat, K_rhythm, K_freq)

        for k, v in att_dic.items():
            att_dic[k] = v.cpu().data.numpy()
        att_dic_all.append(att_dic)
        pred_all.append(pred.cpu().data.numpy())
        Y_test.append(Y.cpu().data.numpy())

    print('test | ', end='')
    pred_all = np.concatenate(pred_all, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)
    res = evaluate(Y_test, pred_all)
    res.append(pred_all)

    with open(log_file, 'a') as fout:
        print('test | ', res, file=fout)

    return res, att_dic_all

def load_data(data_path=r"G:\MINA\data\challenge2017"):
    ##################################################################
    ### read data
    ##################################################################
    with open(os.path.join(data_path, 'mina_info.pkl'), 'rb') as fin:
        res = dill.load(fin)
    Y_train = res['Y_train']
    Y_val = res['Y_val']
    Y_test = res['Y_test']
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))

    fin = open(os.path.join(data_path, 'mina_X_train.bin'), 'rb')
    X_train = np.load(fin)
    fin.close()
    fin = open(os.path.join(data_path, 'mina_X_val.bin'), 'rb')
    X_val = np.load(fin)
    fin.close()
    fin = open(os.path.join(data_path, 'mina_X_test.bin'), 'rb')
    X_test = np.load(fin)
    fin.close()
    X_train = np.swapaxes(X_train, 0, 1)
    X_val = np.swapaxes(X_val, 0, 1)
    X_test = np.swapaxes(X_test, 0, 1)
    print(X_train.shape, X_val.shape, X_test.shape)

    fin = open(os.path.join(data_path, 'mina_K_train_beat.bin'), 'rb')
    K_train_beat = np.load(fin)
    fin.close()
    fin = open(os.path.join(data_path, 'mina_K_val_beat.bin'), 'rb')
    K_val_beat = np.load(fin)
    fin.close()
    fin = open(os.path.join(data_path, 'mina_K_test_beat.bin'), 'rb')
    K_test_beat = np.load(fin)
    fin.close()
    with open(os.path.join(data_path, 'mina_knowledge.pkl'), 'rb') as fin:
        res = dill.load(fin)
    K_train_rhythm = res['K_train_rhythm']
    K_train_freq = res['K_train_freq']
    K_val_rhythm = res['K_val_rhythm']
    K_val_freq = res['K_val_freq']
    K_test_rhythm = res['K_test_rhythm']
    K_test_freq = res['K_test_freq']
    K_train_beat = np.swapaxes(K_train_beat, 0, 1)
    K_train_rhythm = np.swapaxes(K_train_rhythm, 0, 1)
    K_train_freq = np.swapaxes(K_train_freq, 0, 1)
    K_val_beat = np.swapaxes(K_val_beat, 0, 1)
    K_val_rhythm = np.swapaxes(K_val_rhythm, 0, 1)
    K_val_freq = np.swapaxes(K_val_freq, 0, 1)
    K_test_beat = np.swapaxes(K_test_beat, 0, 1)
    K_test_rhythm = np.swapaxes(K_test_rhythm, 0, 1)
    K_test_freq = np.swapaxes(K_test_freq, 0, 1)
    print(K_train_beat.shape, K_train_rhythm.shape, K_train_freq.shape)
    print(K_val_beat.shape, K_val_rhythm.shape, K_val_freq.shape)
    print(K_test_beat.shape, K_test_rhythm.shape, K_test_freq.shape)

    print('load data done!')
    return X_train, X_test, Y_train, Y_test, K_train_beat, K_test_beat, K_train_rhythm, K_test_rhythm, K_train_freq, K_test_freq

def make_merged_data(src=r"G:\MINA\data\challenge2017\1000_cached_data",
                     dst=None, seed=7):
    from sklearn.model_selection import train_test_split
    import pandas as pd
    dst = dst or "%s_permuted%d"%(src, seed)
    dst_train = os.path.join(dst, 'train.pkl')
    dst_test =  os.path.join(dst, 'test.pkl')
    if not (os.path.isfile(dst_train) and os.path.isfile(dst_test)):
        X, _, Y, _, K_beat, _, K_rhythm, _, K_freq, _ = load_data(src)
        train_idx, test_idx = train_test_split(np.arange(X.shape[1]), train_size=0.8, random_state=seed)

        train_data = {"X": X[:, train_idx,:], "Y":Y[train_idx],
                      "K_beat": K_beat[:, train_idx, :], "K_rhythm":K_rhythm[:, train_idx, :], "K_freq":K_freq[:, train_idx,:]}
        test_data = {"X": X[:, test_idx,:], "Y": Y[test_idx],
                      "K_beat": K_beat[:, test_idx,:], "K_rhythm": K_rhythm[:, test_idx,:], "K_freq": K_freq[:, test_idx,:]}
        print([train_data[k].shape for k in train_data])
        print([test_data[k].shape for k in test_data])
        if not os.path.isdir(dst): os.makedirs(dst)
        pd.to_pickle(train_data, dst_train)
        pd.to_pickle(test_data, dst_test)
    return pd.read_pickle(dst_train), pd.read_pickle(dst_test)

def load_permuted_data(data_path=r'G:\MINA\data\challenge2017\1000_cached_data_permuted7', which='train'):
    import pandas as pd
    d = pd.read_pickle(os.path.join(data_path, '%s.pkl'%which))
    return d['X'], d['Y'], d['K_beat'], d['K_rhythm'], d['K_freq']

def run_exp(data_path):
    #n_epoch = 200
    n_epoch=5
    lr = 0.003
    n_split = 50

    ##################################################################
    ### par
    ##################################################################
    #run_id = 'mina_{0}'.format(strftime("%Y-%m-%d-%H-%M-%S", localtime()))
    run_id = 'test_run'
    directory = 'res/{0}'.format(run_id)
    if not os.path.isdir(directory):os.makedirs(directory)
    #try:
    #    os.stat('res/')
    #except:
    #    os.mkdir('res/')
    #try:
    #    os.stat(directory)
    #except:
    #    os.mkdir(directory)

    log_file = '{0}/log.txt'.format(directory)
    #model_file = 'testHW.py'
    #copyfile(model_file, '{0}/{1}'.format(directory, model_file))

    n_dim = 3000
    batch_size = 128

    with open(log_file, 'a') as fout:
        print(run_id, file=fout)

    ##################################################################
    ### read data
    ##################################################################
    #X_train, X_test, Y_train, Y_test, K_train_beat, K_test_beat, K_train_rhythm, K_test_rhythm, K_train_freq, K_test_freq = load_data(data_path)
    train_loader = dld.get_dataloader(data_path, 'train')
    test_loader = dld.get_dataloader(data_path, 'test')
    ##################################################################
    ### train
    ##################################################################

    n_channel = train_loader.dataset[0][0][0].shape[0]
    print('n_channel:', n_channel)

    torch.cuda.manual_seed(0)

    #model = old_model.Net(n_channel, n_dim, n_split)
    model = new_model.NetFreq(n_channel, n_dim, n_split)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    train_res_list = []
    test_res_list = []
    test_att_list = []
    for epoch in range(n_epoch):
        tmp_train = train(model, optimizer, loss_func, epoch,
                          train_loader,
                          log_file)
        tmp_test, tmp_att_test = test(model,
                                      test_loader,
                                      log_file)

        train_res_list.append(tmp_train)
        test_res_list.append(tmp_test)
        test_att_list.append(tmp_att_test)
        #torch.save(model, '{0}/model_{1}.pt'.format(directory, epoch))

    ##################################################################
    ### save results
    ##################################################################
    res_mat = []
    for i in range(n_epoch):
        train_res = train_res_list[i]
        test_res = test_res_list[i]
        res_mat.append([
            train_res[0], train_res[1],
            test_res[0], test_res[1]])
    res_mat = np.array(res_mat)

    res = {'train_res_list': train_res_list,
           'test_res_list': test_res_list}
    with open('{0}/res.pkl'.format(directory), 'wb') as fout:
        dill.dump(res, fout)

    np.savetxt('{0}/res_mat.csv'.format(directory), res_mat, delimiter=',')

    try:
        res = {'test_att_list': test_att_list}
        with open('{0}/res_att.pkl'.format(directory), 'wb') as fout:
            dill.dump(res, fout)
    except:
        print('error in saving attention file')


if __name__ == '__main__':
    #run_exp('../data/challenge2017/')
    run_exp('../data/challenge2017/1000_cached_data_permuted7')
    #make_merged_data()