#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:03:06 2019

@author: xiayezi
"""
import sys
sys.path.append("..")
import numpy as np
from utils.conf import *


def valid_list(low, high, num):
    '''
        Randomly generate distinct numbers, range in (low, high), with size.
    '''
    s = []
    while(len(s)<num):
        x = np.random.randint(low, high)
        if x not in s:
            s.append(x)    
    return s


def gen_distinct_candidates(tgt_list, train_list, candi_size = SEL_CANDID):
    '''
        tgt_list may contain part of elements in train_list
        output the (data_candidates, sel_idx)
    '''
    batch_size = len(tgt_list)
    data_candidates = np.zeros((batch_size, candi_size))
    sel_idx = []
    for i in range(batch_size):
        tmp_idx = np.random.randint(0, candi_size)
        sel_idx.append(tmp_idx)
        for j in range(candi_size):
            if j == 0:
                data_candidates[i,j]=tgt_list[i]
                continue
            rand_candi = random.choice(train_list)
            while (rand_candi in data_candidates[i,:]):
                rand_candi = random.choice(train_list)
            data_candidates[i, j] = rand_candi
        data_candidates[i, 0] = data_candidates[i, tmp_idx]
        data_candidates[i, tmp_idx] = tgt_list[i]
    
    return data_candidates, np.asarray(sel_idx)


def gen_candidates(low, high, valid_list, batch = BATCH_SIZE, candi = SEL_CANDID, train=True):
    if train == True:
        s = []
        num = batch*candi
        while (len(s)<num):
            x = np.random.randint(low, high)
            while (x in valid_list):
                x = np.random.randint(low, high)
            s.append(x)
        return np.asarray(s).reshape((batch, candi))
    elif train == False:
        s = []
        valid_num = len(valid_list)
        while (len(s)<valid_num*candi):
            x = np.random.randint(0,valid_num)
            s.append(valid_list[x])
        return np.asarray(s).reshape((valid_num, candi))

valid_num = int(NUM_SYSTEM**ATTRI_SIZE * VALID_RATIO)
valid_list = valid_list(0, NUM_SYSTEM**ATTRI_SIZE, valid_num)
train_list = list(set([i for i in range(NUM_SYSTEM**ATTRI_SIZE)]) ^ set(valid_list))

def valid_data_gen():
    sel_idx_val = np.random.randint(0,SEL_CANDID, (len(valid_list),))
    valid_candidates = gen_candidates(0, NUM_SYSTEM**ATTRI_SIZE, valid_list, train=False)
    valid_full = np.zeros((valid_num,))
    
    for i in range(valid_num):
        valid_full[i] = valid_candidates[i, sel_idx_val[i]]    
    
    return valid_full, valid_candidates, sel_idx_val


def batch_data_gen():
    num_batches = int(len(train_list)/BATCH_SIZE) # Here we assume batch size=x*100 first
    random.shuffle(train_list)
    
    batch_list = []
    
    for i in range(num_batches):
        one_batch = {}
        tmp_list = train_list[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        train_candidates, sel_idx_train = gen_distinct_candidates(tmp_list, train_list)
        data_batch = np.zeros((BATCH_SIZE,))
        for i in range(BATCH_SIZE):
            train_candidates[i,sel_idx_train[i]] = tmp_list[i]
            
        one_batch['sel_idx'] = sel_idx_train
        one_batch['candidates'] = train_candidates
        one_batch['data'] = np.asarray(tmp_list)
        batch_list.append(one_batch)
    return batch_list

def shuffle_batch(batch_list):
    '''
        Shuffle the order of data in the same batch.
    '''
    shuf_batch_list = []
    for j in range(len(batch_list)):
        tmp_batch = {}
        train_batch, train_candidates, sel_idx_train = batch_list[j]['data'], batch_list[j]['candidates'], batch_list[j]['sel_idx']
        train_batch
        tmp = np.concatenate((train_batch.reshape((-1,1)),
                              train_candidates,
                              sel_idx_train.reshape((-1,1))),axis=1)
        np.random.shuffle(tmp)
        tmp_batch['data'] = tmp[:,0]
        tmp_batch['candidates'] = tmp[:,1:-1]
        tmp_batch['sel_idx'] = tmp[:,-1]
        shuf_batch_list.append(tmp_batch)
    return shuf_batch_list
        
        



batch_list = batch_data_gen()
shuf_batch_list = shuffle_batch(batch_list)
batch_list = batch_data_gen()


