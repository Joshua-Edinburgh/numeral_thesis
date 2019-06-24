#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:03:06 2019

@author: xiayezi
"""
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
valid_list = valid_list(0, 10**ATTRI_SIZE, valid_num)
train_list = list(set([i for i in range(10**ATTRI_SIZE)]) ^ set(valid_list))



sel_idx_train = np.random.randint(0, SEL_CANDID,(BATCH_SIZE,))
sel_idx_val = np.random.randint(0,SEL_CANDID, (len(valid_list),))
train_candidates = gen_candidates(0, 10**ATTRI_SIZE, valid_list, train=True)
valid_candidates = gen_candidates(0, 10**ATTRI_SIZE, valid_list, train=False)

train_batch = np.zeros((BATCH_SIZE,))
for i in range(BATCH_SIZE):
    train_batch[i] = train_candidates[i,sel_idx_train[i]]

valid_full = np.zeros((valid_num,))
for i in range(valid_num):
    valid_full[i] = valid_candidates[i, sel_idx_val[i]]




