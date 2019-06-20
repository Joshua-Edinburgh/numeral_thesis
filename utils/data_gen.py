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

def gen_candidates(low, high, valid_list, batch = BATCH_SIZE, candi = SEL_CANDID):
    s = []
    num = batch*candi
    while (len(s)<num):
        x = np.random.randint(low, high)
        while (x in valid_list):
            x = np.random.randint(low, high)
        s.append(x)
    
    return np.asarray(s).reshape((batch, candi))

valid_num = int(NUM_SYSTEM**ATTRI_SIZE * VALID_RATIO)
valid_list = valid_list(0, 10**ATTRI_SIZE, valid_num)
sel_idx = np.random.randint(0, SEL_CANDID,(BATCH_SIZE,))
data_candidates = gen_candidates(0, 10**ATTRI_SIZE, valid_list)

data_batch = np.zeros((BATCH_SIZE,))
for i in range(BATCH_SIZE):
    data_batch[i] = data_candidates[i,sel_idx[i]]






