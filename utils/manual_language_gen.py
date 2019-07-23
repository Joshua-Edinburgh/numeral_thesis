#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:01:32 2019

@author: xiayezi
"""
import sys 
sys.path.append("..") 
import numpy as np
from utils.conf import *
from utils.data_gen import *
from utils.result_record import *
import matplotlib.pyplot as plt

vocab_table_full = [chr(97+int(v)) for v in range(26)]

char_mapping = ['a','b','c','d','e','f','g','h','i','j']
#random.shuffle(char_mapping)

def value_to_onehot(value, char_mapping):
    '''
        Map value to one-hot tensor. Shape is [ATTRI_SIZE, MSG_VOCSIZE]
    '''
    msg_onehot = torch.zeros((ATTRI_SIZE, MSG_VOCSIZE))
    tmp_idx = 0
    for i in range(len(value)):
        tmp_idx = char_mapping.index(value[i])
        msg_onehot[i,tmp_idx] = 1
    
    return msg_onehot

def key_to_value(key, char_mapping,comp = True):
    '''
        Generate value based on key. Now only for NUM_SYSTEM=10, ATTRI_SIZE=2
    '''
    key[0]
    tmp = ''.join([s for s in key])
    int_key = int(tmp)
    dig_0 = int(key[0])
    dig_1 = int(key[1])
    #dig_2 = np.mod(int(int_key/NUM_SYSTEM**2), NUM_SYSTEM)
    value = []
    if comp == True:
        #value.append(char_mapping[dig_2])
        value.append(char_mapping[dig_1])
        value.append(char_mapping[dig_0])
    else:
        #value.append(char_mapping[np.random.randint(0,len(char_mapping))])
        value.append(char_mapping[np.random.randint(0,len(char_mapping))])
        value.append(char_mapping[np.random.randint(0,len(char_mapping))])
        
    return ''.join(value)

# ========== Degenerate language ===================
deg_all = {}
deg_train = {}
deg_valid = {}

deg_spk_train = {}  # Data for spk training, 'data' should be dicimal, 'msg' one hot
data_list = []
msg_list = []
for i in range(NUM_SYSTEM**ATTRI_SIZE):
    # ===== For dictionary version
    key = num_to_tup(i)
    value = 'aa'
    deg_all[key] = value
    if i in valid_list:
        deg_valid[key] = value
    elif i in train_list:
        deg_train[key] = value
    # ==== For spk training version
    msg_list.append(value_to_onehot(value, char_mapping))
    data_list.append(i)
    
deg_spk_train['data'] = np.asarray(data_list)
deg_spk_train['msg'] = torch.stack(msg_list).transpose(0,1)
    
    
#compos_cal(deg_all)    # Should be approximate 0
        
# ========== Compositional language ===================
comp_all = {}
comp_train = {}
comp_valid = {}

comp_spk_train = {}  # Data for spk training, 'data' should be dicimal, 'msg' one hot
data_list = []
msg_list = []
for i in range(NUM_SYSTEM**ATTRI_SIZE):
    # ===== For dictionary version
    key = num_to_tup(i)
    value = key_to_value(key, char_mapping, True)
    comp_all[key] = value
    comp_valid[key] = value
    # ==== For spk training version
    msg_list.append(value_to_onehot(value, char_mapping))
    data_list.append(i)

comp_spk_train['data'] = np.asarray(data_list)
comp_spk_train['msg'] = torch.stack(msg_list).transpose(0,1)

#compos_cal(comp_all)   # Should approximate 1.


# ========== Holistic language ===================
holi_all = {}
holi_train = {}
holi_valid = {}

holi_spk_train = {}  # Data for spk training, 'data' should be dicimal, 'msg' one hot
data_list = []
msg_list = []

key_all_list = list(comp_all.keys())
value_all_list = list(comp_all.values())
random.shuffle(value_all_list)

for i in range(NUM_SYSTEM**ATTRI_SIZE):
    # ===== For dictionary version
    key = key_all_list[i]
    value = value_all_list[i]
    holi_all[key] = value
    holi_valid[key] = value
    # ==== For spk training version
    msg_list.append(value_to_onehot(value, char_mapping))
    data_list.append(i)    

holi_spk_train['data'] = np.asarray(data_list)
holi_spk_train['msg'] = torch.stack(msg_list).transpose(0,1)
#compos_cal(holi_all)    # Should be smaller than 1



# =================== Manual Language For the listener ========================

def get_lis_curve_msg(lis_curve_batch_ls, language_train):
    '''
        Input is lis_curve_batch [N_B,1]. language should use the *_train version
        Output has the same structure with *_train
        The function only add lis_train['msg'] part
    '''
    lis_train = lis_curve_batch_ls[0]
    tmp_data = lis_train['data']
    msg_table = language_train['msg'].transpose(0,1)
    msg_list = []
    for i in range(tmp_data.shape[0]):   
        tmp_msg = msg_table[tmp_data[i]]
        msg_list.append(tmp_msg)
    lis_train['msg'] = torch.stack(msg_list).transpose(0,1)
    return lis_train 



#comp_p,_, all_msg = compos_cal_inner(comp_spk_train['msg'],comp_spk_train['data'])





'''
test_msg = {}
for i in range(100):
    tmp = []
    key = num_to_tup(i,2)
    dig_0 = np.mod(i, 10)
    dig_1 = np.mod(int(i*0.1),10)
    tmp = [char_mapping[dig_0], char_mapping[dig_1]]
    value = ''.join(tmp)
    test_msg[key] = value
    
compos_cal(test_msg)

simple_msg = {}
simple_msg['0','0'] = 'aa'
simple_msg['0','1'] = 'ab'
simple_msg['1','0'] = 'ba'
simple_msg['1','1'] = 'bb'
compos_cal(simple_msg)

msg = {}
msg['green','box'] = 'aa'     
msg['blue','box'] = 'ba'
msg['green','circle'] = 'ab'      
msg['blue','circle'] = 'bb'
compos_cal(msg) 
'''
