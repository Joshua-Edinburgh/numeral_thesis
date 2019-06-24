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

   
char_mapping = ['a','b','c','d','e','f','g','h','i','j']
#random.shuffle(char_mapping)
char_mapping += '@'

def key_to_value(key, char_mapping,comp = True):
    '''
        Generate value based on key. Now only for NUM_SYSTEM=10, ATTRI_SIZE=3
    '''
    int_key = int(key)
    dig_0 = np.mod(int_key, 10)
    dig_1 = np.mod(int(int_key*0.1), 10)
    dig_2 = np.mod(int(int_key*0.01), 10)
    value = []
    if comp == True:
        value.append(char_mapping[dig_2])
        value.append(char_mapping[dig_1])
        value.append(char_mapping[dig_0])
    else:
        value.append(char_mapping[np.random.randint(0,len(char_mapping))])
        value.append(char_mapping[np.random.randint(0,len(char_mapping))])
        value.append(char_mapping[np.random.randint(0,len(char_mapping))])
        
    return ''.join(value)

# ========== Degenerate language ===================
deg_all = {}
deg_train = {}
deg_valid = {}

for i in range(NUM_SYSTEM**ATTRI_SIZE):
    key = num_to_str(i)
    value = 'aaa'
    deg_all[key] = value
    if i in valid_list:
        deg_valid[key] = value
    elif i in train_list:
        deg_train[key] = value
#compos_cal(deg_valid)
        
# ========== Compositional language ===================
comp_all = {}
comp_train = {}
comp_valid = {}

for i in range(NUM_SYSTEM**ATTRI_SIZE):
    key = num_to_str(i)
    value = key_to_value(key, char_mapping, True)
    comp_all[key] = value
    if i in valid_list:
        comp_valid[key] = value
    elif i in train_list:
        comp_train[key] = value

#compos_cal(comp_valid)

# ========== Holistic language ===================
holi_all = {}
holi_train = {}
holi_valid = {}

for i in range(NUM_SYSTEM**ATTRI_SIZE):
    key = num_to_str(i)
    value = key_to_value(key, char_mapping, False)
    holi_all[key] = value
    if i in valid_list:
        holi_valid[key] = value
    elif i in train_list:
        holi_train[key] = value

#compos_cal(holi_valid)

