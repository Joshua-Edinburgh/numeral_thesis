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

vocab_table_full = [chr(97+int(v)) for v in range(26)]

char_mapping = ['a','b','c','d','e','f','g','h','i','j']
#random.shuffle(char_mapping)

def key_to_value(key, char_mapping,comp = True):
    '''
        Generate value based on key. Now only for NUM_SYSTEM=10, ATTRI_SIZE=3
    '''
    tmp = ''.join([s for s in key])
    int_key = int(tmp)
    dig_0 = np.mod(int_key, NUM_SYSTEM)
    dig_1 = np.mod(int(int_key/NUM_SYSTEM), NUM_SYSTEM)
    dig_2 = np.mod(int(int_key/NUM_SYSTEM**2), NUM_SYSTEM)
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
    key = num_to_tup(i)
    value = 'aaa'
    deg_all[key] = value
    if i in valid_list:
        deg_valid[key] = value
    elif i in train_list:
        deg_train[key] = value
#compos_cal(deg_valid)    # Should be approximate 0
        
# ========== Compositional language ===================
comp_all = {}
comp_train = {}
comp_valid = {}

for i in range(NUM_SYSTEM**ATTRI_SIZE):
    key = num_to_tup(i)
    value = key_to_value(key, char_mapping, True)
    comp_all[key] = value
    if i in valid_list:
        comp_valid[key] = value
    elif i in train_list:
        comp_train[key] = value

#compos_cal(comp_valid)   # Should approximate 1.

# ========== Holistic language ===================
holi_all = {}
holi_train = {}
holi_valid = {}


key_all_list = list(comp_all.keys())
value_all_list = list(comp_all.values())
random.shuffle(value_all_list)

for i in range(NUM_SYSTEM**ATTRI_SIZE):
    key = key_all_list[i]
    value = value_all_list[i]
    holi_all[key] = value
    if i in valid_list:
        holi_valid[key] = value
    elif i in train_list:
        holi_train[key] = value

#compos_cal(holi_valid)    # Should be smaller than 1


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
