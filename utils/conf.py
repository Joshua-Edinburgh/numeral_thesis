#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 21:29:42 2019

@author: xiayezi
"""

import argparse
import random
import numpy as np
import math
import os
import itertools

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch import autograd

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(34313)   # 12345 is valid for N_B=100, SEL_CAN = 5



'''
for training model
'''
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4 # learning rate
DROPOUT_RATIO = 0.2
CLIP = 50.0 # max after clipping gradients
TEACHER_FORCING_RATIO = 0.5
DECODER_LEARING_RATIO = 5.0
NUM_ITERS = 400
PRINT_EVERY = 1
SAVE_EVERY = 2
EVAL_EVERY = 2
OPTIMISER = optim.Adam
LOSS_FUNCTION = nn.CrossEntropyLoss(reduce=False)

'''
for testing model
'''
TEST_MODE = False


defaults = {
        'MSG_TAU': 2      
        }

'''
hyperparameters of model
'''
SEL_CANDID = 5          # Number of candidate when selecting
ATTRI_SIZE = 2          # Number of attributes, i.e., number of digits
NUM_SYSTEM = 5         # Number system, usually just use decimal
HIDDEN_SIZE = 100       
BATCH_SIZE = NUM_SYSTEM**ATTRI_SIZE
MSG_MAX_LEN = ATTRI_SIZE + 0      # Controlled by ourselves
VALID_RATIO = 0      # Ratio of valid set to train set

ROUNDS = 2000
# Size of vocabulary this is available for communication
MSG_VOCSIZE = NUM_SYSTEM+0
MSG_MODE = 'REINFORCE' # 'GUMBEL' or 'REINFORCE'
MSG_HARD = True # Discretized as one-hot vectors



parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tau', type=str, default=defaults['MSG_TAU'],
help='tau in GUMBEL softmax')



args = parser.parse_args()











