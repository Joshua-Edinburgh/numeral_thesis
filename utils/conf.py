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

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4 # learning rate
CLIP = 50.0 # max after clipping gradients
DECODER_LEARING_RATIO = 5.0
OPTIMISER = optim.Adam

'''
for testing model
'''

defaults = {
        'MSG_TAU': 2      
        }

'''
hyperparameters of model
'''
SEL_CANDID = 5         # Number of candidates when playing th game
ATTRI_SIZE = 2          # Number of attributes, i.e., number of digits
NUM_SYSTEM = 5         # Number system, usually just use decimal
HIDDEN_SIZE = 100       
BATCH_SIZE = NUM_SYSTEM**ATTRI_SIZE     # Here one batch feed all the object types
MSG_MAX_LEN = ATTRI_SIZE + 0            # Maximum message length
MSG_VOCSIZE = NUM_SYSTEM+0              # 
VALID_RATIO = 0         # Ratio of valid set to train set

# Size of vocabulary this is available for communication

MSG_MODE = 'REINFORCE' # 'GUMBEL' or 'REINFORCE'
MSG_HARD = True # Discretized as one-hot vectors



parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tau', type=str, default=defaults['MSG_TAU'],
help='tau in GUMBEL softmax')

args = parser.parse_args()











