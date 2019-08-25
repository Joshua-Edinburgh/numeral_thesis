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
import torchvision
from PIL import Image

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True



parser = argparse.ArgumentParser()
parser.add_argument('--tau', type=float, default=2., help='for gumble softmax')
parser.add_argument('--seed', type=int, default=12345, help='random seed')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_ratio',type=float, default=0.5, help='lr for listener is lr*lr_ratio')
parser.add_argument('--sel_candi', type=int, default=15, help='candiates for the listene')
parser.add_argument('--num_sys', type=int, default=10, help='number of attributes')
parser.add_argument('--phA', type=int, default=1000, help='the rounds for phase A')
parser.add_argument('--phB',type=int, default=4000, help='the rounds for phase B')
parser.add_argument('--phLP',type=int, default=200, help='the rounds for phase pre-train listener')
parser.add_argument('--path',type=str, default='test', help='the path to save the results')

args = parser.parse_args()



setup_seed(args.seed)   # 12345 is valid for N_B=100, SEL_CAN = 5



'''
for training model
'''
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = args.lr # learning rate
CLIP = 50.0 # max after clipping gradient
DECODER_LEARING_RATIO = args.lr_ratio
OPTIMISER = optim.Adam


'''
hyperparameters of model
'''
SEL_CANDID = args.sel_candi          # Number of candidate when selecting
ATTRI_SIZE = 2          # Number of attributes, i.e., number of digits
NUM_SYSTEM = args.num_sys         # Number system, usually just use decimal
HIDDEN_SIZE = 128
BATCH_SIZE = NUM_SYSTEM**ATTRI_SIZE
MSG_MAX_LEN = ATTRI_SIZE + 0      # Controlled by ourselves
VALID_RATIO = 0      # Ratio of valid set to train set

# Size of vocabulary this is available for communication
MSG_VOCSIZE = NUM_SYSTEM+0
MSG_MODE = 'REINFORCE' # 'GUMBEL' or 'REINFORCE'
MSG_HARD = True # Discretized as one-hot vectors












