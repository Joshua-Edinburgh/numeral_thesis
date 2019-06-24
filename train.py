#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:21:58 2019

@author: s1583620
"""
from utils.conf import *
from utils.data_gen import *
from utils.result_record import *
from models.model import *
from torch.nn import NLLLoss



speaker = SpeakingAgent().to(DEVICE)
listener = ListeningAgent().to(DEVICE)
spk_optimizer = OPTIMISER(speaker.parameters(), lr=LEARNING_RATE)
lis_optimizer = OPTIMISER(listener.parameters(), lr=LEARNING_RATE * DECODER_LEARING_RATIO)

    
def cal_correct_preds(data_batch, data_candidate, pred_idx):
    '''
        Use to calculate the reward or the valid accuracy. As it is possible that
        there are multiple same elements in one row of data_candidate, we will
        check the predicting object rather than index to decide whether it is correct
    '''
    batch_size = data_batch.shape[0]
    cnt_correct = 0
    for i in range(batch_size):
        if data_candidate[i][pred_idx[i]]==data_batch[i]:
            cnt_correct += 1
    return cnt_correct


def train_epoch(speaker, listener, spk_optimizer, lis_optimizer, train_batch, train_candidates, sel_idx_train, clip=CLIP):
    '''
        Train one epoch for one batch.
    '''
    #speaker.train(True)
    #listener.train(True)
    # =========== Forward and backward propagation =================
    spk_loss_fun = NLLLoss()        # Use NLL for speaker
    
    spk_optimizer.zero_grad()
    lis_optimizer.zero_grad()
    
    true_idx = torch.tensor(sel_idx_train)

            # =========== Forward process =======
    msg, mask, spk_log_prob = speaker(train_batch)
    lg_lis_pred_prob, lis_pred_prob = listener(train_candidates, msg, mask)   
    
    pred_idx = lis_pred_prob.argmax(dim=1)
    reward = cal_correct_preds(train_batch, train_candidates, pred_idx)
    #reward = (true_idx==lis_pred_prob.argmax(dim=1)).sum()  # Reward is the number of correct guesses in a batch
    
            # ========== Perform backpropatation ======
    if MSG_MODE == 'REINFORCE':
        spk_loss = (reward * spk_log_prob).mean()
        spk_loss.backward()
    
    lis_loss = spk_loss_fun(lg_lis_pred_prob, true_idx.long())
    lis_loss.mean().backward()

            # Clip gradients: gradients are modified in place
    nn.utils.clip_grad_norm_(speaker.parameters(), clip)
    nn.utils.clip_grad_norm_(listener.parameters(), clip)

            # Adjust model weights
    spk_optimizer.step()
    lis_optimizer.step()

    # =========== Result Statistics ==============

    return reward


def valid_cal(speaker, listener, valid_full, valid_candidates):
    '''
        Use valid data batch to see the accuracy for validation. 
    '''
    speaker.train(False)
    listener.train(False)

    msg, mask, spk_log_prob = speaker(valid_full)
    lg_lis_pred_prob, lis_pred_prob = listener(valid_candidates, msg, mask)
    
    pred_idx = lis_pred_prob.argmax(dim=1)
    val_acc = cal_correct_preds(valid_full, valid_candidates, pred_idx)    
    
    return val_acc/valid_full.shape[0]
    

rewards = []
comp_ps = []
comp_ss = []
for i in range(500):
    print('=====Round %d'%i)
    reward = train_epoch(speaker, listener, spk_optimizer, lis_optimizer, train_batch, train_candidates, sel_idx_train, clip=CLIP)
    rewards.append(reward)
    if i%5 ==0:
        all_msgs = msg_generator(speaker, train_list, vocab_table_full, padding=True)
        comp_p, comp_s = compos_cal(all_msgs)
        comp_ps.append(comp_p)
        comp_ss.append(comp_s)
    
    

#all_msgs = msg_generator(speaker, train_list, vocab_table_full, padding=True)
#comp_p, comp_s = compos_cal(all_msgs)


