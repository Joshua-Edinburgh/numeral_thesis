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


def train_epoch(speaker, listener, spk_optimizer, lis_optimizer, train_batch, train_candidates, sel_idx_train, update='BOTH', clip=CLIP):
    '''
        Train one epoch for one batch.
    '''
    #speaker.train(True)
    #listener.train(True)
    # =========== Forward and backward propagation =================
    spk_loss_fun = NLLLoss()        # Use NLL for speaker
    
    spk_optimizer.zero_grad()
    lis_optimizer.zero_grad()
    
    true_idx = torch.tensor(sel_idx_train).to(DEVICE)

            # =========== Forward process =======
    msg, mask, spk_log_prob = speaker(train_batch)
    lg_lis_pred_prob, lis_pred_prob = listener(train_candidates, msg, mask)   
    
    pred_idx = lis_pred_prob.argmax(dim=1)
    reward = cal_correct_preds(train_batch, train_candidates, pred_idx)
    
            # ========== Perform backpropatation ======
    if MSG_MODE == 'REINFORCE':
        spk_loss = (reward * spk_log_prob).mean()
        spk_loss.backward()
    
    lis_loss = spk_loss_fun(lg_lis_pred_prob, true_idx.long())
    lis_loss.mean().backward()

            # Clip gradients: gradients are modified in place
    nn.utils.clip_grad_norm_(speaker.parameters(), clip)
    nn.utils.clip_grad_norm_(listener.parameters(), clip)

    if update == 'BOTH':
        spk_optimizer.step()
        lis_optimizer.step()        
    elif update == 'SPEAKER':
        spk_optimizer.step()
        lis_optimizer.zero_grad()
    elif update == 'LISTENER':
        spk_optimizer.zero_grad()
        lis_optimizer.step()
    else:
        print('Please input "BOTH", "SPEAKER" or "LISTENER" for the train_epoch function')

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
    




# ============= Iterated method 2: alternatively initialize spk and lis =======
rewards = []
comp_ps = []
comp_ss = []
valid_accs = []
for i in range(100):
    print('==============Round %d ==============='%i)
    for j in range(5):
        reward = train_epoch(speaker, listener, spk_optimizer, lis_optimizer, 
                         train_batch, train_candidates, sel_idx_train, update='SPEAKER')
        rewards.append(reward)    
    for j in range(5):
        reward = train_epoch(speaker, listener, spk_optimizer, lis_optimizer, 
                         train_batch, train_candidates, sel_idx_train, update='LISTENER')
        rewards.append(reward)
        
    if i%5 == 0:
        all_msgs = msg_generator(speaker, train_list, vocab_table_full, padding=True)
        comp_p, comp_s = compos_cal(all_msgs)
        valid_acc = valid_cal(speaker, listener, valid_full, valid_candidates)        
        print('Valid acc is %4f'%valid_acc)        
        print('Train acc is %d'%reward)
        valid_accs.append(valid_acc)
        comp_ps.append(comp_p)
        comp_ss.append(comp_s)

    if i%10 == 5:
        listener.reset_params()
    if i%10 == 0:
        speaker.reset_params()
        #listener = ListeningAgent().to(DEVICE)
        #lis_optimizer = OPTIMISER(listener.parameters(), lr=LEARNING_RATE * DECODER_LEARING_RATIO)
print(comp_ps)
print(comp_ss)


'''
# ============= Iterated method 1: regularly initialize listener ==============
rewards = []
comp_ps = []
comp_ss = []
valid_accs = []
for i in range(100):
    print('==============Round %d ==============='%i)
    reward = train_epoch(speaker, listener, spk_optimizer, lis_optimizer, 
                         train_batch, train_candidates, sel_idx_train, update='BOTH')
    rewards.append(reward)    
    if i%5 == 0:
        all_msgs = msg_generator(speaker, train_list, vocab_table_full, padding=True)
        comp_p, comp_s = compos_cal(all_msgs)
        valid_acc = valid_cal(speaker, listener, valid_full, valid_candidates)        
        print('Valid acc is %4f'%valid_acc)
        print('Train acc is %4f'%(rewards/BATCH_SIZE))
        valid_accs.append(valid_acc)
        comp_ps.append(comp_p)
        comp_ss.append(comp_s)

    if i%10 == 0:
        listener.reset_params()
        #listener = ListeningAgent().to(DEVICE)
        #lis_optimizer = OPTIMISER(listener.parameters(), lr=LEARNING_RATE * DECODER_LEARING_RATIO)
 
'''    

'''
# ============= General trianing-valid procedure ====================
rewards = []
comp_ps = []
comp_ss = []
for i in range(300):
    print('=====Round %d'%i)
    reward = train_epoch(speaker, listener, spk_optimizer, lis_optimizer, train_batch, train_candidates, sel_idx_train, clip=CLIP)
    rewards.append(reward)
    if i%5 ==0:
        all_msgs = msg_generator(speaker, train_list, vocab_table_full, padding=True)
        comp_p, comp_s = compos_cal(all_msgs)
        comp_ps.append(comp_p)
        comp_ss.append(comp_s)
''' 
  

#all_msgs = msg_generator(speaker, train_list, vocab_table_full, padding=True)
#comp_p, comp_s = compos_cal(all_msgs)


