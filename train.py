#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:21:58 2019

@author: s1583620
"""
from utils.conf import *
from models.model import *


sel_idx = np.random.randint(0, SEL_CANDID,(BATCH_SIZE,))
data_candidates = np.random.randint(0, 10**ATTRI_SIZE, (BATCH_SIZE, SEL_CANDID))
data_batch = np.zeros((BATCH_SIZE,))
for i in range(BATCH_SIZE):
    data_batch[i] = data_candidates[i,sel_idx[i]]


speaker = SpeakingAgent()
listener = ListeningAgent()

msg, mask, spk_log_prob = speaker(data_batch)
lis_pred_prob = listener(data_candidates, msg, mask)

reward = (torch.tensor(sel_idx)==pred_prob.argmax(dim=1)).sum() / BATCH_SIZE


'''
def train_epoch(speaker, listener, spk_optimizer, lis_optimizer, data_batch, data_candidates, clip=CLIP):
    # Zero gradients
    spk_optimizer.zero_grad()
    lis_optimizer.zero_grad()

    # =========== Forward process =======
    msg, mask, spk_log_prob = speaker(data_batch)
    lis_pred_prob = listener(data_candidates, msg, mask)


    # ========== Calculate reward =============
    reward = (torch.tensor(sel_idx)==pred_prob.argmax(dim=1)).sum() / BATCH_SIZE
    
    # ========== Perform backpropatation ======
    lis_loss = lis_pred_prob
    lis_loss.mean().backward()
    
    if MSG_MODE == 'REINFORCE':
        log_msg_prob = (reward * spk_log_prob).mean()
        log_msg_prob.backward()
    

    # Clip gradients: gradients are modified in place
    nn.utils.clip_grad_norm_(model.parameters(), clip)

    # Adjust model weights
    spk_optimizer.step()
    lis_optimizer.step()

    return seq_acc, tok_acc, sum(print_losses) / len(print_losses)
'''









