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


vocab_table_full = [chr(97+int(v)) for v in range(27)]
vocab_table_full[-1] = '@'

def msg_translator(one_msg, vocab_table_full, padding=True):
    '''
        Translate the message [MAX_LEN, VOCAB+1] to [MAX_LEN] sentence
    '''
    max_len, vocab_len = one_msg.shape
    vocab_table = vocab_table_full[:vocab_len]
    vocab_table[-1] = vocab_table_full[-1]
    
    stop_flag = False
    sentence = []
    for i in range(max_len):
        voc_idx = one_msg[i].argmax()
        tmp_word = vocab_table[voc_idx]
        
        if tmp_word == vocab_table[-1]:
            stop_flag = True
        if padding == False and stop_flag:            
            break
        if padding == True and stop_flag:
            tmp_word = vocab_table[-1]
        sentence.append(tmp_word)
    
    return ''.join(sentence)
    
    



def train_epoch(speaker, listener, spk_optimizer, lis_optimizer, data_batch, data_candidates, sel_idx, clip=CLIP):
    # =========== Forward and backward propagation =================
    spk_loss_fun = NLLLoss()        # Use NLL for speaker
    
    spk_optimizer.zero_grad()
    lis_optimizer.zero_grad()
    
    true_idx = torch.tensor(sel_idx)

            # =========== Forward process =======
    msg, mask, spk_log_prob = speaker(data_batch)
    lg_lis_pred_prob, lis_pred_prob = listener(data_candidates, msg, mask)
    
    msg_translator(msg.transpose(0,1)[0],vocab_table_full,padding=True)
    
    
    reward = (true_idx==lis_pred_prob.argmax(dim=1)).sum()  # Reward is the number of correct guesses in a batch
    
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







for i in range(500):
    reward = train_epoch(speaker, listener, spk_optimizer, lis_optimizer, data_batch, data_candidates, sel_idx, clip=CLIP)
    print(reward)






