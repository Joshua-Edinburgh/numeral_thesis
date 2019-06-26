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

valid_full, valid_candidates, sel_idx_val = valid_data_gen()
batch_list = batch_data_gen()


def msg_tau_schedule(best_acc):
    if best_acc >= 0.85:
        args.tau = 0.05
    elif best_acc >= 0.8:
        args.tau = 0.1
    elif best_acc >= 0.7:
        args.tau = 0.5
    elif best_acc >= 0.6:
        args.tau = 1.
    else:
        args.tau = 2.

def cal_correct_preds(data_batch, data_candidate, pred_idx):
    '''
        Use to calculate the reward or the valid accuracy. As it is possible that
        there are multiple same elements in one row of data_candidate, we will
        check the predicting object rather than index to decide whether it is correct
    '''
    batch_size = data_batch.shape[0]
    cnt_correct = 0
    idx_correct = torch.zeros((batch_size,))
    for i in range(batch_size):
        if data_candidate[i][pred_idx[i]]==data_batch[i]:
            cnt_correct += 1
            idx_correct[i] = 1
    return cnt_correct, idx_correct


def train_epoch(speaker, listener, spk_optimizer, lis_optimizer, train_batch, train_candidates, sel_idx_train, update='BOTH', clip=CLIP):
    '''
        Train one epoch for one batch.
    '''
    speaker.train()
    listener.train()
    lis_loss_fun = nn.CrossEntropyLoss()
    
    spk_optimizer.zero_grad()
    lis_optimizer.zero_grad()
    
    true_idx = torch.tensor(sel_idx_train).to(DEVICE)

            # =========== Forward process =======
    msg, spk_log_prob, spk_entropy = speaker(train_batch)
    pred_vector = listener(train_candidates, msg)  
    lis_entropy = -(F.softmax(pred_vector)*F.log_softmax(pred_vector)).sum(dim=1)    
    lis_log_prob = F.log_softmax(pred_vector.max(dim=1)[0])
    
    pred_idx = F.softmax(pred_vector).argmax(dim=1)
    reward, reward_vector = cal_correct_preds(train_batch, train_candidates, pred_idx)
 
            # ========== Perform backpropatation ======
    #lis_loss = lis_loss_fun(pred_vector, true_idx.long().detach())
    lis_loss = -((reward_vector.detach()*lis_log_prob).mean() + 0.05*lis_entropy.mean())
    lis_loss.backward()
    
    if MSG_MODE == 'REINFORCE':
        spk_loss = -((reward_vector.detach()*spk_log_prob).mean() + 0.1*spk_entropy.mean())
        spk_loss.backward()
    elif MSG_MODE == 'SCST':
        speaker.eval()
        listener.eval()
        
        msg_, spk_log_prob_, _ = speaker(train_batch)
        pred_vector_ = listener(train_candidates, msg_)
        pred_idx_ = F.softmax(pred_vector_).argmax(dim=1)
        _, reward_vector_ = cal_correct_preds(train_batch, train_candidates, pred_idx_)
        
        speaker.train()
        listener.train()
        
        spk_loss = -(((reward_vector.detach()-reward_vector_.detach())*spk_log_prob).mean() + 0.1*spk_entropy.mean())
        spk_loss.backward()                    
    elif MSG_MODE == 'GUMBEL':
        spk_loss = lis_loss

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

    return reward, spk_loss.item(), lis_loss.mean().item()


def valid_cal(speaker, listener, valid_full, valid_candidates):
    '''
        Use valid data batch to see the accuracy for validation. 
    '''
    with torch.no_grad():  
        speaker.eval()
        listener.eval()
        msg, spk_log_prob, spk_entropy = speaker(valid_full)
        pred_vector = listener(valid_candidates, msg)
    
        pred_idx = F.softmax(pred_vector).argmax(dim=1)
        val_acc, _ = cal_correct_preds(valid_full, valid_candidates, pred_idx)    
    
        return val_acc/valid_full.shape[0]
    
# ============= Iterated method 2: alternatively initialize spk and lis =======
rewards = []
comp_ps = []
comp_ss = []
msg_types = []
valid_accs = []
for i in range(5000):
    print('==============Round %d ==============='%i)
    #j = np.mod(i,2)
    batch_list = batch_data_gen()#shuffle_batch(batch_list)
    j = 0
    train_batch, train_candidates, sel_idx_train = batch_list[j]['data'], batch_list[j]['candidates'], batch_list[j]['sel_idx']
    reward, spk_loss, lis_loss = train_epoch(speaker, listener, spk_optimizer, lis_optimizer, 
                             train_batch, train_candidates, sel_idx_train)    
    rewards.append(reward)
    print(reward, spk_loss, lis_loss)

    #if i%500 == 0: batch_list = batch_data_gen()
    if i%10 == 1:
        if i > 10:
            best_acc = np.asarray(rewards[:-10:-1]).mean()/25
            msg_tau_schedule(best_acc)
        all_msgs = msg_generator(speaker, train_list, vocab_table_full, padding=True)
        msg_types.append(len(set(all_msgs.values())))
        comp_p, comp_s = compos_cal(all_msgs)
        #valid_acc = valid_cal(speaker, listener, valid_full, valid_candidates)        
        #print('Valid acc is %4f'%valid_acc)        
        print('Train acc is %d'%reward)
        #valid_accs.append(valid_acc)
        comp_ps.append(comp_p)
        comp_ss.append(comp_s)
'''        
    if i%1000 == 0:
        listener = ListeningAgent().to(DEVICE)
        lis_optimizer = OPTIMISER(listener.parameters(), lr=LEARNING_RATE * DECODER_LEARING_RATIO)

    if i%1000 == 0:
        speaker = SpeakingAgent().to(DEVICE)
        spk_optimizer = OPTIMISER(speaker.parameters(), lr=LEARNING_RATE)
''' 



#all_msgs = msg_generator(speaker, train_list, vocab_table_full, padding=True)
#comp_p, comp_s = compos_cal(all_msgs)

