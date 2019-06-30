#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 21:57:39 2019

@author: xiayezi
"""
from utils.conf import *
from utils.data_gen import *
from utils.result_record import *
from models.model import *
from torch.nn import NLLLoss
import matplotlib.pyplot as plt
from utils.manual_language_gen import *

from train_multiphase import train_phaseA, cal_correct_preds



# ================== See which type of language learns faster =================
# ====== For the speaker =========
speaker_comp = SpeakingAgent().to(DEVICE)
speaker_holi = SpeakingAgent().to(DEVICE)
spk_optimizer_comp = OPTIMISER(speaker_comp.parameters(), lr=LEARNING_RATE)
spk_optimizer_holi = OPTIMISER(speaker_holi.parameters(), lr=LEARNING_RATE)

acc_comp = []
acc_holi = []
for i in range(2500):
    acc_holi.append(train_phaseA(speaker_holi, spk_optimizer_holi, holi_spk_train))
    acc_comp.append(train_phaseA(speaker_comp, spk_optimizer_comp, comp_spk_train))    
    if i%50 == 1:
        print('Round %d, holi acc is %.4f, comp acc is %.4f'%(i,acc_holi[-1],acc_comp[-1]))
        

plt.plot(acc_comp,label='Comp lang')
plt.plot(acc_holi,label='Holi lang')
plt.legend()
plt.show()

# ====== For the listener =========
'''
def manual_lang_lis_train(listener, lis_optimizer, lang_lis_train, clip=CLIP):
    """
        Use manually language to train the listener, output the accuracy.
    """
    listener.train()
    lis_loss_fun = nn.CrossEntropyLoss()
    lis_optimizer.zero_grad()
    
    train_batch, train_candidates, sel_idx = lang_lis_train['data'], lang_lis_train['candidates'], lang_lis_train['sel_idx']
    msg = lang_lis_train['msg']
    pred_vector = listener(train_candidates, msg)
    lis_entropy = -(F.softmax(pred_vector)*F.log_softmax(pred_vector)).sum(dim=1)
    lis_log_prob = F.log_softmax(pred_vector.max(dim=1)[0])
    pred_idx = F.softmax(pred_vector).argmax(dim=1)
    reward, reward_vector = cal_correct_preds(train_batch, train_candidates, pred_idx)
    
    lis_loss = -((reward_vector.detach()*lis_log_prob).mean() + 0.05*lis_entropy.mean())
    lis_loss.backward()
    nn.utils.clip_grad_norm_(listener.parameters(), clip)
    lis_optimizer.step()
    
    return reward/len(reward_vector)


listener_holi = ListeningAgent().to(DEVICE)
listener_comp = ListeningAgent().to(DEVICE)
lis_optimizer_holi = OPTIMISER(listener_holi.parameters(), lr=LEARNING_RATE * DECODER_LEARING_RATIO)
lis_optimizer_comp = OPTIMISER(listener_comp.parameters(), lr=LEARNING_RATE * DECODER_LEARING_RATIO)

holi_lis_train = get_lis_curve_msg(batch_data_gen(), holi_spk_train)
comp_lis_train = get_lis_curve_msg(batch_data_gen(), comp_spk_train)
deg_lis_train = get_lis_curve_msg(batch_data_gen(), deg_spk_train)

acc_holi = []
acc_comp = []

for i in range(1000):
    holi_lis_train = get_lis_curve_msg(batch_data_gen(), holi_spk_train)
    comp_lis_train = get_lis_curve_msg(batch_data_gen(), comp_spk_train)
    acc_holi.append(manual_lang_lis_train(listener_holi, lis_optimizer_holi, holi_lis_train))
    acc_comp.append(manual_lang_lis_train(listener_comp, lis_optimizer_comp, comp_lis_train))    
    if i%50 == 1:
        print('Round %d, holi acc is %.4f, comp acc is %.4f'%(i,acc_holi[-1],acc_comp[-1]))

plt.plot(acc_comp,label='Comp lang')
plt.plot(acc_holi,label='Holi lang')
plt.legend()
plt.show()

'''

