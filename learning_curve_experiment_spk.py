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



def train_phaseA(speaker, spk_optimizer, data_for_spk, clip=CLIP):
    '''
        After re-initialization of the speaker, we should use the D[t-1] to pre-train
        it to make sure it have the knowledge from its predecesors.
        Input:
            data_for_spk is a dictionary, data_for_spk['data'] is the x,
            data_for_spk['msg'] is the y, both of which has size BATCH_SIZE.
            msg is on-hot vector.

            this value should be designed based on the learning curve of speaker.
    '''
    speaker.train()
    spk_optimizer.zero_grad()
    spk_loss_fun = nn.CrossEntropyLoss()

    X = data_for_spk['data']
    Y = data_for_spk['msg']
    Y_hat = Y.transpose(0,1).argmax(dim=2)
    msg, _, _, Y_hiddens = speaker(X)
    spk_loss = spk_loss_fun(Y_hiddens.transpose(0,1).transpose(1,2), Y_hat)
    spk_loss.backward()
    nn.utils.clip_grad_norm_(speaker.parameters(), clip)
    spk_optimizer.step()

    acc_cnt = 0
    for i in range(X.shape[0]):
        Y_pred = msg.transpose(0,1).argmax(dim=2)
        if (Y_pred[i]==Y_hat[i]).sum()==ATTRI_SIZE:
            acc_cnt += 1

    return acc_cnt/X.shape[0]

# ================== See which type of language learns faster =================
# ====== For the speaker =========
speaker_comp = SpeakingAgent().to(DEVICE)
speaker_holi = SpeakingAgent().to(DEVICE)
speaker_holi2 = SpeakingAgent().to(DEVICE)
speaker_holi3 = SpeakingAgent().to(DEVICE)
spk_optimizer_comp = OPTIMISER(speaker_comp.parameters(), lr=LEARNING_RATE)
spk_optimizer_holi = OPTIMISER(speaker_holi.parameters(), lr=LEARNING_RATE)
spk_optimizer_holi2 = OPTIMISER(speaker_holi2.parameters(), lr=LEARNING_RATE)
spk_optimizer_holi3 = OPTIMISER(speaker_holi3.parameters(), lr=LEARNING_RATE)

acc_comp = []
acc_holi = []
acc_holi2 = []
acc_holi3 = []
acc_holi_avg20, acc_holi2_avg20, acc_holi3_avg20, acc_comp_avg20 = 0, 0,0,0
shuf_pairs_holi = pair_gen([holi_spk_train], phA_rnds = 100, sub_batch_size = 1)
shuf_pairs_holi2 = pair_gen([holi_spk_train2], phA_rnds = 100, sub_batch_size = 1)
shuf_pairs_holi3 = pair_gen([holi_spk_train3], phA_rnds = 100, sub_batch_size = 1)
shuf_pairs_comp = pair_gen([comp_spk_train], phA_rnds = 100, sub_batch_size = 1)

for i in range(3000):
    acc_ho = train_phaseA(speaker_holi, spk_optimizer_holi, random.choice(shuf_pairs_holi))
    acc_ho2 = train_phaseA(speaker_holi2, spk_optimizer_holi2, random.choice(shuf_pairs_holi2))
    acc_ho3 = train_phaseA(speaker_holi3, spk_optimizer_holi3, random.choice(shuf_pairs_holi3))
    acc_co = train_phaseA(speaker_comp, spk_optimizer_comp, random.choice(shuf_pairs_comp))
    
    acc_holi_avg20 = (1-0.01)*acc_holi_avg20 + 0.01*acc_ho
    acc_holi2_avg20 = (1-0.01)*acc_holi2_avg20 + 0.01*acc_ho2
    acc_holi3_avg20 = (1-0.01)*acc_holi3_avg20 + 0.01*acc_ho3
    acc_comp_avg20 = (1-0.01)*acc_comp_avg20 + 0.01*acc_co
    
    
    acc_holi.append(acc_holi_avg20)
    acc_holi2.append(acc_holi2_avg20)
    acc_holi3.append(acc_holi3_avg20)
    acc_comp.append(acc_comp_avg20) 
    if i%50 == 1:
        print('Round %d, holi acc is %.4f, comp acc is %.4f'%(i,acc_holi[-1],acc_comp[-1]))
        
fig_rwd2 = plt.figure(figsize=(6,4))
ax = fig_rwd2.add_subplot(1,1,1)

ax.plot(acc_comp,label=r'$\rho$'+'=1.0')
ax.plot(acc_holi3,label=r'$\rho$'+'=0.85')
ax.plot(acc_holi2,label=r'$\rho$'+'=0.62')
ax.plot(acc_holi,label=r'$\rho$'+'=0.21')
ax.legend()
#ax.show()

plt.xlabel('Number of pre-train rounds',fontsize=16)
plt.ylabel('Smoothed training accuracy',fontsize=16)
ax.legend(fontsize=14)
ax.grid('on')
fig_rwd2.tight_layout()
fig_rwd2.savefig('Figures/Learning_spd_spk.pdf')





