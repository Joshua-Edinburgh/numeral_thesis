#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:33:04 2019

@author: s1583620
"""

from utils.conf import *
from utils.data_gen import *
from utils.result_record import *
from models.model import *
from torch.nn import NLLLoss
import matplotlib.pyplot as plt
from utils.manual_language_gen import *



def train_phaseA_curve(speaker, spk_optimizer, data_for_spk, clip=CLIP):
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


def smooth_rwd(rwd,ratio=10):
    new_rwd = np.zeros(rwd.shape)
    tmp = rwd[0]
    for i in range(rwd.size):
        tmp = (1-1/ratio)*tmp + 1/ratio*rwd[i]
        new_rwd[i] = tmp
    return new_rwd
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


def get_learning_comp_curve(agent, optimizer, data_set):
    acc_avg20 = 0
    acc_list = []
    comp_list = []
    for i in range(4000):
        acc = train_phaseA_curve(agent, optimizer, random.choice(data_set))
        acc_avg20 = (1-0.01)*acc_avg20 + 0.01*acc
        acc_list.append(acc_avg20)
        if i%5 == 0:
            print(i)
            for j in range(1):
                all_msgs = msg_generator(agent, train_list, vocab_table_full, padding=True)
                comp_p, comp_s = compos_cal(all_msgs)
                comp_list.append(comp_p)
    
    return acc_list, comp_list

print('COMP')
acc_comp,comp_comp = get_learning_comp_curve(speaker_comp, spk_optimizer_comp, shuf_pairs_comp)
print('HOLI')
acc_holi,comp_holi = get_learning_comp_curve(speaker_holi, spk_optimizer_holi, shuf_pairs_holi)
print('HOLI2')
acc_holi2,comp_holi2 = get_learning_comp_curve(speaker_holi2, spk_optimizer_holi2, shuf_pairs_holi2)
print('HOLI3')
acc_holi3,comp_holi3 = get_learning_comp_curve(speaker_holi3, spk_optimizer_holi3, shuf_pairs_holi3)


comp_comp_avg = np.asarray(comp_comp).reshape(800,1).mean(1)
comp_holi_avg = np.asarray(comp_holi).reshape(800,1).mean(1)
comp_holi_avg2 = np.asarray(comp_holi2).reshape(800,1).mean(1)
comp_holi_avg3 = np.asarray(comp_holi3).reshape(800,1).mean(1)

comp_comp_std = np.asarray(comp_comp).reshape(800,1).std(1)
comp_holi_std = np.asarray(comp_holi).reshape(800,1).std(1)
comp_holi_std2 = np.asarray(comp_holi2).reshape(800,1).std(1)
comp_holi_std3 = np.asarray(comp_holi3).reshape(800,1).std(1)

fig_rwd2 = plt.figure(figsize=(6,4))
ax = fig_rwd2.add_subplot(1,1,1)

x = np.arange(0,800)*50
#ax.plot(x,comp_comp_avg,label='Rho = 1.0',color='blue')
ax.plot(x,smooth_rwd(comp_holi_avg3,4),label=r'$\rho$'+'=0.46',color='blue')
ax.plot(x,smooth_rwd(comp_holi_avg2,4),label=r'$\rho$'+'=0.35',color='green')
ax.plot(x,smooth_rwd(comp_holi_avg,4),label=r'$\rho$'+'=0.28',color='red')

#ax.fill_between(x, comp_comp_avg - 2*comp_comp_std, comp_comp_avg+2*comp_comp_std, color='blue', alpha=0.2)
ax.fill_between(x, comp_holi_avg3 - 2*comp_holi_std3, comp_holi_avg3+2*comp_holi_std3, color='blue', alpha=0.2)
ax.fill_between(x, comp_holi_avg2 - 2*comp_holi_std2, comp_holi_avg2+2*comp_holi_std2, color='green', alpha=0.2)
ax.fill_between(x, comp_holi_avg - 2*comp_holi_std, comp_holi_avg+2*comp_holi_std, color='red', alpha=0.2)
ax.legend()
#ax.show()

plt.xlabel('Number of rounds',fontsize=16)
plt.ylabel('Topological similarity:'+r'$\rho$',fontsize=16)
ax.legend(fontsize=14)
ax.grid('on')
fig_rwd2.tight_layout()
fig_rwd2.savefig('Figures/Learning_comp_spk.pdf')

'''




for i in range(3000):
    acc_ho = train_phaseA_curve(speaker_holi, spk_optimizer_holi, random.choice(shuf_pairs_holi))
    acc_ho2 = train_phaseA_curve(speaker_holi2, spk_optimizer_holi2, random.choice(shuf_pairs_holi2))
    acc_ho3 = train_phaseA_curve(speaker_holi3, spk_optimizer_holi3, random.choice(shuf_pairs_holi3))
    acc_co = train_phaseA_curve(speaker_comp, spk_optimizer_comp, random.choice(shuf_pairs_comp))
    
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
        

'''