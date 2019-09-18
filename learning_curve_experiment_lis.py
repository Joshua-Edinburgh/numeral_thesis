#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:02:53 2019

@author: s1583620
"""

from utils.conf import *
from utils.data_gen import *
from utils.result_record import *
from models.model import *
from torch.nn import NLLLoss
import matplotlib.pyplot as plt
from utils.manual_language_gen import *


def cal_correct_preds(data_batch, data_candidate, pred_idx):
    '''
        Use to calculate the reward or the valid accuracy. As it is possible that
        there are multiple same elements in one row of data_candidate, we will
        check the predicting object rather than index to decide whether it is correct
    '''
    batch_size = data_batch.shape[0]
    cnt_correct = 0
    idx_correct = torch.zeros((batch_size,)).to(DEVICE)
    for i in range(batch_size):
        if data_candidate[i][pred_idx[i]]==data_batch[i]:
            cnt_correct += 1
            idx_correct[i] = 1
    return cnt_correct, idx_correct

# ====== For the listener =========
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


listener_comp = ListeningAgent().to(DEVICE)
lis_optimizer_comp = OPTIMISER(listener_comp.parameters(), lr=LEARNING_RATE * DECODER_LEARING_RATIO)
acc_comp = []

for i in range(1000):
    comp_lis_train = get_lis_curve_msg(batch_data_gen(), comp_spk_train)
    acc_comp.append(manual_lang_lis_train(listener_comp, lis_optimizer_comp, comp_lis_train))    
    if i%50 == 1:
        print('Round %d acc is %.4f'%(i,acc_comp[-1]))



listener_holi = ListeningAgent().to(DEVICE)
lis_optimizer_holi = OPTIMISER(listener_holi.parameters(), lr=LEARNING_RATE * DECODER_LEARING_RATIO)
acc_holi = []

for i in range(1000):
    holi_lis_train = get_lis_curve_msg(batch_data_gen(), holi_spk_train)
    acc_holi.append(manual_lang_lis_train(listener_holi, lis_optimizer_holi, holi_lis_train))    
    if i%50 == 1:
        print('Round %d acc is %.4f'%(i,acc_holi[-1]))


listener_holi2 = ListeningAgent().to(DEVICE)
lis_optimizer_holi2 = OPTIMISER(listener_holi2.parameters(), lr=LEARNING_RATE * DECODER_LEARING_RATIO)
acc_holi2 = []

for i in range(1000):
    holi_lis_train2 = get_lis_curve_msg(batch_data_gen(), holi_spk_train2)
    acc_holi2.append(manual_lang_lis_train(listener_holi2, lis_optimizer_holi2, holi_lis_train2))    
    if i%50 == 1:
        print('Round %d acc is %.4f'%(i,acc_holi2[-1]))

listener_holi3 = ListeningAgent().to(DEVICE)
lis_optimizer_holi3 = OPTIMISER(listener_holi3.parameters(), lr=LEARNING_RATE * DECODER_LEARING_RATIO)
acc_holi3 = []

for i in range(1000):
    holi_lis_train3 = get_lis_curve_msg(batch_data_gen(), holi_spk_train3)
    acc_holi3.append(manual_lang_lis_train(listener_holi3, lis_optimizer_holi3, holi_lis_train3))    
    if i%50 == 1:
        print('Round %d acc is %.4f'%(i,acc_holi3[-1]))


def smooth_rwd(rwd,ratio=10):
    new_rwd = np.zeros(rwd.shape)
    tmp = rwd[0]
    for i in range(rwd.size):
        tmp = (1-1/ratio)*tmp + 1/ratio*rwd[i]
        new_rwd[i] = tmp
    return new_rwd




fig_rwd2 = plt.figure(figsize=(6,4))
ax = fig_rwd2.add_subplot(1,1,1)
x = np.arange(0,1000,1)
ax.plot(x,smooth_rwd(np.asarray(acc_comp),3),label=r'$\rho$'+'=1.0')
ax.plot(x,smooth_rwd(np.asarray(acc_holi3),3),label=r'$\rho$'+'=0.85')
ax.plot(x,smooth_rwd(np.asarray(acc_holi2),3),label=r'$\rho$'+'=0.62')
ax.plot(x,smooth_rwd(np.asarray(acc_holi),3),label=r'$\rho$'+'=0.21')
plt.legend()

plt.xlabel('Number of pre-train rounds',fontsize=20)
plt.ylabel('Learning accuracy',fontsize=20)
ax.legend(fontsize=20)
ax.grid('on')
fig_rwd2.tight_layout()
fig_rwd2.savefig('Figures/Learning_spd_lis.pdf')


