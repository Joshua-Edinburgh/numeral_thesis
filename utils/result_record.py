#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:51:54 2019

@author: xiayezi
"""

from utils.conf import *
import pandas as pd

vocab_table_full = [chr(97+int(v)) for v in range(26)]
#vocab_table_full[-1] = '@'

# concept1, concept2 = ('blue','box'), ('red','circle')
# concept1, concept2 = ('blue','circle'), ('red','circle')
def hanmming_dist(concept1, concept2):
    '''
        Calculate the hanmming distance of two concepts.
        The input concepts should be tuple, e.g. ('red','box')
        We require concept1 and 2 have the same number of attributes,
        i.e., len(concept1)==len(concept2)
    '''
    acc_dist = 0
    for i in range(len(concept1)):
        if concept1[i]!=concept2[i]:
            acc_dist += 1
    
    return acc_dist
    

# str1, str2 = 'horse', 'rose'
def edit_dist(str1, str2):
    '''
        Calculate the edit distance of two strings.
        Insert/delete/replace all cause 1.
    '''
    len1, len2 = len(str1), len(str2)
    DM = [0]
    for i in range(len1):
        DM.append(i+1)
        
    for j in range(len2):
        DM_new=[j+1]
        for i in range(len1):
            tmp = 0 if str1[i]==str2[j] else 1
            new = min(DM[i+1]+1, DM_new[i]+1, DM[i]+tmp)
            DM_new.append(new)
        DM = DM_new
        
    return DM[-1]


def num_to_tup(num, length=ATTRI_SIZE, num_sys = NUM_SYSTEM):
    '''
        Manually change 0 to '000' to ('0', '0', '0'). 
    '''
    tmp_list = []
    
    for i in range(ATTRI_SIZE):
        index = int(np.mod(num / (NUM_SYSTEM**i), NUM_SYSTEM))
        tmp_list.append(str(index))      
    tmp_list.reverse()
    return tuple([s for s in tmp_list])


def one_msg_translator(one_msg, vocab_table_full, padding=True):
    '''
        Translate the message [MAX_LEN, VOCAB+1] to [MAX_LEN] sentence
    '''
    max_len, vocab_len = one_msg.shape
    vocab_table = vocab_table_full[:vocab_len]
    #vocab_table[-1] = vocab_table_full[-1]
    
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


def msg_generator(speaker, object_list, vocab_table_full, padding=True):
    '''
        Use this function to generate messages for all items in object_list. 
        Padding is to control whether msg have the same length.
    '''
    with torch.no_grad():
        speaker.eval()
        all_msg = {}
        
        all_batch = np.asarray(object_list)
        msgs, _, _ = speaker(all_batch)
    
        msgs = msgs.transpose(0,1)
        for i in range(msgs.shape[0]):
            key = num_to_tup(object_list[i])
            value = one_msg_translator(msgs[i], vocab_table_full, True)
            all_msg[key] = value
        
        return all_msg

def compos_cal_inner(msg, train_batch):
    all_msg = {}
    msg = msg.transpose(0,1)
    for i in range(msg.shape[0]):
        key = num_to_tup(train_batch[i])
        value = one_msg_translator(msg[i], vocab_table_full, True)
        all_msg[key] = value
    comp_p, comp_s = compos_cal(all_msg)
    return comp_p, comp_s
    

def compos_cal(msg):
    '''
        Calculate the compositionalities using metric mentioned in:
        Language as an evolutionary system -- Appendix A (Kirby 2005)
        Input:
            msg: dictionary for a all possible {(color, shape):msg}
        Output:
            corr_pearson:   person correlation
            corr_spearma:  spearman correlation
    '''
    keys_list = list(msg.keys())
    concept_pairs = []
    message_pairs = []
    # ===== Form concepts and message pairs ========
    for i in range(len(keys_list)):
        #for j in range(i+1, len(keys_list)):
        for j in range(len(keys_list)):
            tmp1 = (keys_list[i],keys_list[j])
            concept_pairs.append((keys_list[i],keys_list[j]))
            tmp2 = (msg[tmp1[0]],msg[tmp1[1]])
            message_pairs.append(tmp2)
            
    # ===== Calculate distant for these pairs ======
    concept_HD = []
    message_ED = []
    for i in range(len(concept_pairs)):
        concept1, concept2 = concept_pairs[i]
        message1, message2 = message_pairs[i]
        concept_HD.append(hanmming_dist(concept1, concept2))
        message_ED.append(edit_dist(message1, message2))
    
    if np.sum(message_ED)==0:
        message_ED = np.asarray(message_ED)+0.1
        message_ED[-1] -= 0.01
 
    dist_table = pd.DataFrame({'HD':np.asarray(concept_HD),
                               'ED':np.asarray(message_ED)})    
    corr_pearson = dist_table.corr()['ED']['HD']
    corr_spearma = dist_table.corr('spearman')['ED']['HD']
     
    return corr_pearson, corr_spearma

'''
msg = {}   
# === Degenerate =====   
msg['000'] = ['aaa']       
msg['001'] = ['aaa'] 
msg['010'] = ['aaa']       
msg['100'] = ['aaa'] 
compos_cal(msg)
# === Holistic =====   
msg['000'] = ['a']       
msg['001'] = ['b'] 
msg['010'] = ['c']       
msg['100'] = ['d']
compos_cal(msg)
# === Compositional == 
msg['000'] = ['aaa']       
msg['001'] = ['aab'] 
msg['010'] = ['aba']       
msg['100'] = ['baa'] 
compos_cal(msg) 
'''



