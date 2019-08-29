#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:51:54 2019

@author: xiayezi
"""

from utils.conf import *
import pandas as pd
import os
from utils.data_gen import batch_data_gen

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


def label_to_tup(label, length=ATTRI_SIZE, num_sys = NUM_SYSTEM):
    '''
        Manually change '01' to ('0', '1'). 
    '''
    assert length == 2
    assert num_sys == 6
    return tuple([s for s in label])


def one_msg_translator(one_msg, vocab_table_full):
    '''
        Translate the message [MAX_LEN, VOCAB+1] to [MAX_LEN] sentence
        one_msg: torch.tensor;
        vocab_table_full: dict;
        return: 'ab'
    '''
    max_len, vocab_len = one_msg.shape
    vocab_table = vocab_table_full
    
    sentence = []
    for i in range(max_len):
        voc_idx = one_msg[i].argmax().item()
        tmp_word = vocab_table[voc_idx]
        sentence.append(tmp_word)
    
    return ''.join(sentence)


def msg_generator_sample(speaker, vocab_table_full):
    '''
        Use this function to generate messages for all items in object_list. 
        Padding is to control whether msg have the same length.
    '''
    all_msg = {}

    with torch.no_grad():
        speaker.train()
        batch_list = batch_data_gen()
        all_batch  = batch_list['data']
        label = batch_list['label']

        msgs, _, _, _ = speaker(all_batch)
        msgs = msgs.transpose(0,1)
        for i in range(msgs.shape[0]):
            key = label_to_tup(label[i])
            value = one_msg_translator(msgs[i], vocab_table_full)
            all_msg[key] = value

        return all_msg


def msg_generator(speaker, vocab_table_full):
    '''
        Use this function to generate messages for all items in object_list. 
        Padding is to control whether msg have the same length.
    '''
    all_msg = {}

    with torch.no_grad():
        speaker.eval()
        batch_list = batch_data_gen()
        all_batch  = batch_list['data']
        label = batch_list['label']

        msgs, _, _, _ = speaker(all_batch)
        msgs = msgs.transpose(0,1)
        for i in range(msgs.shape[0]):
            key = label_to_tup(label[i])
            value = one_msg_translator(msgs[i], vocab_table_full)
            all_msg[key] = value

        return all_msg

def compos_cal_inner(msg, label_list):
    all_msg = {}
    msg = msg.transpose(0,1)
    for i in range(msg.shape[0]):
        key = label_to_tup(label_list[i])
        value = one_msg_translator(msg[i], vocab_table_full)
        all_msg[key] = value
    comp_p, comp_s = compos_cal(all_msg)
    return comp_p, comp_s, all_msg
    

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


def msg_print_to_file(msg_all, path):
    '''
        Given the msg_all dictionary, write it down to the file
    '''
    if not os.path.exists(path):
        os.mkdir(path)
    save_path = path+'msg_all.txt'

    with open(save_path,'a') as f:
        for i in range(NUM_SYSTEM+1):
            line = ''
            for j in range(NUM_SYSTEM+1):
                if i==0:
                    line = line + str(j)+'\t'
                elif j==0:
                    line = line + str(i)+'\t'
                else:
                    key = (str(j-1),str(i-1))
                    tmp = msg_all[key]
                    line = line + tmp+'\t'
            f.write(line+'\n')
        
def smooth(matrix, ratio=20):
    '''
        Smooth the matrix according rows
    '''
    new_matrix = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        tmp = 0
        for j in range(matrix.shape[1]):
            tmp = (1-1/ratio)*tmp + 1/ratio*matrix[i,j]
            new_matrix[i,j] = tmp
    return new_matrix

'''
histo_list = []
for comp_list in comp_generations:
    tmp_histo = np.histogram(comp_list, bins=5)[0]
    histo_list.append(tmp_histo)

histo_matrix = np.asarray(histo_list).transpose()
smoth_matrix = smooth(histo_matrix,20)

for i in range (5):
    tmp = str(i*2/10)
    plt.plot(smoth_matrix[i,:],label='Rho= '+tmp)
plt.legend()
plt.show()

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



