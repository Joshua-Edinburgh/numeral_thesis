#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:03:06 2019

@author: Shawn Guo
"""
import sys
sys.path.append("..")
from utils.conf import *


def load_img_set(dir_path):
    img_file_names = os.listdir(dir_path)
    imgs = [Image.open(os.path.join(dir_path, name)).convert('RGB') for name in img_file_names]
    return img_file_names, imgs


def build_img_tensors(imgs, device=DEVICE):
    tensors = []
    for img in imgs:
        tensors.append(torchvision.transforms.ToTensor()(img))
    tensors = torch.stack(tensors).to(device)
    return tensors


def generate_one_distractor(batch, device=DEVICE):
    batch_size = batch.shape[0]

    original_idx = torch.arange(batch_size, device=device)
    new_idx = torch.randperm(batch_size, device=device)

    while not (original_idx == new_idx).sum().eq(0):
        new_idx = torch.randperm(batch_size, device=device)

    shuffled_batch = batch[new_idx]

    return shuffled_batch


def batch_data_gen(device=DEVICE):
    ret = {}

    _, imgs = load_img_set('./data/img_set/')

    c = random.shuffle(imgs)

    # shape of batch: 36 * 3 * 100 * 50
    img_batch = build_img_tensors(imgs, device=device)

    # SEL_CANDID is the number of candidates
    sel_idx = np.random.randint(0, high=SEL_CANDID, size=(len(imgs)))

    candidates = []
    for i in range(SEL_CANDID):
        candidates.append(generate_one_distractor(img_batch, device=device))

    candidates = torch.stack(candidates).to(device)
    # TODO: verify this operation
    candidates.transpose_(0, 1)
    select_idx = torch.from_numpy(sel_idx).to(device).to(torch.long)
    for i in range(len(imgs)):
        candidates[i, sel_idx[i], :, :] = img_batch[i, :, :]

    ret['data'] = img_batch
    ret['sel_idx'] = sel_idx # is numpy array
    ret['candidates'] = candidates

    return ret

if __name__ == '__main__':
    batch_data_gen()
