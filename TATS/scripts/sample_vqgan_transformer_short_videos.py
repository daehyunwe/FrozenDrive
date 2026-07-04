# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import tqdm
import time
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
from einops import repeat

from tats import VideoData, Net2NetTransformer, load_transformer, load_vqgan
from tats.utils import save_video_grid
from tats.utils import shift_dim

parser = argparse.ArgumentParser()
parser = VideoData.add_data_specific_args(parser)
parser.add_argument('--gpt_ckpt', type=str, default='')
parser.add_argument('--vqgan_ckpt', type=str, default='')
parser.add_argument('--save', type=str, default='./results/tats')
parser.add_argument('--run', type=int, default=0)
parser.add_argument('--top_k', type=int, default=2048)
parser.add_argument('--top_p', type=float, default=0.92)
parser.add_argument('--n_sample', type=int, default=2048)
parser.add_argument('--dataset', type=str, default='ucf101', choices=['ucf101', 'sky', 'taichi'])
parser.add_argument('--class_cond', action='store_true')
parser.add_argument('--save_videos', action='store_true')
parser.add_argument('--compute_fvd', action='store_true')
args = parser.parse_args()

if args.top_k:
    save_dir = '%s/videos/%s/topp%.2f_topk%d_run%d'%(args.save, args.dataset, args.top_p, args.top_k, args.run)
    save_np = '%s/numpy_files/%s/topp%.2f_topk%d_run%d_eval.npy'%(args.save, args.dataset, args.top_p, args.top_k, args.run)
else:
    save_dir = '%s/videos/%s/toppNA_topkNA_run%d'%(args.save, args.dataset, args.run)
    save_np = '%s/numpy_files/%s/toppNA_topkNA_run%d_eval.npy'%(args.save, args.dataset, args.run)

all_data_np = np.load(save_np)

if args.compute_fvd:
    from tats.fvd.fvd import FVD_SAMPLE_SIZE, MAX_BATCH, get_fvd_logits, frechet_distance, \
        load_fvd_model, preprocess, TARGET_RESOLUTION, polynomial_mmd
    device = torch.device('cuda')
    i3d = load_fvd_model(device)
    data = VideoData(args)
    loader = data.train_dataloader()
    real_embeddings = []
    print('computing fvd embeddings for real videos')
    for batch in tqdm.tqdm(loader, desc="Computing real embeddings"):
        real_embeddings.append(get_fvd_logits(shift_dim((batch['video']+0.5)*255, 1, -1).byte().data.numpy(), i3d=i3d, device=device))
        if len(real_embeddings)*args.batch_size >= 2048: break
    print('caoncat fvd embeddings for real videos')
    real_embeddings = torch.cat(real_embeddings, 0)[:2048]
    print('computing fvd embeddings for fake videos')
    fake_embeddings = []
    n_batch = all_data_np.shape[0]//args.batch_size
    for i in tqdm.tqdm(range(n_batch), desc="Computing fake embeddings"):
        fake_embeddings.append(get_fvd_logits(all_data_np[i*args.batch_size:(i+1)*args.batch_size], i3d=i3d, device=device))
    print('caoncat fvd embeddings for fake videos')
    fake_embeddings = torch.cat(fake_embeddings, 0)[:2048]
    print('FVD = %.2f'%(frechet_distance(fake_embeddings, real_embeddings)))
    print('KVD = %.2f'%(polynomial_mmd(fake_embeddings.cpu(), real_embeddings.cpu())))
