import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from albumentations.pytorch.transforms import ToTensorV2
import torch
import albumentations as A
import torchvision.transforms as transforms

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from einops import rearrange
import shutil

from utils import cfg
from dataset import FixSequenceInterpolation

#### NOTE!! ####
# You must run data2csv.py again after oversampling.
# This code will delete all the existed syn_images and create new syn_images.
#
# Example
# python3 oversampling --dataset SMIC_CROP
################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['SAMM', 'SAMM_CROP', 'SMIC_CROP', 'SMIC', 'CASME2', 'CASME2_CROP'], help='Dataset to convert', required=True)
args = parser.parse_args()

csv_file = cfg.CSV[args.dataset]
df = pd.read_csv(csv_file)


def del_syn_images(df: pd.DataFrame):
    syn_df_index = df[df['syn'] == True].index
    for i in syn_df_index:
        filepath = Path(df.iloc[i]['filepath'])
        shutil.rmtree(filepath.parent)
    df = df.drop(syn_df_index)
    return df

df = df[df['side'] == 'o']
df = del_syn_images(df)
df.to_csv(csv_file, sep=',', index=False) 

counts = df['label'].value_counts()
if len(set(counts)) == 1:
    raise NotImplementedError("Already have balanced dataset!")

if 'SAMM' in args.dataset:
    n_frames = cfg.FRAMES.SAMM
    int2label = list(cfg.ORIGINAL_CATEGORIES.SAMM.keys())
elif 'SMIC' in args.dataset:
    n_frames = cfg.FRAMES.SMIC
    int2label = list(cfg.ORIGINAL_CATEGORIES.SMIC.keys())
elif 'CASME2' in args.dataset:
    n_frames = cfg.FRAMES.CASME2
    int2label = list(cfg.ORIGINAL_CATEGORIES.CASME2.keys())
else:
    raise NotImplementedError()

log_df = pd.DataFrame(columns=['image1', 'image2', 'alpha', 'label', 'dest'])
interpolate = FixSequenceInterpolation(sequence_length=n_frames)

max_n_class = counts.max()
for k in counts.keys():
    k_df = df[df['label'] == k] 
    if len(k_df) == max_n_class: # if a class has the same number of samples, then pass.
        continue
    
    for idx in range(max_n_class - len(k_df)):
        temp = k_df.sample(n=2)
        
        a_path = Path(temp.iloc[0]['filepath'])
        b_path = Path(temp.iloc[1]['filepath'])
        a = np.load(a_path)
        b = np.load(b_path)
        
        transform = A.ReplayCompose([
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE, always_apply=True),
            ToTensorV2()
        ])

        new_a = []
        for i in range(len(a)):
            a_frame = transform(image=a[i])
            new_a.append(a_frame['image']/255.)
        new_a = torch.stack(new_a, dim=0)    
        
        new_b = []
        for i in range(len(b)):
            b_frame = transform(image=b[i])
            new_b.append(b_frame['image']/255.)
        new_b = torch.stack(new_b, dim=0)
        
        new_a = rearrange(new_a, 's c h w -> c s h w')
        new_b = rearrange(new_b, 's c h w -> c s h w') 
        
        
        # interpolate
        new_a = interpolate(new_a)
        new_b = interpolate(new_b)
        
        alpha = np.random.rand() 
        
        new_video = None
        for i in range(n_frames):
            new_img = (1. - alpha) * new_a[:,i] +  alpha * new_b[:,i]
            new_img = np.array(transforms.ToPILImage()(new_img))
            if isinstance(new_video, np.ndarray):
                new_video = np.concatenate((new_video, new_img[np.newaxis,:]), axis=0)
            else:
                new_video = new_img[np.newaxis, :]
        
        
        name_ = 'syn_' + str(idx) + '_' + int2label[k]
        dir_ = os.path.join(a_path.parents[1], name_)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        np.save(os.path.join(dir_, f'{name_}.npy'), new_video)
        print(dir_) 
        # for logging
        log_df = log_df.append(pd.DataFrame([(a_path, b_path, alpha, k, dir_)], columns=['image1', 'image2', 'alpha', 'label', 'dest']), sort=False)

log_df.to_csv('oversampling_log.csv', sep=',', index=False) 
