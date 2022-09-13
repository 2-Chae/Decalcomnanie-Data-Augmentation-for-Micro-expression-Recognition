import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import albumentations as A

import os
import pandas as pd
import numpy as np
from typing import List, Any, Tuple, Callable, Optional
from pathlib import Path


import random
seed = 338
random_seed = seed
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 




def split_dataset(df: pd.DataFrame , out_subject: Any, use_syn: bool = False, use_side: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Split dataset with specified out_subject for LOSO validation. 
    
    Args:
        df (pd.DataFrame): dataset dataframe.
        out_subject (Any): out subject for LOSO validation.
        use_syn (bool): if use synthetic dataset. 
        use_side (bool) : if use side frames. (this is for exp_ori_decal.py)
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: splitted trainset, testset
    """
    if not use_syn: # if not using syn frames, just use original frames
        df = df[df['syn'] == False]
    if not use_side: # if not using side framse, just use original frames 
        df = df[df['side'] == 'o']

    # for LOSO metric
    trainset = df[df['subject']!=out_subject]
    testset = df[df['subject']==out_subject]   

    if use_syn:
        testset = testset[testset['syn']==False]
    if use_side:
        testset = testset[testset['side']=='o']
        
    return trainset, testset

def get_meta_data(df: pd.DataFrame) -> Tuple[List, List]:
    """Extract metadata for dataset.

    Args:
        df (pd.DataFrame): dataframe of dataset.
        
    Returns:
        tuple: (path_list, apex_list, labels)
    """
    path_list = [Path(p) for p in df['filepath']]
    apex_list = list(df['apex'])
    labels = list(df['label'])
    return path_list, apex_list, labels


class FixSequenceInterpolation:
    def __init__(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length

    def __call__(self, x) -> None:
        _, s, h, w = x.shape
        if s == self.sequence_length:
            return x
        else:
            x = x.unsqueeze(0)
            x = F.interpolate(
                x, size=(self.sequence_length, h, w), mode='trilinear')
            x = x.squeeze(0)
            return x

    
class MEDataset(Dataset):
    """Micro Expression Dataset.

    Args:
        path_list (List): list of image paths.
        labels (List): list of labels.
        apex_list (List): list of apex files.
        n_frames (int): the number of frames to be used for training.
        transform (sequence, optional): list of transformations. Defaults to None.
    """
    def __init__(self, path_list: List, labels: List, apex_list: List, n_frames: int, transform: Optional[Callable] = None, only_ori: bool = False) -> None:
        super().__init__()
        self.path_list = path_list
        self.labels = labels
        self.apex_list = apex_list
        self.transform = transform
        self.n_frames = n_frames
        self.only_ori = only_ori
        self.interpolate = FixSequenceInterpolation(sequence_length=n_frames)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        If n_frames is 1, this will return only the apex frame. --> frame (C, H, W): torch.Tensor
        Otherwise, it will return #n_frames frames. --> video (C, T, H, W): torch.Tensor
        
        Args:
            idx (int): Index

        Returns:
            tuple: (frame, label) where label is index of the target class.
        """
        label = self.labels[idx]
        
        path = self.path_list[idx]
        frames = np.load(path).astype(np.float32)
        

        #######
        # For Apex only --> !! make sure there is no "nan" data in apex list !!
        #######
        if self.n_frames == 1:
            frame = frames[int(self.apex_list[idx])]
            if self.transform:
                frame = self.transform(frame)
            return frame, label
        
        ######
        # For Sequential 
        ######

        if not self.only_ori:
            f_name = path.name.split('.')[0]
            left_face = np.load(path.parents[1].joinpath(str(f_name)+'_left', f'{f_name}_left.npy'))
            right_face = np.load(path.parents[1].joinpath(str(f_name)+'_right', f'{f_name}_right.npy'))
          
            if self.transform:
                video_original = self.transform(frames)
                video_left = self.transform(left_face)
                video_right = self.transform(right_face) 
                
            else:
                raise NotImplementedError('should have transform!')
            
            if video_original.size(1) != self.n_frames:
                video_original = self.interpolate(video_original)
                video_left = self.interpolate(video_left) 
                video_right = self.interpolate(video_right) 
            
            
            return video_original, video_left, video_right, label
        else:     
            if self.transform:
                video = self.transform(frames)
            else:
                raise NotImplementedError('should have transform!')  

            if video.size(1) != self.n_frames:
                video = self.interpolate(video)

            return video, label

        
    def __len__(self) -> int:
        return len(self.path_list)
   
    def get_labels(self):
        return self.labels 