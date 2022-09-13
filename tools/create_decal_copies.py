import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def save_decalcomanies(npy):
    frames = np.load(npy)
    filename, extension = npy.name.split('.')

    w = frames.shape[2]
    # left
    left_side = frames[:, :, :int(w/2), :]
    left_face = np.concatenate([left_side, np.flip(left_side, 2)], axis=2)
    path_to_save = npy.parents[1].joinpath(npy.parent.name+'_left')
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    np.save(os.path.join(path_to_save, filename + '_left.' + extension), left_face)

    # right
    right_side = frames[:, :, int(w/2):, :]
    right_face = np.concatenate([np.flip(right_side, 2), right_side], axis=2)
    path_to_save = npy.parents[1].joinpath(npy.parent.name+'_right')
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    np.save(os.path.join(path_to_save, filename + '_right.' + extension), right_face)


class DatasetLoader:
    def __init__(self, path: str, dataset: str):
        self.path = Path(path)
        self.dataset = dataset

    def get_npys(self):
        raise NotImplementedError()

class SMICLoader(DatasetLoader):
    def get_npys(self):
        path = self.path.joinpath('SMIC', self.dataset, 'HS')
        npys = list(path.rglob('**/*.npy'))
        return npys

class SAMMLoader(DatasetLoader):
    def get_npys(self):
        path = self.path.joinpath('samm', self.dataset)
        npys = list(path.rglob('**/*.npy'))
        return npys

class CASME2Loader(DatasetLoader):
    def get_npys(self):
        path = self.path.joinpath('CASME2')
        path_part = 'Cropped' if 'cropped' in self.dataset else 'CASME2_RAW_selected'
        path = path.joinpath(path_part)
        npys = list(path.rglob('**/*.npy'))
        return npys 

class CKLoader(DatasetLoader):
    def get_npys(self):
        path = self.path.joinpath('CK', 'CK+')
        path_part = 'ck_cropped' if self.dataset == 'CK+_cropped' else 'cohn-kanade-images'
        path = path.joinpath(path_part)
        npys = list(path.rglob('**/*.npy'))
        return npys 

class OULULoader(DatasetLoader):
    def get_npys(self): 
        path = self.path.joinpath('OULU')
        if 'OULU_cropped' == self.dataset:
            path = path.joinpath('PreProcess', 'VL_Acropped')
        else:
            path = path.joinpath('OriginalImg', 'VL')
        choices = ['Strong', 'Weak'] # exclude Dark set

        npys = []
        for choice in choices:
            upper_dir = path.joinpath(choice)
            npys += list(upper_dir.rglob(f'**/*.npy'))    
        return npys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=Path, default='/root/dataset', help='Path to the data directory')
    parser.add_argument('--dataset', choices=['SAMM', 'SAMM_CROP', 'SMIC_all_cropped', 'SMIC_all_raw', 'CK+', 'CK+_cropped', 'OULU', 'OULU_cropped', 'CASME2', 'CASME2_cropped', 'merge'], help='Dataset to convert', required=True)
    args = parser.parse_args()

    if 'SAMM' in args.dataset:
        loader = SAMMLoader(args.dir, args.dataset)
    elif 'SMIC' in args.dataset:
        loader = SMICLoader(args.dir, args.dataset)
    elif 'CASME2' in args.dataset:
        loader = CASME2Loader(args.dir, args.dataset)
    elif 'CK' in args.dataset:
       loader = CKLoader(args.dir, args.dataset) 
    elif 'OULU' in args.dataset:
       loader = OULULoader(args.dir, args.dataset) 
    
    npys = loader.get_npys()

    for npy in tqdm.tqdm(npys):
        save_decalcomanies(npy)
        

