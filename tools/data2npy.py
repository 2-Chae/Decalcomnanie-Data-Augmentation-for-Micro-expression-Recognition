import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['SAMM', 'SAMM_CROP', 'SMIC_all_cropped', 'SMIC_all_raw', 'CASME2', 'CASME2_cropped', 'Oulu', 'CK'], required=True)
parser.add_argument('--dir', type=str, help='path to dataset dir', required=True)
args = parser.parse_args()

path = Path(args.dir)
exts = {'SAMM': '*.jpg', 'SAMM_CROP': '*.jpg', 'SMIC_all_cropped':'*.bmp', 'SMIC_all_raw':'*.bmp', 'CASME2': '*.jpg', 'CASME2_cropped': '*.jpg', 'Oulu': '*.jpeg', 'CK': '*.png'}

imgs = list(path.rglob(f'**/{exts[args.dataset]}'))
dirs = set([d.parent for d in imgs])

for i, p in enumerate(dirs):
    frames = sorted(list(map(str, p.rglob(exts[args.dataset]))))
    if len(frames):
        ls = []
        min_h, min_w = (10000,10000)
        for f in frames:
            img = Image.open(f).convert('RGB')
            min_h = min(min_h, img.size[0])
            min_w = min(min_w, img.size[1])
            ls.append(img)
        ls = [np.array(img.resize((min_h, min_w), Image.LANCZOS)) for img in ls]
        seq = np.stack(ls)
        np.save(os.path.join(p, f'{p.name}.npy'), seq)
        print(i, p)
        for f in frames:
            os.remove(f)
     