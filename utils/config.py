from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.IMG_SIZE = 112

#
# CSV file path
# 
cfg.CSV = edict()
cfg.CSV.SAMM_CROP = 'dataset_csv/SAMM_CROP.csv'
cfg.CSV.SAMM = 'dataset_csv/SAMM.csv'
cfg.CSV.SMIC_CROP = 'dataset_csv/SMIC_all_cropped.csv'
cfg.CSV.SMIC = 'dataset_csv/SMIC_all_raw.csv'
cfg.CSV.CASME2 = 'dataset_csv/CASME2.csv'
cfg.CSV.CASME2_CROP = 'dataset_csv/CASME2_cropped.csv'
cfg.CSV.MERGED = 'dataset_csv/Merged.csv'
cfg.CSV.MERGED_3LABELS = 'dataset_csv/Merged_3labels.csv'

#
# Dataset Detail
#
cfg.FRAMES = edict()
cfg.FRAMES.SAMM = 74 
cfg.FRAMES.CASME2 = 66
cfg.FRAMES.SMIC = 34



# 
#  Labels for each dataset
# 

# original categories (SMIC: 3classes, CASME2: 5classes, SAMM: 5classes)
cfg.ORIGINAL_CATEGORIES = edict()
cfg.ORIGINAL_CATEGORIES.SMIC = {'negative': 0, 'positive': 1, 'surprise': 2}
cfg.ORIGINAL_CATEGORIES.CASME2 = {'others': 0, 'happiness': 1, 'surprise': 2, 'disgust': 3, 'repression': 4, 'fear': -1, 'sadness': -1}
cfg.ORIGINAL_CATEGORIES.SAMM = {'other': 0, 'happiness': 1, 'surprise': 2, 'anger': 3, 'contempt': 4, 'disgust': -1, 'fear': -1, 'sadness': -1}
cfg.ORIGINAL_CATEGORIES.CK = {'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}
cfg.ORIGINAL_CATEGORIES.OULU = {'happiness': 0, 'surprise': 1, 'sadness': 2, 'disgust': 3, 'anger': 4, 'fear': 5}


# negative=0, positive=1, surprise=2
cfg.THREE_CATEGORIES = edict()
cfg.THREE_CATEGORIES.CASME2 = {'others': -1, 'happiness': 1, 'surprise': 2, 'disgust': 0, 'repression': 0, 'fear': 0, 'sadness': 0}
cfg.THREE_CATEGORIES.SAMM = {'other': -1, 'happiness': 1, 'surprise': 2, 'anger': 0, 'contempt': 0, 'disgust': 0, 'fear': 0, 'sadness': 0}
cfg.THREE_CATEGORIES.CK = [-1, 0, 0, 0, 0, 1, 0, 2]
cfg.THREE_CATEGORIES.OULU = {'happiness': 1, 'surprise': 2, 'sadness': 0, 'disgust': 0, 'anger': 0, 'fear': 0}
cfg.THREE_CATEGORIES.MERGED = {'sadness': 0, 'anger': 0, 'disgust': 0, 'repression': 0, 'contempt': 0, 'fear': 0, 'negative': 0, 'happiness': 1, 'positive': 1, 'surprise': 2}


# #
# # Labels for each dataset
# #
# cfg.LABELS = edict()
# cfg.LABELS.SMIC_CROP = ['negative', 'positive', 'surprise']
# cfg.LABELS.SMIC = ['negative', 'positive', 'surprise']
# cfg.LABELS.SAMM_CROP = ['Anger', 'Sadness', 'Surprise', 'Fear', 'Happiness', 'Disgust', 'Contempt']
# cfg.LABELS.SAMM = ['Anger', 'Sadness', 'Surprise', 'Fear', 'Happiness', 'Disgust', 'Contempt']
# cfg.LABELS.MERGED = ['Anger', 'Sadness', 'Surprise', 'Fear', 'Happiness', 'Disgust', 'Contempt']



# cfg.LABELS.CASME2 = ['others', 'happiness', 'surprise', 'disgust', 'repression']

