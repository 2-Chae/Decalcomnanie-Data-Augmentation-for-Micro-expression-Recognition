from genericpath import exists
import os
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Tuple

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.config import cfg

class FMEDataset:
    def __init__(self, path: str, dataset: str, use_three_labels: bool = False, exist_xlsx: bool = False):
        self.df = pd.DataFrame(columns=['subject', 'label', 'filepath', 'apex', 'syn', 'side', 'type'])
        self.path = Path(path) if path else None
        self.dataset = dataset
        self.use_three_labels = use_three_labels
        self.exist_xlsx = exist_xlsx
        self.filename = dataset

    def _drop_cols(self, columns: List[str]):
        if not isinstance(columns, list):
            columns = [columns]
        self.df = self.df.drop(columns, axis=1)

    def _remove_with_keys(self, keys: List[Tuple]):
        if not isinstance(keys, list):
            keys = [keys] 
        for key, condition in keys:
            self.df = self.df[self.df[key] != condition]

    def create_csv(self, drop_cols: List[str] = None, remove_keys: List[Tuple] = None):
        """
        Args:
            drop_cols (List[str], optional): columns to remove. Defaults to None.
            remove_keys (List[Tuple], optional): keys to remove (key, condition). Defaults to None.
        """
        raise NotImplementedError()

    def _save_csv(self):
        self.df.drop_duplicates(inplace=True)
        if self.use_three_labels:
            self.filename += '_3labels'

        self.df.to_csv(f'dataset_csv/{self.filename}.csv',sep=',',index = False) 
        print(f'saved dataset_csv/{self.filename}.csv')


   
class SMIC(FMEDataset):
    def create_csv(self, drop_cols: List[str] = None, remove_keys: List[Tuple] = None):
        path = self.path.joinpath('SMIC', self.dataset, 'HS')
        npys = list(path.rglob('**/*.npy'))

        for npy_file in npys:
            if 'non_micro' in str(npy_file):
                continue

            subject = npy_file.parents[3].name
            label = cfg.ORIGINAL_CATEGORIES.SMIC[npy_file.parents[1].name]

            is_syn = True if 'syn' in npy_file.name else False
            
            side = 'o'
            if 'left' in npy_file.name:
                side = 'l'
            elif 'right' in npy_file.name:
                side = 'r'

            self.df = self.df.append(pd.DataFrame([(subject, label, npy_file, is_syn, side, 'micro')], columns=['subject', 'label', 'filepath', 'syn', 'side', 'type']), sort=False)

        if drop_cols:
            self._drop_cols(drop_cols)
        if remove_keys:
            self._remove_with_keys(remove_keys)

        self._save_csv()


class CASME2(FMEDataset):
    def create_csv(self, drop_cols: List[str] = None, remove_keys: List[Tuple] = None):
        path = self.path.joinpath('CASME2')
        xlsx_file = list(path.rglob('*.xlsx'))
        assert len(xlsx_file) == 1, "xlsx file does not exist" 

        original_df = pd.read_excel(xlsx_file[0])[
            ['Subject', 'Estimated Emotion', 'ApexFrame', 'Filename', 'OnsetFrame']
        ]
  
        path_part = 'Cropped' if 'cropped' in self.dataset else 'CASME2_RAW_selected'
        path = path.joinpath(path_part)

        npys = list(path.rglob('**/*.npy'))
        for npy_file in npys:
            subject = npy_file.parents[1].name[3:]
            parent_dir = npy_file.parent.name
            parent_dir = parent_dir.replace('_left', '')    
            parent_dir = parent_dir.replace('_right', '')     

            is_syn = False
            apex_frame = None

            if 'syn' in parent_dir:
                str_label = parent_dir.split('_')[2]
                is_syn = True # for synthetic data
                
            else:
                cur_item = original_df[(original_df['Filename']==parent_dir) & (original_df['Subject'] == int(subject))]
                print(npy_file)
                str_label = cur_item['Estimated Emotion'].item()

                # for apex
                if not isinstance(cur_item['ApexFrame'].item(), str):
                    apex_frame = (cur_item['ApexFrame'] - cur_item['OnsetFrame']).item()
                    apex_frame = None if apex_frame < 0 else apex_frame

            label = cfg.THREE_CATEGORIES.CASME2[str_label] if self.use_three_labels else cfg.ORIGINAL_CATEGORIES.CASME2[str_label] 

            # for decalcomanie
            side = 'o'
            if 'left' in npy_file.name:
                side = 'l'
            elif 'right' in npy_file.name:
                side = 'r'

            self.df = self.df.append(pd.DataFrame([(subject, label, npy_file, apex_frame, is_syn, side, 'micro')], columns=['subject', 'label', 'filepath', 'apex', 'syn', 'side', 'type']), sort=False)

        if drop_cols:
            self._drop_cols(drop_cols)

        # remove labels with value -1 
        remove_keys = [('label', -1)] if remove_keys == None else remove_keys + [('label', -1)] 
        if remove_keys:
            self._remove_with_keys(remove_keys)

        self._save_csv()   


class SAMM(FMEDataset):
    def create_csv(self, drop_cols: List[str] = None, remove_keys: List[Tuple] = None):
        path = self.path.joinpath('samm', args.dataset)
        xlsx_file = list(path.rglob('*.xlsx'))
        assert len(xlsx_file) == 1, "xlsx file does not exist" 

        original_df = pd.read_excel(xlsx_file[0], skiprows=range(13))[
            ['Subject', 'Estimated Emotion', 'Apex Frame', 'Filename', 'Onset Frame', 'Duration']
        ]
  
        npys = list(path.rglob('**/*.npy'))
        for npy_file in npys:
            subject = npy_file.parents[1].name
            parent_dir = npy_file.parent.name
            parent_dir = parent_dir.replace('_left', '')    
            parent_dir = parent_dir.replace('_right', '')     

            is_syn = False
            apex_frame = None

            if 'syn' in parent_dir:
                str_label = parent_dir.split('_')[2].lower()
                is_syn = True # for synthetic data
                
            else:
                cur_item = original_df[original_df['Filename']==parent_dir]
                
                str_label = cur_item['Estimated Emotion'].item().lower()
 
                # for apex
                apex_frame = (cur_item['Apex Frame'] - cur_item['Onset Frame']).item()
                apex_frame = None if apex_frame < 0 or apex_frame >= cur_item['Duration'].item() else apex_frame

            label = cfg.THREE_CATEGORIES.SAMM[str_label] if self.use_three_labels else cfg.ORIGINAL_CATEGORIES.SAMM[str_label] 

            # for decalcomanie
            side = 'o'
            if 'left' in npy_file.name:
                side = 'l'
            elif 'right' in npy_file.name:
                side = 'r'
        
            self.df = self.df.append(pd.DataFrame([(subject, label, npy_file, apex_frame, is_syn, side, 'micro')], columns=['subject', 'label', 'filepath', 'apex', 'syn', 'side', 'type']), sort=False)

        if drop_cols:
            self._drop_cols(drop_cols)

        # remove labels with value -1 
        remove_keys = [('label', -1)] if remove_keys == None else remove_keys + [('label', -1)] 
        if remove_keys:
            self._remove_with_keys(remove_keys)

        self._save_csv()   


class CK(FMEDataset):
    """
    Originally, CK+ has neutral label (0) but it does not have any clips for neutral label.
    """
    def create_csv(self, drop_cols: List[str] = None, remove_keys: List[Tuple] = None):
        path = self.path.joinpath('CK', 'CK+')
        label_path = path.joinpath('Emotion')

        path_part = 'ck_cropped' if self.dataset == 'CK+_cropped' else 'cohn-kanade-images'
        path = path.joinpath(path_part)

        npys = list(path.rglob('**/*.npy'))
        
        for npy_file in npys:
            apex_frame = -1 # last frame is apex frame

            cur_dir = npy_file.parent
            subject = cur_dir.parent.name

            label = -1
            label_path = str(cur_dir).replace(path_part, 'Emotion')
            label_file = list(Path(label_path).rglob('*.txt'))
            if not len(label_file) == 0: # if there is not emotion .txt file, label is None.
                with open(label_file[0]) as f:
                    label = int(float(f.read().strip()))

                label = cfg.THREE_CATEGORIES.CK[label] if self.use_three_labels else label - 1
            
            is_syn = True if 'syn' in subject else False
            
            # for decalcomanie
            side = 'o'
            if 'left' in subject:
                side = 'l'
            elif 'right' in subject:
                side = 'r'

            self.df = self.df.append(pd.DataFrame([(subject, label, npy_file, apex_frame, is_syn, side, 'macro')], columns=['subject', 'label', 'filepath', 'apex', 'syn', 'side', 'type']), sort=False)

        if drop_cols:
            self._drop_cols(drop_cols)

        # remove labels with value -1 
        remove_keys = [('label', -1)] if remove_keys == None else remove_keys + [('label', -1)] 
        if remove_keys:
            self._remove_with_keys(remove_keys)

        self._save_csv()   


class OULU(FMEDataset):
    def create_csv(self, drop_cols: List[str] = None, remove_keys: List[Tuple] = None):
        path = self.path.joinpath('OULU')
        if 'OULU_cropped' == self.dataset:
            path = path.joinpath('PreProcess', 'VL_Acropped')
        else:
            path = path.joinpath('OriginalImg', 'VL')

        choices = ['Strong', 'Weak', 'Dark'] # exclude Dark set

        for choice in choices:
            upper_dir = path.joinpath(choice)
            npys = list(upper_dir.rglob(f'**/*.npy'))    

            for npy_file in npys:
                apex_frame = -1 # last frame is apex frame
                cur_dir = npy_file.parent
                subject = cur_dir.parent.name
                str_label = cur_dir.name.lower()
                label = cfg.THREE_CATEGORIES.OULU[str_label] if self.use_three_labels else cfg.ORIGINAL_CATEGORIES.OULU[str_label] 

                is_syn = True if 'syn' in str(cur_dir) else False

                # for decalcomanie
                side = 'o'
                if 'left' in subject:
                    side = 'l'
                elif 'right' in subject:
                    side = 'r'

                self.df = self.df.append(pd.DataFrame([(subject, label, npy_file, apex_frame, is_syn, side, 'macro')], columns=['subject', 'label', 'filepath', 'apex', 'syn', 'side', 'type']), sort=False)

        if drop_cols:
            self._drop_cols(drop_cols)
        if remove_keys:
            self._remove_with_keys(remove_keys)

        self._save_csv()   

class MergedDataset(FMEDataset):
    """
    MergedDataset only supports datasets that already have .csv files.
    
    If you want to merge files using 3 labels (ne, pos, sur), you should create each {dataset}_3label.csv files first 
    and then use this MergedDataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = 'Merged'
        self._check_csv_existence()
        
    def _check_csv_existence(self):
        for d in self.dataset:
            assert os.path.exists(os.path.join(f'dataset_csv/{d}.csv')), f'{d}.csv does not exist!'
            
            if '3labels' in d:
                raise NotImplementedError('*_3labels csv file is not supported! Just use original csv.')
    
    def _get_int2str_label(self, dataset: str):
        if 'CASME2' in dataset:
            str2int = cfg.ORIGINAL_CATEGORIES.CASME2
        elif 'SAMM' in dataset:
            str2int = cfg.ORIGINAL_CATEGORIES.SAMM
        elif 'CK' in dataset:
            str2int = cfg.ORIGINAL_CATEGORIES.CK
        elif 'OULU' in dataset:
            str2int = cfg.ORIGINAL_CATEGORIES.OULU
       
        int2str = {}
        for k, v in str2int.items():
            int2str[v] = k
        
        return int2str
                        
    def merge(self, drop_cols: List[str] = None, remove_keys: List[Tuple] = None):
        for d in self.dataset:
            df = pd.read_csv(os.path.join(f'dataset_csv/{d}.csv'))
            int2str = self._get_int2str_label(d)
            
            df['label'] = [int2str[i] for i in df['label']]
            df['dataset'] = [d] * len(df)
            
            self.df = pd.concat([self.df, df])
        
        if self.use_three_labels:
            self.df['label'] = [cfg.THREE_CATEGORIES.MERGED[i] for i in self.df['label']]
        else:
            self.df['label'] = pd.Categorical(pd.factorize(self.df['label'])[0])
        
        if drop_cols:
            self._drop_cols(drop_cols)

        # remove labels with value -1 
        remove_keys = [('label', -1)] if remove_keys == None else remove_keys + [('label', -1)] 
        if remove_keys:
            self._remove_with_keys(remove_keys)

        self._save_csv()   


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=Path, default='/root/dataset', help='Path to the data directory')
    parser.add_argument('--dataset', choices=['SAMM', 'SAMM_CROP', 'SMIC_all_cropped', 'SMIC_all_raw', 'CK+', 'CK+_cropped', 'OULU', 'OULU_cropped', 'CASME2', 'CASME2_cropped', 'merge'], help='Dataset to convert')
    parser.add_argument('--three_labels', action='store_true', default=False, help='Make labels to 3 category (pos, neg, surprise)')
    parser.add_argument('--csvs', nargs='*', default=None, help='csv files to be merged')

    args = parser.parse_args()

    if not os.path.exists('dataset_csv'):
        os.mkdir('dataset_csv')

    # merge files
    if args.csvs:
        csvMaker = MergedDataset(path=None, dataset=args.csvs, use_three_labels=args.three_labels)
        csvMaker.merge()

    else:
        if 'SAMM' in args.dataset:
            csvMaker = SAMM(path=args.dir, dataset=args.dataset, use_three_labels=args.three_labels)
        elif 'SMIC' in args.dataset:
            csvMaker = SMIC(path=args.dir, dataset=args.dataset, use_three_labels=args.three_labels)
        elif 'CASME2' in args.dataset:
            csvMaker = CASME2(path=args.dir, dataset=args.dataset, use_three_labels=args.three_labels)
        elif 'CK' in args.dataset:
            csvMaker = CK(path=args.dir, dataset=args.dataset, use_three_labels=args.three_labels)
        elif 'OULU' in args.dataset:
            csvMaker = OULU(path=args.dir, dataset=args.dataset, use_three_labels=args.three_labels)  
        else:
            KeyError(f'{args.dataset} is not supported!')
        
        csvMaker.create_csv()