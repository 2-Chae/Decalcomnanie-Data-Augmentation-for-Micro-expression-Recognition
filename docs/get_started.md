# Installation

## Requirements
- Python 3.9 +
- Pytorch 1.7.1
- TorchVision 0.8.2  
- Details are specified in [requirements.txt](requirements.txt).  

`pip install -r requirements.txt` before you run this.

## CSV file
Before you start training, you need a csv file for each dataset.

Make sure your data directory has this structure.
```
path/to/data
├── samm
│   ├── SAMM
│   └── SAMM_CROP
└── SMIC
    ├── SMIC_all_cropped
    └── SMIC_all_raw
```

and then run the below code
```
python data2csv.py --dir path/to/data --dataset [dataset_name] 

ex) python data2csv.py --dir path/to/data --dataset SAMM [--five-frames]
```
- dataset_name : SAMM / SAMM_CROP / SMIC_all_cropped / SMIC_all_raw  
[Note] `--five-frames` flag is optinal arguments to save 5 frames. Only works for apex dataset.

You can create these 5 files in `./dataset_csv`.
```
./dataset_csv
├── SAMM_CROP.csv
├── SAMM_CROP_five_frames.csv
├── SAMM.csv
├── SMIC_all_cropped.csv
└── SMIC_all_raw.csv
```

Finally it's time to train the model!!

