## Make csv file
``` 
#!/bin/bash
 python3 ./tools/data2csv.py --dir [direcotry] --dataset [dataset]
```
### Directory structure
```
.
├── CASME2
│   ├── CASME2-coding-20190701.xlsx
│   └── Cropped
│       └── ...
├── CK
│   └── CK+
│       ├── Emotion
│       └── ck_cropped
├── OULU
│   └── PreProcess
│       ├── NI_Acropped
│       ├── VL_Acropped
│       └── change_log.txt
├── SMIC
│   ├── SMIC_all_cropped
│   │   ├── HS
│   │   ├── NIR
│   │   ├── VIS
│   │   └── li2013microexpressions.pdf
│   └── SMIC_all_raw
│       ├── HS
│       ├── NIR
│       └── VIS
└── samm
    └── SAMM_CROP
        ├── ...
        └── SAMM_Micro_FACS_Codes_v2.xlsx
```


## Create Decalcomanie frames
``` 
#!/bin/bash
 python3 ./tools/create_decal_copies.py --dataset [dataset]
```
