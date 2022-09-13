# Decalcomanie Data Augmentation for Micro-expresssion Recognition

PyTorch Code for the paper:   
["N-Step Pre-Training and Décalcomanie Data Augmentation for Micro-Expression Recognition"](https://www.mdpi.com/1424-8220/22/17/6671), MDPI Sensors 2022.  
Chaehyeon Lee, Jiuk Hong and Heechul Jung

Note! This code does not contain "N-step pre-training" part.  



## Installation
See [get_started.md](docs/get_started.md).

## Before you start training,
See [preprocessing.md](docs/preprocessing.md)

## Shared bacbone and multiple losses (SBML)

You can choose the code depending on the case instead of `SMBL_LR_infer_LR.py`.  
Check [here](SBML/).
``` 
#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 ./SMBL/SMBL_LR_infer_LR.py --dataset SAMM_CROP \
                                                          --model resnext101 \
                                                          --n-frames 34 \
                                                          --epochs 100 \
                                                          --img-size 112 \
                                                          --optimizer Adam \
                                                          --scheduler Step \
                                                          --lr 1e-4
```



## Fusion with shared backbone (FSB)
You can choose the code depending on the case instead of `FSB_LR_infer_LR.py`.  
Check [here](FSB/).  
``` 
#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 ./FSB/FSB_LR_infer_LR.py --dataset SAMM_CROP \
                                                          --model resnext101 \
                                                          --n-frames 34 \
                                                          --epochs 100 \
                                                          --img-size 112 \
                                                          --optimizer Adam \
                                                          --scheduler Step \
                                                          --lr 1e-4 \
                                                          [--use_mlp]
```
`--use_mlp` is an optional argument for late fusion experiment.




### Command line arguments
- `--dataset` : SAMM / SAMM_CROP / SMIC / SMIC_CROP / CASME2 / CASME2_CROP
- `--model` : resnext101, SwinTransformer(T, S, L, B)
  - Models are specified in [here](models/__init__.py).
- `--n-frames` : the number of frames.
- `--img-size` : input image size


## Citation

If this repository is helpful, please consider citing:  

```BibTex
@Article{s22176671,
AUTHOR = {Lee, Chaehyeon and Hong, Jiuk and Jung, Heechul},
TITLE = {N-Step Pre-Training and D&eacute;calcomanie Data Augmentation for Micro-Expression Recognition},
JOURNAL = {Sensors},
VOLUME = {22},
YEAR = {2022},
NUMBER = {17},
ARTICLE-NUMBER = {6671},
URL = {https://www.mdpi.com/1424-8220/22/17/6671},
PubMedID = {36081132},
ISSN = {1424-8220},
ABSTRACT = {Facial expressions are divided into micro- and macro-expressions. Micro-expressions are low-intensity emotions presented for a short moment of about 0.25 s, whereas macro-expressions last up to 4 s. To derive micro-expressions, participants are asked to suppress their emotions as much as possible while watching emotion-inducing videos. However, it is a challenging process, and the number of samples collected tends to be less than those of macro-expressions. Because training models with insufficient data may lead to decreased performance, this study proposes two ways to solve the problem of insufficient data for micro-expression training. The first method involves N-step pre-training, which performs multiple transfer learning from action recognition datasets to those in the facial domain. Second, we propose D&eacute;calcomanie data augmentation, which is based on facial symmetry, to create a composite image by cutting and pasting both faces around their center lines. The results show that the proposed methods can successfully overcome the data shortage problem and achieve high performance.},
DOI = {10.3390/s22176671}
}
```