import torch
import torch.nn as nn

import models.resnet as resnet
import models.resnext as resnext
from models.swin_transformer import Swin
from models.param import *

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(
            m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def build_model(model: str = False, n_classes: int = 7, img_size: int = 96, n_frames: int = 96, pretrained: str = None, use_mlp: bool = False, n_blocks: int = 1, hidden_dim: int = 256, init_layer: bool = False, ckpt_in_features: int = 101):
    if model == 'resnext101':
        last_fc = True
        if n_blocks != 1:
            last_fc = False

        net = resnext.resnet101(num_classes=ckpt_in_features, shortcut_type='B', cardinality=32,
                                      sample_size=img_size, sample_duration=n_frames,
                                      last_fc=last_fc)
        if pretrained is not None:
            checkpoint = torch.load(pretrained)
            net.load_state_dict(checkpoint['state_dict'])
            print('loaded!')
        
        if not use_mlp:
            net.fc = nn.Linear(net.fc.in_features * n_blocks, n_classes)
        else:
            net.fc = nn.Sequential(
                nn.Linear(net.fc.in_features * n_blocks, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, n_classes)
            )
        return net

    elif 'Swin' in model:
        if 'Swin_T' == model:
            P = P_SWIN_T
        elif 'Swin_S' == model:
            P = P_SWIN_S
        elif 'Swin_L' == model:
            P = P_SWIN_L
        else:
            P = P_SWIN_B

        net = Swin(
            P,
            in_chans=n_frames,
            num_classes=n_classes,
        )
        if pretrained is not None:
            print("Load Pretrained Model")
            # if info:
            #     net.load_checkpoint(f"macro_pretrain/{info}.pth")
            # else:
            #     net.load_checkpoint()
        return net    
    elif model == 'resnet18':
        net = resnet.resnet18(num_classes=ckpt_in_features, shortcut_type='A', sample_size=img_size, sample_duration=n_frames)
        
        if pretrained is not None:
            checkpoint = torch.load(pretrained)
            net.load_state_dict(checkpoint['state_dict'])
            print('loaded!') 
            if init_layer:
                net.layer4.apply(weights_init)
                print('layer4 is initialized!')
            
            
        net.fc = nn.Linear(net.fc.in_features, n_classes)
        return net
    elif model == 'resnet50':
        net = resnet.resnet50(num_classes=ckpt_in_features, shortcut_type='B', sample_size=img_size, sample_duration=n_frames)
        if pretrained is not None:
            checkpoint = torch.load(pretrained)
            net.load_state_dict(checkpoint['state_dict'])
            print('loaded!') 
            if init_layer:
                net.layer4.apply(weights_init)
                print('layer4 is initialized!')
        net.fc = nn.Linear(net.fc.in_features, n_classes)
        return net

    else:
        raise NotImplementedError
