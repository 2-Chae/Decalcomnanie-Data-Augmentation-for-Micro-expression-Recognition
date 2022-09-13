import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import random

from termcolor import cprint
from torchsampler import ImbalancedDatasetSampler
from torchvideotransforms import video_transforms, volume_transforms

if "__file__" in globals():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import *
from utils.misc import _create_info_folder
from utils.slack import post_message
from dataset import split_dataset, get_meta_data, MEDataset
from models import *

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Micro Facial Expression')

# Dataset
parser.add_argument('--dataset', choices=['SAMM', 'SAMM_CROP', 'SMIC_CROP', 'SMIC', 'CASME2', 'CASME2_CROP'], help='Dataset to convert', required=True)
parser.add_argument('--n-frames', default=25, type=int, help='the number of frames to be used for training')
parser.add_argument('--img-size', default=96, type=int, help='input image size')

# training env
parser.add_argument('--model', choices=['resnet50', 'resnext101', 'Swin_T', 'Swin_S', 'Swin_L', 'Swin_B'], help='model name to be used for training', default='resnext101')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--lr-decay', type=float, default=0.9, help='learning rate decay')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=100, help='# of epochs to train')
parser.add_argument('--optimizer', choices=['SGD', 'Adam', 'AdamW'], help='Choose optimizer', default='Adam')
parser.add_argument('--scheduler', choices=['Exp', 'Plateau', 'Step', 'None'], help='Choose scheduler', default='Exp')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--batch-train', type=int, default=10, help='batch size for train set')
parser.add_argument('--batch-test', type=int, default=1, help='batch size for test set')
parser.add_argument('--use_syn', action='store_true', help='if use synthetic images')
parser.add_argument('--checkpoint', default=None, type=str, help='path to pretrained weights')
parser.add_argument('--ckpt_in_features', default=101, type=int, help='in_features value for checkpoint')
parser.add_argument('--seed', type=int, default=338, help='random seed value')
# AMP
parser.add_argument('--amp', action='store_true', help='if use amp')
args = parser.parse_args()
print(args)


# fix seed 
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
random_seed = args.seed
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# dataset setting
csv_file = cfg.CSV[args.dataset]
# csv_file = 'dataset_csv/SMIC_all_cropped_2.csv'

df = pd.read_csv(csv_file)
if args.n_frames == 1: # Apex only
    df = df.dropna(axis=0)
assert len(df)!=0, 'Something wrong with the dataset!'
n_classes = len(df['label'].unique())
    
transform_list_train = [
    video_transforms.RandomRotation(30),
    video_transforms.RandomResize(ratio=(1.1, 1.1)),
    video_transforms.Resize((args.img_size, args.img_size)),
    video_transforms.RandomHorizontalFlip(),
    volume_transforms.ClipToTensor(div_255=False)]
transform_train = video_transforms.Compose(transform_list_train)   
   
transform_list_test = [
    video_transforms.Resize((args.img_size, args.img_size)),
    volume_transforms.ClipToTensor(div_255=True)]
transform_test = video_transforms.Compose(transform_list_test)   

total_metric = Metric(n_classes=n_classes) 
scaler = None

# log dir
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join(f'runs/{current_time}')
upper_board_writer = SummaryWriter(log_dir)
_create_info_folder(upper_board_writer, args, files_to_save=["./exp2/exp2_OR_infer_O.py", "dataset.py", "utils/config.py", "models/__init__.py"])

def test(epoch, dataloader, metric):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            metric.append_data(targets.cpu().numpy(), predicted.cpu().numpy())
            # if epoch == args.epochs - 1:
            #     total_metric.append_data(targets.cpu().numpy(), predicted.cpu().numpy())

            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    test_acc = 100.*correct/total
    test_loss = test_loss/(batch_idx+1)
    return test_acc, test_loss, metric



# Training
def train(dataloader, metric):
    net.train()
    train_loss = 0
    correct_ori = 0
    correct_r = 0
    total = 0

    for batch_idx, (inputs_ori, _, inputs_r, targets) in enumerate(dataloader):
        inputs_ori, inputs_r,  targets = inputs_ori.to(device), inputs_r.to(device),  targets.to(device)

        if args.amp:
            with torch.cuda.amp.autocast(): 
                outputs_ori = net(inputs_ori)
                loss_ori = criterion(outputs_ori, targets)

                outputs_r = net(inputs_r)
                loss_r = criterion(outputs_r, targets)

                loss = 0.5 * loss_ori + 0.5 * loss_r 
        else:
            outputs_ori = net(inputs_ori)
            loss_ori = criterion(outputs_ori, targets)

            outputs_r = net(inputs_r)
            loss_r = criterion(outputs_r, targets)

            loss = 0.5 * loss_ori + 0.5 * loss_r 

        optimizer.zero_grad()
        
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update() 
        else:
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        total += targets.size(0)
        
        _, predicted = outputs_r.max(1)
        correct_r += predicted.eq(targets).sum().item()

        _, predicted = outputs_ori.max(1)
        correct_ori += predicted.eq(targets).sum().item()

        metric.append_data(targets.cpu().numpy(), predicted.cpu().numpy())

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc(ori): %.3f%% (%d/%d) | Acc(r): %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct_ori/total, correct_ori, total,100.*correct_r/total, correct_r, total))

    train_acc = 100.*correct_ori/total
    train_loss = train_loss/(batch_idx+1)
    return train_acc, train_loss, metric

total_test_list = {'test_acc':[], 'test_uf1':[], 'test_uar':[]}
if 'MERGED' in args.dataset:
    datalist = df[df.type=='micro'].subject.unique()
else:
    datalist = df.subject.unique()
    
for idx, subject_out in enumerate(datalist):
    scaler = torch.cuda.amp.GradScaler() # AMP
 
   # model
    net = build_model(args.model, n_classes=n_classes, img_size=args.img_size, n_frames=args.n_frames, pretrained=args.checkpoint, ckpt_in_features=args.ckpt_in_features)
    net = net.to(device)
    if device == 'cuda':
        net = nn.DataParallel(net)
        cudnn.benchmark = True  
    
    if idx == 0:
        print(net)    
    
    # dataset
    print('\n\n#####################\n#', end='')
    cprint('{:^19}'.format('subject out: '+str(subject_out)), 'yellow', end='')
    print('#\n#####################')
    board_writer = SummaryWriter(os.path.join(log_dir, f'sub_{subject_out}'))

    train_df, test_df = split_dataset(df, subject_out, args.use_syn)
    train_df = shuffle(train_df)
    train_path_list, train_apex_list, train_labels = get_meta_data(train_df)
    test_path_list, test_apex_list, test_labels = get_meta_data(test_df)
    
    trainset = MEDataset(train_path_list, train_labels, train_apex_list, n_frames=args.n_frames,  transform=transform_train)
    testset = MEDataset(test_path_list, test_labels, test_apex_list, n_frames=args.n_frames, transform=transform_test, only_ori=True)
    # print('Input Shape: ', trainset[0][0].shape)
     
    trainloader = DataLoader(trainset, batch_size=args.batch_train, num_workers=16, sampler=ImbalancedDatasetSampler(trainset))
    testloader = DataLoader(testset, batch_size=args.batch_test, num_workers=16)

    # evaluation metric
    train_metric = Metric(n_classes=n_classes)
    test_metric = Metric(n_classes=n_classes)
    best_test_uf1 = 0.0
    best_test_uar = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    # opimizer
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05)


    # scheduler
    if args.scheduler == 'Exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    elif args.scheduler == 'Plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
    elif args.scheduler == 'Step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)
    
    test_list = {'test_uf1':[], 'test_uar':[]}
    for epoch in range(args.epochs):
        print('\n[Epoch: %d]' % epoch)
        
        train_acc, train_loss, train_metric = train(trainloader, train_metric)
        train_uf1, train_uar, train_war = train_metric.calculate()
    
        test_acc, test_loss, test_metric = test(epoch, testloader, test_metric)
        test_uf1, test_uar, train_war = test_metric.calculate()

        printFormatter((train_acc, train_uf1, train_uar), (test_acc, test_uf1, test_uar))

        if test_uf1 >= best_test_uf1:
            best_test_uf1 = test_uf1
            best_test_uar = test_uar
            b_targets = test_metric.targets.copy()
            b_preds = test_metric.preds.copy()

        # TODO: test_list 없애도 될듯..
        test_list['test_uf1'].append(test_uf1)
        test_list['test_uar'].append(test_uar)

        
        if args.scheduler == 'Exp' or args.scheduler == 'Plateau':
            scheduler.step(train_loss)
        elif args.scheduler == 'Step':
            scheduler.step()

        train_metric.reset()
        test_metric.reset()
            
        board_writer.add_scalar('Acc/train', train_acc, epoch)
        board_writer.add_scalar('Acc/test', test_acc, epoch)

        board_writer.add_scalar('UF1/train', train_uf1, epoch)
        board_writer.add_scalar('UF1/test', test_uf1, epoch)
        
        board_writer.add_scalar('UAR/train', train_uar, epoch)
        board_writer.add_scalar('UAR/test', test_uar, epoch)

        board_writer.add_scalar('Loss/train', train_loss, epoch)
        board_writer.add_scalar('Loss/test', test_loss, epoch)

    total_metric.append_data(b_targets, b_preds)

    for k, v in test_list.items():
        total_test_list[k].append(np.array(v)) 
    board_writer.close()

    total_uf1, total_uar, total_war = total_metric.calculate()
    print(f'total_uf1:{total_uf1}\ntotal_uar:{total_uar}\n, total_war:{total_war}\n')
    upper_board_writer.add_text('Best uf1', str(total_uf1), idx)
    upper_board_writer.add_text('Best uar', str(total_uar), idx)
    upper_board_writer.add_text('Best war', str(total_war), idx)
    del scaler

for k in total_test_list.keys():
    total_test_list[k] = np.array(total_test_list[k]).mean(0)

for epoch in range(args.epochs):
    # upper_board_writer.add_scalar('Avg Mine Acc', total_test_list['test_acc'][epoch], epoch)
    upper_board_writer.add_scalar('Avg UF1', total_test_list['test_uf1'][epoch], epoch)
    upper_board_writer.add_scalar('Avg UAR', total_test_list['test_uar'][epoch], epoch)

# confusion matrix
# total_uf1, total_uar = total_metric.calculate()
# cm_fig = plot_confusion_matrix(targets=total_metric.targets, pred=total_metric.preds, 
#                       target_names=cfg.LABELS[args.dataset], normalize=False, labels=True, title='Confusion Matrix')
# upper_board_writer.add_figure('Confusion matrix', cm_fig)

# save model checkpoint
model_checkpoints_ = os.path.join(upper_board_writer.log_dir, 'checkpoints', 'model.pth')
torch.save({
    'net': net.state_dict(),
}, model_checkpoints_)
upper_board_writer.close()


post_message('#alarm', f'exp2_OR_infer_O.py [{upper_board_writer.log_dir}] done! UF1:{total_uf1}, UAR:{total_uar}, WAR:{total_war}')