from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
# from efficientnet.model import EfficientNet
from tensorboardX import SummaryWriter
from RAdam import *
import random
import numpy as np
from efficientnet.model_effnet import EfficientNet
from efficientnet.model_effnet import get_effnet
import json
import shutil
from ranger import *


## RESUME
resume = True
model_path = '/workspace/hwr/EfficientNet-Pytorch/Results/0617_newEff_Radam_0.001_class800_size112_0617_800_cleanaug5_tungaug20_0606pretrain102_5/models/epoch-0.pth'


## Set random seeds ##
random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


# CUDA
use_gpu = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)
print('use_gpu:',use_gpu)
print(os.environ["CUDA_VISIBLE_DEVICES"])
print(torch.cuda.set_device)


# Cofiguration
data_dir  = 'dataset/0618_tungvalwrong1:30_c10_w30'
batch_size = 128 
lr = 0.0001
num_epochs = 200
input_size = 112
class_num = 800


# Save path
exp_name = '0618_newEff_Ranger_0.0001_class800_size112_0618_tungvalwrong1:30_c10_w30_pretrain102_5_0-1' 
save_path = f'./Results/{exp_name}'
model_root = os.path.join(save_path,'models')
os.makedirs(model_root,exist_ok=True)


# Tensorboard
log_path = os.path.join(save_path,'logs')
times = time.localtime()
time_stamp = time.strftime("%Y-%m-%d_{}:%M:%S".format(times[3], time.localtime()))
log_dir = os.path.join(log_path,time_stamp)
os.makedirs(log_dir,exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)


# save setting
settings = [f'exp_name:{exp_name}',
            f'time_stamp:{time_stamp}',
             'Network:EfficientNet[new]',
            f'input_size:{input_size}',
            f'class_num:{class_num}',
            f'lr:{lr}',
            f'batch_size:{batch_size}',
            f'num_epochs:{num_epochs}',
            f'Data:{data_dir}',]

with open(f'{save_path}/setting.txt','w')as f:
    for item in settings :
        f.write(item + '\n')

def loaddata(data_dir, batch_size, set_name, shuffle):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [set_name]}

    dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                      batch_size=batch_size,
                                                      shuffle=shuffle, num_workers=4) for x in [set_name]}
    data_set_sizes = len(image_datasets[set_name])

    return dataset_loaders, data_set_sizes


def train_model(model_ft, criterion, optimizer, lr_scheduler, num_epochs=50):
    train_loss = []
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_acc = 0.0
    model_ft.train(True)

    for epoch in range(num_epochs):
        dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=batch_size, set_name='train', shuffle=True)
        print('Data Size', dset_sizes)

        running_loss = 0.0
        running_corrects = 0
        count = 0

        for data in dset_loaders['train']:
            inputs, labels = data

            labels = torch.squeeze(labels.type(torch.LongTensor))
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1
            if count % 20 == 0 or outputs.size()[0] < batch_size:
                print('Epoch:{} | {}/{} |loss:{:.3f}'.format(epoch,count,len(dset_loaders['train']),loss.item()))
                train_loss.append(loss.item())

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.double() / dset_sizes

        print('Epoch:{} | Loss: {:.4f} Acc: {:.4f}'.format(epoch,epoch_loss, epoch_acc))
        writer.add_scalar('train/loss',epoch_loss, epoch)
        writer.add_scalar('train/accuracy',epoch_acc , epoch)

        if epoch % 1 == 0:
            test_loss , test_acc = test_model(model_ft, criterion) 
            writer.add_scalar('test/loss',test_loss, epoch)
            writer.add_scalar('test/accuracy',test_acc , epoch)

        if (test_acc > best_acc):
            best_acc = test_acc
        if test_acc > 0.6:
            model_wts = model_ft.state_dict()
            model_ft.load_state_dict(model_wts) 
            model_root = os.path.join(save_path,'models')
            print(model_root)
            model_out_path = f'{model_root}/epoch-{epoch}.pth'
            torch.save(model_ft, model_out_path)


    # save best model
    model_ft.load_state_dict(best_model_wts)
    model_out_path = f'{model_root}/best.pth'
    torch.save(model_ft, model_out_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return train_loss, best_model_wts


def test_model(model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=64, set_name='test', shuffle=False)
    for data in dset_loaders['test']:
        inputs, labels = data
        labels = torch.squeeze(labels.type(torch.LongTensor))
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        if cont == 0:
            outPre = outputs.data.cpu()
            outLabel = labels.data.cpu()
        else:
            outPre = torch.cat((outPre, outputs.data.cpu()), 0)
            outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        cont += 1


    print('Loss: {:.4f} Acc: {:.4f}'.format(running_loss / dset_sizes,
                                            running_corrects.double() / dset_sizes))
    return running_loss, running_corrects.double() / dset_sizes


def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=100):
    lr = init_lr * (0.8**(epoch // lr_decay_epoch))
    # print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


# train
pth_map = {
    'efficientnet-b0': 'efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'efficientnet-b7-dcc49843.pth',
}

model_ft = get_effnet('efficientnet-b0',class_num)

## FC
if resume:
    model_ft = torch.load(model_path)
else:
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, class_num)

criterion = nn.CrossEntropyLoss()

if use_gpu:
    model_ft = model_ft.cuda()
    criterion = criterion.cuda()

optimizer = Ranger(model_ft.parameters(),lr=lr)
train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)

