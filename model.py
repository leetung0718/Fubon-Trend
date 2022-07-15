from PIL import Image
import matplotlib.pyplot as plt
plt.ion()
from tqdm import tqdm
import numpy as np
import time
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from efficientnet_pytorch import EfficientNet


########################################################################
# config
########################################################################
"""
Data
"""
IMGSPATH = "/Users/tunglee/Desktop/Fubon-CNN/imgs"
TRAINPATH = IMGSPATH + '/' + 'train'
VAILDPATH = IMGSPATH + '/' + 'val'
INPUT_SIZE = min(Image.open(TRAINPATH + '/' + '1' + '/' + '0.png').size)
BATCH_SIZE = 10

"""
CPU/GPU
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}\n")

########################################################################
# Data
########################################################################
def visualize_model(model, device, dataloaders, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0

    plt.figure(figsize=(18,9))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1

                img_display = np.transpose(inputs.cpu().data[j].numpy(), (1,2,0)) #numpy:CHW, PIL:HWC
                plt.subplot(num_images//2,2,images_so_far),plt.imshow(img_display) #nrow,ncol,image_idx
                plt.title(f'predicted: {class_names[preds[j]]}')
                plt.savefig("test.jpg")
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    #原先Normalize是對每個channel個別做 減去mean, 再除上std
    inp1 = std * inp + mean

    plt.imshow(inp)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.imshow(inp1)
    if title is not None:
        plt.title(title)

data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(INPUT_SIZE),
        # transforms.Resize(input_size),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize(560),
        transforms.CenterCrop(INPUT_SIZE),
        # transforms.Resize(input_size),
        transforms.ToTensor(),
    ])
}

image_datasets = {x: datasets.ImageFolder(IMGSPATH,
                                          data_transforms[x])
                 for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
                for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# # Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))
print(class_names)

# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[class_names[x] for x in classes])


# ########################################################################
# # Model
# ########################################################################
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# #using efficientnet model based transfer learning
# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         self.resnet =  EfficientNet.from_pretrained('efficientnet-b0')
#         self.l1 = nn.Linear(1000 , 256)
#         self.dropout = nn.Dropout(0.75)
#         self.l2 = nn.Linear(256,6)
#         self.relu = nn.ReLU()

#     def forward(self, input):
#         x = self.resnet(input)
#         x = x.view(x.size(0),-1)
#         x = self.dropout(self.relu(self.l1(x)))
#         x = self.l2(x)
#         return x

# model_ft = Classifier().to(device)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, len(class_names))

# model_ft = model_ft.to(device)
# parameter_count = count_parameters(model_ft)
# print(f"#parameters:{parameter_count}")
# print(f"batch_size:{BATCH_SIZE}")

# ########################################################################
# # Training
# ########################################################################
# def train_model(model, criterion, device, dataloaders, dataset_sizes, optimizer, scheduler, num_epochs=25):
#     since = time.time()

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#     train_loss, valid_loss = [], []
#     train_acc, valid_acc = [], []

#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch+1, num_epochs))
#         print('-' * 10)

#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode

#             running_loss = 0.0
#             running_corrects = 0

#             # Iterate over data.
#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         # zero the parameter gradients
#                         optimizer.zero_grad()
#                         loss.backward()
#                         optimizer.step()

#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#             if phase == 'train':
#                 scheduler.step()

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             if phase == 'train':
#                 train_loss.append(epoch_loss)
#                 train_acc.append(epoch_acc)
#             else:
#                 valid_loss.append(epoch_loss)
#                 valid_acc.append(epoch_acc)

#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))

#             # deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())


#     plt.figure(0)
#     plt.plot(range(1,num_epochs+1,1), np.array(train_loss), 'r-', label= "train loss") #relative global step
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend()
#     plt.savefig(f"./train_loss.png")

#     plt.figure(1)
#     plt.plot(range(1,num_epochs+1,1), np.array(valid_loss), 'b-', label= "eval loss") #--evaluate_during_training True 在啟用eval
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend()
#     plt.savefig(f"./eval_loss.png")

#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))

#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     #torch.save(model.state_dict(),"model.pt")
#     return model

# ########################################################################
# # Main
# ########################################################################

# """
# Model Config
# """
# LR = 1e-3
# CRITERION = nn.CrossEntropyLoss()
# # Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# model_ft = train_model(model_ft, CRITERION, device, dataloaders, dataset_sizes, optimizer_ft, exp_lr_scheduler, num_epochs=1)
# visualize_model(model_ft, device, dataloaders, class_names)
