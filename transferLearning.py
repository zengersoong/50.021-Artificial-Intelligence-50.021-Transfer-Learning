#!/usr/bin/env python
# coding: utf-8

# In[27]:


"""
Written by : Soong Cun Yuan
SUTD : 1002074
Course: Artificial Intelligence 50.021
Course Instructor Professor Alexander Binder
Transfer Learning
Fine Tuning of Neural Networks

"""
import scipy as sp
import numpy as np
import csv
import random
import os
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import torch
import torch.utils.data as utils
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
photo_dir = "photos_8000/"
import sys
import matplotlib


# In[2]:


# [1] ---Data preprocessing [Start]--- 
data= np.genfromtxt('concepts_2011.txt')


# In[3]:


file = 'trainset_gt_annotations.txt'
train_set =[]
f = open(file, 'r')
no_lines = 0
for line in f:
    line_split = line.split()
    train_set.append(line_split)
    no_lines+=1
print(no_lines)


# In[29]:


# Annotation
# index 14 = indoor, index 15 = outdoor
outdoor = []
indoor = []
for i in range(no_lines):
    #index zero is y value therefore +1
    if train_set[i][14]=='1':
        indoor.append(train_set[i])
    if train_set[i][15]=='1':
        outdoor.append(train_set[i])
print("Total size of Indoor data: {}\nTotal Size of Outdoor data: {}".format(len(indoor),len(outdoor)))


# In[5]:


def xy_splitter(ar):
    """
    [1] Shuffles the data.
    [2] Split data into features and target.
    [3] Save it as name of file first.
    """
    random.shuffle(ar)
    x = []
    y = []
    for n in range(len(ar)):
#         x.append(Image.open(photo_dir+ar[n][0][:-4]+".jpg").convert('RGB'))
#         x.append(np.load(photo_dir+ar[n][0]+"_ft.npy"))
        x.append(photo_dir+ar[n][0][:-4]+".jpg")

        y.append(ar[n][1:])
    return x,y


# In[6]:


# Splitting the xy into features and targets
features_i, target_i = xy_splitter(indoor)
features_o, target_o = xy_splitter(outdoor)
# Split them into train, validation and test accordingly
# Split training : validation : test in 0.7:0.15:0.15 ratio 

# This is done with respect to the proportion of indoor : outdoor
itrain_size = int(np.floor((len(indoor))*0.7))
ivalidate_size = int(np.floor((len(indoor))*0.15))
itest_size =  int(len(indoor) - itrain_size - ivalidate_size)

otrain_size = int(np.floor((len(outdoor))*0.7))
ovalidate_size = int(np.floor((len(outdoor))*0.15))
otest_size =  int(len(outdoor) - otrain_size - ovalidate_size)

# This is done independently when splitting to ensure that training, validating and testing, we get an equal portion of indoor/outdoor in the sets.
indoor_train_dataset, indoor_validate_dataset ,indoor_test_dataset = torch.utils.data.random_split(indoor, [itrain_size, ivalidate_size, itest_size])
outdoor_train_dataset, outdoor_validate_dataset ,outdoor_test_dataset = torch.utils.data.random_split(outdoor, [otrain_size, ovalidate_size, otest_size])

# Create the full splitted datasets # Note remember to shuffle
train_dataset = indoor_train_dataset + outdoor_train_dataset
validate_dataset = indoor_validate_dataset + outdoor_validate_dataset
test_dataset = indoor_test_dataset + outdoor_test_dataset

# [1] ---Data preprocessing [End]--- 


# In[8]:


# [2] ---Define class the training,validation and test sets [start]---
class Dataset(utils.Dataset):
    """
    Dataset for transfer learning
    """
    def __init__(self, dataset,transform):
        """ 
        [1] Input the dataset
        """
        self.dataset = np.array(dataset)
        self.feature, self.target = xy_splitter(np.array(self.dataset))
        self.len = len(self.dataset)
        self.transform = transform
    
    def __len__(self):
        return self.len

    def __getitem__(self,index):
        # here we load the image 
        feature = self.feature[index]
#         print(feature)
        feature = Image.open(feature).convert('RGB')
        feature_t = self.transform(feature)
        target_t = self.target[index]
        # indoor is 1, outdoor is 0.
        _, target_t = torch.max(torch.tensor([int(target_t[13]), int(target_t[14])]), 0)
        if self.transform == five:
                # if using five crop we need multiple the target by five
            target_t = torch.stack((target_t,target_t,target_t,target_t,target_t))
        data = [feature_t,target_t]
        return data
# All the transform.
# five crop
five = transforms.Compose([
        transforms.Resize(280),
        transforms.FiveCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])(crop) for crop in crops])),
        ])
# single crop with random resize and flips
one = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
# [2] ---Define class for training,validation and test sets [End]---


# In[10]:


# train a deep neural network in three different modes A,B,C


# In[11]:


# [3] ---Define Train and Evaluate methods [Start]---

def train(model, dataloader,optimizer,criterion,transform_type):
    model.train()
    no_data = len(dataloader.dataset)
    correct = 0
    if transform_type ==one:
        for n in dataloader:
            image = n[0].to(device)
            target = n[1].to(device)
            target = target.view([-1])
            optimizer.zero_grad() # set to zero 
            results = model(image)
            pred = results.argmax(dim=1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss = criterion(results,target)
            loss.backward() # Update weights
            #optimizer.step updates the value of x using the gradient x.grad
            optimizer.step() # x.grad += dloss/dx
    if transform_type == five:
        for n in dataloader:
            # Make pytorch Infer the shape instead of going crazy
            image = n[0].view([-1,n[0].shape[-3],n[0].shape[-2],n[0].shape[-1]])
            image = image.to(device)
            target = n[1].to(device)
            target = target.view([-1])
            optimizer.zero_grad()
            results = model(image)
            pred = results.argmax(dim=1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss = criterion(results,target)
            loss.backward()
            optimizer.step()
    accuracy = correct/no_data
#     print("\nEpoch:{} \nTraining Loss is {}".format(epoch_num,loss))
    return loss , accuracy

def evaluate(model,dataloader,criterion):
    tLoss = 0
    correct = 0
    model.eval()
    no_data = len(dataloader.dataset)
    with torch.no_grad():
        for n in dataloader:
            image, target = n[0].to(device) ,n[1].to(device)
            results = model(image)
            tLoss += criterion(results,target).item()
            
            pred = results.argmax(dim=1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / no_data
    tLoss = tLoss/no_data
#     print("Evaluation Loss :{}\n Accuracy :{}".format(tLoss,accuracy))
    return tLoss, accuracy

# [3] ---Define Train and Evaluate methods [End]---


# In[12]:


# [A]--- Transfer learning once without loading weights and training all layers[Start]---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# trying out resnet for part A
resnet = models.resnet18(pretrained=False)
# Model resnet
num_ftrs = resnet.fc.in_features
# print(num_ftrs)
resnet.fc = nn.Linear(num_ftrs, 2)
resnetA=resnet.to(device)
criterion = nn.CrossEntropyLoss()
batchSize = 4
optimizer = optim.SGD(resnet.parameters(), lr = 1e-3, momentum=0.9)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,10)

# Create Training Instance and its loader, THIS IS DONE ONCE FOR ALL THREE VERSION OF TRANSFER LEARNING
trainInstance = Dataset(train_dataset, one)
trainLoader = utils.DataLoader(trainInstance, batchSize,
                        shuffle=True, num_workers=0)

# Create validation Instance and its loader
validInstance = Dataset(validate_dataset,one)
validLoader = utils.DataLoader(validInstance, batchSize,
                        shuffle=True, num_workers=0)

# Create test Instance and its loader
testInstance = Dataset(test_dataset,one)
testLoader = utils.DataLoader(testInstance, batchSize,
                        shuffle=True, num_workers=0)


# In[13]:


epochs = 30
lowest_loss = sys.maxsize
epoch_record = []
trainLoss_record = []
trainAccuracy_record = []
valLoss_record = []
valAccuracy_record=[]


for epoch in range(epochs):
    print("\nStarting Next Epoch\nLearning Rate:",scheduler.get_lr())
    epoch_record.append(epoch)
    print("Training Now... Epoch[{} / {}]".format(epoch+1,epochs))
    train_loss, train_accuracy = train(resnetA,trainLoader,optimizer,criterion,one)
    trainLoss_record.append(train_loss)
    trainAccuracy_record.append(train_accuracy)
    scheduler.step()
    
    print("Validating Now... Epoch[{} / {}]".format(epoch+1,epochs))
    validation_loss, validation_accuracy = evaluate(resnetA,validLoader,criterion)
    valLoss_record.append(validation_loss)
    valAccuracy_record.append(validation_accuracy)
    if validation_loss<= lowest_loss:
        lowest_loss = validation_loss
        print("Found a better loss! Epoch: {} Validation Loss:{}  Validation Accuracy:{}".format(epoch+1,validation_loss,validation_accuracy))
        torch.save(resnetA.state_dict(),"bestweights.pt")
    print("\nEpoch Summary of Epoch:{}\nTraining Loss:{} train_accuracy:{}\nValidation Loss:{}  Validation Accuracy:{}".format(epoch+1,train_loss,train_accuracy,validation_loss, validation_accuracy))
    


# In[14]:


# Part A results,without loading weights and training all layers.
resnetA = models.resnet18(pretrained=False)
#  Fully connected layer funnel to 2 dimension indoor and outdoor
num_ftrs = resnet.fc.in_features
resnetA.fc = nn.Linear(num_ftrs, 2)
resnetA.load_state_dict(torch.load("bestweights.pt"))
resnetA = resnet.to(device)
test_loss, accuracy = evaluate(resnetA,testLoader,criterion)


# In[15]:


print("Test Loss:{} & Accuracy:{}".format(test_loss, accuracy))


# In[16]:


matplotlib.use('Agg')
plt.figure()
plt.title("[A]Training Loss versus Epoch")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epoch_record, trainLoss_record)
plt.show()
plt.savefig('trainLoss_A.png')

plt.figure()
plt.title("[A]Training Accuracy versus Epoch")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(epoch_record, trainAccuracy_record)
plt.show()
plt.savefig('trainAcc_A.png')

plt.figure()
plt.title("[A]Validation Loss versus Epoch")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epoch_record, valLoss_record)
plt.show()
plt.savefig('validLoss_A.png')

plt.figure()
plt.title("[A]Validation Accuracy versus Epoch")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(epoch_record, valAccuracy_record)
plt.show()
plt.savefig('validAcc_A.png')

# [A]--- Transfer learning once without loading weights and training all layers[End]---


# In[17]:


# [B] --- Transfer learning once with loading model weights before training and training all layers[Start]---
resnetB = models.resnet18(pretrained=True) # Changed to True
# Model resnet
num_ftrs = resnetB.fc.in_features
resnetB.fc = nn.Linear(num_ftrs, 2)
resnetB=resnetB.to(device)
criterion = nn.CrossEntropyLoss()
batchSize = 4
optimizer = optim.SGD(resnetB.parameters(), lr = 1e-3, momentum=0.9)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,10)


# In[18]:


epochs = 30
lowest_loss = sys.maxsize
epoch_recordB = []
trainLoss_recordB = []
trainAccuracy_recordB = []
valLoss_recordB = []
valAccuracy_recordB=[]


for epoch in range(epochs):
    print("Learning Rate:",scheduler.get_lr())
    epoch_recordB.append(epoch)
    print("Training Now... Epoch[{} / {}]".format(epoch+1,epochs))
    train_loss, train_accuracy = train(resnetB,trainLoader,optimizer,criterion,one)
    trainLoss_recordB.append(train_loss)
    trainAccuracy_recordB.append(train_accuracy)
    scheduler.step()
    print("Validating Now... Epoch[{} / {}]".format(epoch+1,epochs))
    validation_loss, validation_accuracy = evaluate(resnetB,validLoader,criterion)
    valLoss_recordB.append(validation_loss)
    valAccuracy_recordB.append(validation_accuracy)
    if validation_loss<= lowest_loss:
        lowest_loss = validation_loss
        print("Found a better loss! Epoch: {} Validation Loss:{}  Validation Accuracy:{}".format(epoch,validation_loss,validation_accuracy))
        torch.save(resnetB.state_dict(),"bestweightsB.pt")
    print("\nEpoch:{} \nTraining Loss:{} train_accuracy:{}\nValidation Loss:{} Validation Accuracy:{}".format(epoch+1,train_loss,train_accuracy,validation_loss, validation_accuracy))


# In[19]:


# Part B results,loading weights and training all layers.
resnetB = models.resnet18(pretrained=True)
# Model resnet
num_ftrs = resnet.fc.in_features
resnetB.fc = nn.Linear(num_ftrs, 2)
resnetB.load_state_dict(torch.load("bestweightsB.pt"))
resnetB = resnetB.to(device)
test_loss, accuracy = evaluate(resnetB,testLoader,criterion)


# In[20]:


print("Test Loss:{} & Accuracy:{}".format(test_loss, accuracy))


# In[21]:


matplotlib.use('Agg')
plt.figure()
plt.title("[B]Training Loss versus Epoch")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epoch_recordB, trainLoss_recordB)
plt.show()
plt.savefig('trainLoss_B.png')

plt.figure()
plt.title("[B]Training Accuracy versus Epoch")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(epoch_recordB, trainAccuracy_recordB)
plt.show()
plt.savefig('trainAcc_B.png')

plt.figure()
plt.title("[B]Validation Loss versus Epoch")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epoch_recordB, valLoss_recordB)
plt.show()
plt.savefig('validLoss_B.png')

plt.figure()
plt.title("[B]Validation Accuracy versus Epoch")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(epoch_recordB, valAccuracy_recordB)
plt.show()
plt.savefig('validAcc_B.png')

# [B] --- Transfer learning once with loading model weights before training and training all layers[End]---


# In[22]:


# [C]--- Transfer learning oonce with training only last layer, freezing the others [Start]---
# Since Resnet has 4 layers with 1 conv layer, we dont update params for those layers
# Only train the last one.
resnetC = models.resnet18(pretrained=True)
for p in resnetC.conv1.parameters():
    p.requires_grad = False
for p in resnetC.layer1.parameters():
    p.requires_grad = False
for p in resnetC.layer2.parameters():
    p.requires_grad = False
for p in resnetC.layer3.parameters():
    p.requires_grad = False
#  Fully connected layer funnel to 2 dimension indoor and outdoor
num_ftrs = resnetC.fc.in_features
resnetC.fc = nn.Linear(num_ftrs, 2)
resnetC = resnetC.to(device)
optimizer = optim.SGD(resnetC.parameters(), lr = 1e-3, momentum=0.9)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,10)


# In[23]:


epochs = 30
lowest_loss = sys.maxsize
epoch_recordC = []
trainLoss_recordC = []
trainAccuracy_recordC = []
valLoss_recordC = []
valAccuracy_recordC=[]


for epoch in range(epochs):
    print("Learning Rate:",scheduler.get_lr())
    epoch_recordC.append(epoch)
    print("Training Now... Epoch[{} / {}]".format(epoch+1,epochs))
    train_loss, train_accuracy = train(resnetC,trainLoader,optimizer,criterion,one)
    trainLoss_recordC.append(train_loss)
    trainAccuracy_recordC.append(train_accuracy)
    scheduler.step()
    print("Validating Now... Epoch[{} / {}]".format(epoch+1,epochs))
    validation_loss, validation_accuracy = evaluate(resnetC,validLoader,criterion)
    valLoss_recordC.append(validation_loss)
    valAccuracy_recordC.append(validation_accuracy)
    if validation_loss<= lowest_loss:
        lowest_loss = validation_loss
        print("Found a better loss! Epoch: {} Validation Loss:{}  Validation Accuracy:{}".format(epoch,validation_loss,validation_accuracy))
        torch.save(resnetC.state_dict(),"bestweightsC.pt")
    print("\nEpoch:{} \nTraining Loss:{} train_accuracy:{}\nValidation Loss:{} & Accuracy:{}".format(epoch+1,train_loss,train_accuracy,validation_loss, validation_accuracy))
    


# In[24]:


# Part C results,loading weights and training all layers.
resnetC = models.resnet18(pretrained=True)
# Model resnet
num_ftrs = resnet.fc.in_features
resnetC.fc = nn.Linear(num_ftrs, 2)
resnetC.load_state_dict(torch.load("bestweightsC.pt"))
resnetC = resnetC.to(device)
test_loss, accuracy = evaluate(resnetC,testLoader,criterion)


# In[25]:


print("Test Loss:{} & Accuracy:{}".format(test_loss, accuracy))


# In[26]:


matplotlib.use('Agg')
plt.figure()
plt.title("[C]Training Loss versus Epoch")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epoch_recordC, trainLoss_recordC)
plt.show()
plt.savefig('trainLoss_C.png')

plt.figure()
plt.title("[C]Training Accuracy versus Epoch")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(epoch_recordC, trainAccuracy_recordC)
plt.show()
plt.savefig('trainAcc_C.png')

plt.figure()
plt.title("[C]Validation Loss versus Epoch")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epoch_recordC, valLoss_recordC)
plt.show()
plt.savefig('validLoss_C.png')

plt.figure()
plt.title("[C]Validation Accuracy versus Epoch")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(epoch_recordC, valAccuracy_recordC)
plt.show()
plt.savefig('validAcc_C.png')

# [C]--- Transfer learning oonce with training only last layer, freezing the others [End]---


# In[ ]:




