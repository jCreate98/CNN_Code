import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torchvision
from torchvision import transforms, datasets
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.io import read_image



import torch.nn.init
from ast import increment_lineno

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


import torch.nn as nn
import torch.nn.functional as F

import os
import pandas as pd

import cv2

if torch.cuda.is_available():
     DEVICE=torch.device('cuda')
else:
     DEVICE=torch.device('cpu')

print('Using Pytorch version:',torch.__version__,'Device:',DEVICE)

BATCH_SIZE=32
EPOCHS=10

root = "./train_Image"
test_root = "./test_Image"

trans = transforms.Compose([transforms.Resize((128,128)),
                            transforms.ToTensor(),
                            ])

tests = transforms.Compose([transforms.Resize((128,128)),
                            transforms.ToTensor(),
                            ])

test_dataset = torchvision.datasets.ImageFolder(root = test_root,
                                           transform = tests)

trainset = torchvision.datasets.ImageFolder(root = root,
                                           transform = trans)

train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          drop_last=True)

for(X_train,Y_train)in train_loader:
    print('X_train:',X_train.size(),'type:', X_train.type())
    print('Y_train:',Y_train.size(),'type:', Y_train.type())
    break

pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))

for i in range(10):
    plt.subplot(1,10,i+1)     # 여러 그래프 그리기, 첫숫자 : 행, 둘째 : 열
    plt.axis('off')           # 축없음
    plt.imshow(np.transpose(X_train[i],(1,2,0)))
    plt.title('Class: ' + str(Y_train[i].item()))
# imshow : 이미지 출력

class CNN(nn.Module):                 # nn.Module을 상속받는 Net클래스 생성
    def __init__(self):   
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, 
                               out_channels = 8, 
                               kernel_size = 3,
							   padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 8, 
                               out_channels = 16, 
                               kernel_size = 3,
							   padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 16,
		                       out_channels= 32,
							   kernel_size= 3,
							   padding = 1)

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1=nn.Linear(16*16*32,64)
        self.fc2=nn.Linear(64,32)   # 이전 output node수 = 다음 input node수
        self.fc3=nn.Linear(32,10)
        
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = x.view(-1, 16*16*32)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x,dim=1)
        return x

#7.Optimizer, Object Function 설정하기
model=CNN().to(DEVICE)                  # MLP모델을 기존에 선정한 'DEVICE'에 할당합니다. 
optimizer = torch.optim.Adam(model.parameters(),lr=0.005)
# Back Propagation을 이용해 파라미터를 업데이트할 때 이용하는 Optimizer를 정의합니다.
# SGD알고리즘을 이용하며 Learning Rate = 0.01, momentum=0.5로 설정
criterion=nn.CrossEntropyLoss()
# MLP모델의 output값과 계산될 Label값은 Class를 표현하는 원-핫 인코딩 값입니다.
# MLP모델의 output값과 원-핫 인코딩 값과의 Loss는 CrossEntropy를 이용해 계산하기 위해
# criterion은 nn.CrossEntropyLoss() 로 설정. 

print(model)

def train(model, train_loader, optimizer, log_interval):
    model.train()         # 학습상태로 지정
    for batch_idx,(image, label) in enumerate(train_loader):
        image=image.to(DEVICE)
        label=label.to(DEVICE)
        optimizer.zero_grad()
        output=model(image)
        loss=criterion(output,label)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval==0:
            print("Train Epoch: {}[{}/{}({:.0f}%)]\tTrain Loss: {:.6f}"
                  .format(Epoch,batch_idx * len(image),
                len(train_loader.dataset),100.*batch_idx/len(train_loader),loss.item()))

def evaluate(model,test_loader):
    model.eval()                  # 평가상태로 지정
    test_loss=0
    correct=0
    
    with torch.no_grad():
        for image, label in test_loader:
            image=image.to(DEVICE)              # 8과 동일
            label=label.to(DEVICE)              # 8과 동일
            output=model(image)                 # 8과 동일
            test_loss+=criterion(output,label).item()
            prediction = output.max(1,keepdim = True)[1]
            # MLP 모델의 output값은 크기가 10인 벡터값입니다. 
            # 계산된 벡터값 내 가장 큰 값인 위치에 대해 해당 위치에 대응하는 클래스로 예측했다고 판단합니다.
            correct += prediction.eq(label.view_as(prediction)).sum().item()
            # MLP모델이 최종으로 예측한 클래스 값과 실제 레이블이 의미하는 클래스가 맞으면 correct에 더해 올바르게 예측한 횟수를 저장
                        
    test_loss /= len(test_loader.dataset)
    # 현재까지 계산된 test_loss 값을 test_loader내에 존재하는 Mini-Batch 개수(=10)만큼 나눠 평균 Loss값으로 계산.
    test_accuracy=100.*correct / len(test_loader.dataset)
    # test_loader 데이터중 얼마나 맞췄는지를 계산해 정확도를 계산합니다.
    return test_loss, test_accuracy

for Epoch in range(1,EPOCHS+1):
    train(model,train_loader,optimizer,log_interval = 200)
    test_loss, test_accuracy=evaluate(model, test_loader)
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %\n".
          format(Epoch,test_loss,test_accuracy))