import imp
from tkinter import N
import torch.cuda
import torchvision
import torch.nn as nn
import torch.optim
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.transforms as transforms
train_data = torchvision.datasets.CIFAR10(root = "./data" , train=True , transform = torchvision.transforms.ToTensor(),download = True)
test_data = torchvision.datasets.CIFAR10(root = "./testdata" , train=False , transform = torchvision.transforms.ToTensor(),download = True)
writer = SummaryWriter('hw_1')
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.m = nn.Sequential(
            torchvision.models.vgg19(pretrained=True),
            nn.Linear(1000,10),
            nn.Softmax()
        )
    def forward(self , input):
        return self.m(input)
        
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
model = MyModel()
model.to(device)
train_dataloader = DataLoader(train_data , batch_size = 64 , shuffle=True)
test_dataloader = DataLoader(test_data , batch_size= 64 ,shuffle=False)
optimizer = torch.optim.SGD(model.parameters(),lr = 1e-3 , momentum=0.9)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
epoch = 30
train_epoch_loss = []
test_epoch_loss = []


for epoch_num in range(epoch):
    #train
    model.train()
    train_loss = 0.0
    train_correct = 0
    for data in tqdm(train_dataloader):
        imgs , labels = [t.to(device) for t in data]
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = loss_fn(output ,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        res = torch.argmax(output,axis = 1)
        train_correct += (res == labels).sum().item()


    train_loss /= len(train_data)
    train_acc = train_correct / len(train_data)
    print('epoch: {} train loss: {} train acc: {}'.format(epoch_num + 1 , train_loss,train_acc))
    train_epoch_loss.append(train_loss)
    torch.save(model.state_dict() , 'pretrainedVGG19.pth')


    #test
    model.eval()
    test_acc = 0.0
    test_loss = 0.0
    test_correct = 0
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            imgs, labels = [t.to(device) for t in data]
            output = model(imgs)
            res = torch.argmax(output ,axis = 1)
            test_correct += (res == labels).sum().item()
        test_acc = test_correct / len(test_data)

    writer.add_scalars('Loss', {'train':train_loss} ,epoch_num + 1)
    writer.add_scalars('Accuracy' , {'Training':train_acc , 'Testing':test_acc} , epoch_num + 1)

writer.close()
