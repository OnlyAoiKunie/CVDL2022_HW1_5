import os
import sys
import copy
import cv2
import numpy
import torch
import torchvision
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image
from hw1_5_UI import Ui_MainWindow
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torchsummary import summary
from torch.utils.data import DataLoader

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



class myMainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(QtWidgets.QMainWindow, self).__init__()
        self.setupUi(self)
        self.onBindingUI()
        self.train_data = None
        self.test_data = None
        self.img = None
    def onBindingUI(self):
        self.pushButton.clicked.connect(self.loadImage)
        self.pushButton_2.clicked.connect(self.showTrainImages)
        self.pushButton_3.clicked.connect(self.showModelStructure)
        self.pushButton_4.clicked.connect(self.showDataAugmentation)
        self.pushButton_5.clicked.connect(self.showAccuracyAndLoss)
        self.pushButton_6.clicked.connect(self.inference)



    def loadImage(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'folder')
        # path = './dataBase/disparity/imL.png'
        self.img = path


    def showTrainImages(self):
        self.train_data = torchvision.datasets.CIFAR10(root="./data", train=True,
                                                  transform=torchvision.transforms.ToTensor(), download=True)
        self.test_data = torchvision.datasets.CIFAR10(root="./testdata", train=False,
                                                 transform=torchvision.transforms.ToTensor(), download=True)
        dataLoader = DataLoader(self.train_data , batch_size=9 , shuffle=True)
        imgs = None
        labels = None
        ListOfImgs = []
        LisOfLabel = []
        for data in dataLoader:
            imgs , labels = data
            for img in imgs:
                im = np.transpose(np.array(img) , (1,2,0))
                ListOfImgs.append(im)
            for label in labels:
                if(label.item() == 0):
                    LisOfLabel.append('airplane')
                elif(label.item() == 1):
                    LisOfLabel.append('automobile')
                elif (label.item() == 2):
                    LisOfLabel.append('bird')
                elif (label.item() == 3):
                    LisOfLabel.append('cat')
                elif (label.item() == 4):
                    LisOfLabel.append('deer')
                elif (label.item() == 5):
                    LisOfLabel.append('dog')
                elif (label.item() == 6):
                    LisOfLabel.append('frog')
                elif (label.item() == 7):
                    LisOfLabel.append('house')
                elif (label.item() == 8):
                    LisOfLabel.append('ship')
                elif (label.item() == 9):
                    LisOfLabel.append('trunk')
            break
        fig = plt.figure()
        for i in range(1,10):
            fig.add_subplot(3,3,i)
            plt.imshow(ListOfImgs[i - 1])
            plt.axis('off')
            plt.title(LisOfLabel[i - 1])
        plt.show()
    def showModelStructure(self):
        summary(MyModel().cuda() , (3,32,32))
    def showDataAugmentation(self):
        IMG = Image.open(self.img, mode='r')
        IMG = IMG.convert('RGB')

        #show image in UI
        cvImg = cv2.imread(self.img)
        cvImg = cv2.resize(cvImg ,(self.label.height() , self.label.width()))
        height , width ,channel = cvImg.shape
        bytesPerline = 3 * width
        qImg = QImage(cvImg.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(qImg))


        trans_RF = torchvision.transforms.RandomHorizontalFlip(p = 1)
        RF_img = trans_RF(IMG)
        trans_RV = torchvision.transforms.RandomAffine(30)
        RV_img = trans_RV(IMG)
        trans_RR = torchvision.transforms.RandomRotation((90))
        RR_img = trans_RR(IMG)
        fig = plt.figure()
        fig.add_subplot(1,3,1)
        plt.imshow(RF_img)
        plt.axis('off')
        fig.add_subplot(1, 3, 2)
        plt.imshow(RV_img)
        plt.axis('off')
        fig.add_subplot(1, 3, 3)
        plt.imshow(RR_img)
        plt.axis('off')
        plt.show()

    def showAccuracyAndLoss(self):
        cvImg = cv2.imread('tensorboard.PNG')
        cvImg = cv2.resize(cvImg, (self.label_4.height(), self.label_4.width()))
        height, width, channel = cvImg.shape
        bytesPerline = 3 * width
        qImg = QImage(cvImg.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_4.setPixmap(QPixmap.fromImage(qImg))
    def inference(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        myModel = MyModel()
        myModel.load_state_dict(torch.load('pretrainedVGG19.pth',map_location='cuda:0'))
        IMG = torch.Tensor(cv2.imread(self.img))
        reshapedIMG = torch.permute(IMG , (2,0,1))
        k = torch.unsqueeze(reshapedIMG , 0)
        myModel.eval()
        output = myModel(k)
        res = None

        cvImg = cv2.imread(self.img)
        cvImg = cv2.resize(cvImg, (self.label.height(), self.label.width()))
        height, width, channel = cvImg.shape
        bytesPerline = 3 * width
        qImg = QImage(cvImg.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label_4.close()
        self.label.setPixmap(QPixmap.fromImage(qImg))

        if (output.argmax(1).item() == 0):
            res = 'airplane'
        elif (output.argmax(1).item() == 1):
            res = 'automobile'
        elif (output.argmax(1).item() == 2):
            res = 'bird'
        elif (output.argmax(1).item() == 3):
            res = 'cat'
        elif (output.argmax(1).item() == 4):
            res = 'deer'
        elif (output.argmax(1).item() == 5):
            res = 'dog'
        elif (output.argmax(1).item() == 6):
            res = 'frog'
        elif (output.argmax(1).item() == 7):
            res = 'house'
        elif (output.argmax(1).item() == 8):
            res = 'ship'
        elif (output.argmax(1).item() == 9):
            res = 'trunk'

        self.label_2.setText('Confidence Score: {}'.format(torch.max(output)))
        self.label_3.setText('Prediction Label: {}'.format(res))



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = myMainWindow()
    window.show()
    sys.exit(app.exec_())