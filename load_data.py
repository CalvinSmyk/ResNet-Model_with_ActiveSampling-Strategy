import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle

class Data():
    def __init__(self, download,path,train):
        self.download = download
        self.path = path
        self.train = train

    def unpickle(self,file_to_open):
        with open(file_to_open, 'rb') as file:
            myDict = pickle.load(file,encoding='latin1')
        return myDict

    def open_data(self):
        Dict = self.unpickle(self.path)
        data = self.reshape_data(Dict['data'])
        labels = Dict['fine_labels']
        classes = self.define_classes()
        #self.transform()
        #data = self.transform_train(data) if self.train else self.transform_test(data)
        if self.train:
            return data,labels,classes
        else:
            return data,labels

    def reshape_data(self,Dict):
        Data = Dict.reshape(len(Dict),3,32,32).transpose(0,2,3,1)
        return Data


    def download_data(self):
        if self.download:
            return torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=self.transform)
        elif self.download == 0:
            return torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=self.transform)

    def transform(self):
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    def define_classes(self):
        classes = self.unpickle("./cifar-100-python/meta")
        return classes['fine_label_names']

