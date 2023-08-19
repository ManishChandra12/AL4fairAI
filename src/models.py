import torch
import torch.nn as nn
import torchvision.models as models
#import torch.nn.functional as F

class CustomResnet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        self.features = nn.ModuleList(resnet18.children())[:-1]
        self.features = nn.Sequential(*self.features)
        in_features = resnet18.fc.in_features #resnet, googlenet
        self.sm = nn.Softmax(dim=1)
        self.fc = nn.Linear(in_features, 2)

    def forward(self, input_imgs, aux_task=False, joint=False):
        output = self.features(input_imgs)
        output = output.view(input_imgs.size(0), -1)
        output_p = self.fc(output)
        return self.sm(output_p)

class CustomResnet18_inprocess(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        self.features = nn.ModuleList(resnet18.children())[:-1]
        self.features = nn.Sequential(*self.features)
        in_features = resnet18.fc.in_features #resnet, googlenet
        self.sm1 = nn.Softmax(dim=1)
        self.sm2 = nn.Softmax(dim=1)
        self.fc1 = nn.Linear(in_features, 2)
        self.fc2 = nn.Linear(in_features+2, 2)

    def forward(self, input_imgs, sensitive1, sensitive2, aux_task=False, joint=False):
        output = self.features(input_imgs)
        output = output.view(input_imgs.size(0), -1)
        output_p = self.fc1(output)
        sensitive1 = sensitive1.view(input_imgs.size(0), -1)
        sensitive2 = sensitive2.view(input_imgs.size(0), -1)
        output_a = torch.cat((output, sensitive1, sensitive2), 1)
        output_a = self.fc2(output_a)
        return self.sm1(output_p), self.sm2(output_a)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(15, 10)
        self.fc2 = nn.Linear(10, 2)
        self.sm = nn.Softmax(dim=1)
        self.r = nn.ReLU()

    def forward(self, input_i, aux_task=False, joint=False):
        output = self.r(self.fc1(input_i.float()))
        output = self.fc2(output)
        return self.sm(output)

class Net_inprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(15, 10)
        self.fc2 = nn.Linear(10, 2)
        self.fc3 = nn.Linear(10+1, 2)
        self.sm1 = nn.Softmax(dim=1)
        self.sm2 = nn.Softmax(dim=1)
        self.r = nn.ReLU()

    def forward(self, input_i, sensitive, aux_task=False, joint=False):
        output = self.r(self.fc1(input_i.float()))
        output_p = self.fc2(output)
        sensitive = sensitive.view(input_i.size(0), -1)
        output_a = torch.cat((output, sensitive), 1)
        output_a = self.fc3(output_a)
        return self.sm1(output_p), self.sm2(output_a)

