import torch
from torch import nn
from torch.nn import Module, Dropout, Tanh, LogSoftmax
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import MaxPool1d
from torch.nn import ReLU
from torch.nn import LogSigmoid
from torch import flatten
from torch.nn import BatchNorm1d


class OAEClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(960, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 14),
            nn.Softmax(dim=1)
        )
        self.encoder = nn.Sequential(
            nn.Linear(960, 128),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, 960)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return self.layers(decoded)


class OAEClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv1d(in_channels=1, out_channels=20,
        kernel_size=3)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool1d(kernel_size=2, stride=2)
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv1d(in_channels=20, out_channels=50,
        kernel_size=3)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool1d(kernel_size=2, stride=2)
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=11900, out_features=50)
        self.relu3 = ReLU()
        # initialize our softmax classifier
        self.fc2 = Linear(in_features=50, out_features=14)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output



class OAEClassifierSTFTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = Dropout(0.75)
        self.conv1 = Conv1d(in_channels=1, out_channels=258,
        kernel_size=3)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool1d(kernel_size=2, stride=2)
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv1d(in_channels=258, out_channels=200,
        kernel_size=3)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool1d(kernel_size=2, stride=2)
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=95600, out_features=400)
        self.relu3 = ReLU()
        # initialize our softmax classifier
        self.fc2 = Linear(in_features=400, out_features=42)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output


class OAEClassifierLargerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = Dropout(0.5)
        self.conv1 = Conv1d(in_channels=1, out_channels=258,
        kernel_size=3)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool1d(kernel_size=2, stride=2)
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv1d(in_channels=258, out_channels=200,
        kernel_size=3)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool1d(kernel_size=2, stride=2)
        ####
        self.conv3 = Conv1d(in_channels=200, out_channels=100,
                            kernel_size=3)
        self.relu3 = ReLU()
        self.maxpool3 = MaxPool1d(kernel_size=2, stride=2)
        ###
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=23800, out_features=42)
        #self.relu3 = ReLU()
        # initialize our softmax classifier
        #self.fc2 = Linear(in_features=5000, out_features=42)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        ###
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.dropout(x)
        ###

        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output


class OAEClassifierSmallerLayersCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = Dropout(0.5)
        self.conv1 = Conv1d(in_channels=1, out_channels=25,
        kernel_size=3)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool1d(kernel_size=2, stride=2)
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv1d(in_channels=25, out_channels=40,
        kernel_size=3)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool1d(kernel_size=2, stride=2)
        ####
        self.conv3 = Conv1d(in_channels=40, out_channels=20,
                            kernel_size=3)
        self.relu3 = ReLU()
        self.maxpool3 = MaxPool1d(kernel_size=2, stride=2)
        ###
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=4300, out_features=2150)
        self.relu3 = ReLU()
        # initialize our softmax classifier
        self.fc2 = Linear(in_features=2150, out_features=42)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        ###
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.dropout(x)
        ###

        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output

class one_d_cnn_1acc(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = Dropout(0.2)
        self.conv1 = Conv1d(in_channels=1, out_channels=10,
        kernel_size=3)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool1d(kernel_size=3, stride=3)
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv1d(in_channels=10, out_channels=100,
        kernel_size=3)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool1d(kernel_size=3, stride=3)
        ####
        self.conv3 = Conv1d(in_channels=100, out_channels=500,
                            kernel_size=3)
        self.relu3 = ReLU()
        self.maxpool3 = MaxPool1d(kernel_size=3, stride=3)
        ###
        self.conv4 = Conv1d(in_channels=500, out_channels=842,
                            kernel_size=3)
        self.relu4 = ReLU()
        self.maxpool4 = MaxPool1d(kernel_size=5, stride=1)
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=47994, out_features=220)
        self.relu5= ReLU()

        self.fc2 = Linear(in_features=220, out_features=42)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        ###
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        ###
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.dropout(x)
        ###

        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)


        x = self.fc1(x)
        x = self.relu5(x)

        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output

class relu(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = Dropout(0.3)
        self.conv1 = Conv1d(in_channels=1, out_channels=10,
        kernel_size=3)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool1d(kernel_size=3, stride=3)
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv1d(in_channels=10, out_channels=100,
        kernel_size=3)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool1d(kernel_size=3, stride=3)
        ####
        self.conv3 = Conv1d(in_channels=100, out_channels=500,
                            kernel_size=3)
        self.relu3 = ReLU()
        self.maxpool3 = MaxPool1d(kernel_size=3, stride=3)
        ###
        self.conv4 = Conv1d(in_channels=500, out_channels=842,
                            kernel_size=3)
        self.relu4 = ReLU()
        self.maxpool4 = MaxPool1d(kernel_size=5, stride=1)
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=47994, out_features=730)
        self.relu5= ReLU()

        self.fc2 = Linear(in_features=730, out_features=42)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        ###
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        ###
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.dropout(x)
        ###

        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)


        x = self.fc1(x)
        x = self.relu5(x)

        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output

class MainOAEClassifierSOf(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = Dropout(0.5)


        self.conv1 = Conv1d(in_channels=1, out_channels=500,
        kernel_size=3)
        self.tanh1 = Tanh()
        self.batchnorm1 = BatchNorm1d(500)
        self.maxpool1 = MaxPool1d(kernel_size=3, stride=3)



        ####
        self.conv2 = Conv1d(in_channels=500, out_channels=1000,
        kernel_size=3)
        self.tanh2 = Tanh()
        self.batchnorm2 = BatchNorm1d(1000)
        self.maxpool2 = MaxPool1d(kernel_size=3, stride=3)

        ####
        self.conv3 = Conv1d(in_channels=1000, out_channels=1750,
                            kernel_size=3)
        self.tanh3 = Tanh()
        self.batchnorm3 = BatchNorm1d(1750)
        self.maxpool3 = MaxPool1d(kernel_size=3, stride=3)

        ####
        self.conv4 = Conv1d(in_channels=1750, out_channels=2400,
                            kernel_size=3)
        self.tanh4 = Tanh()
        self.batchnorm4 = BatchNorm1d(2400)
        self.maxpool4 = MaxPool1d(kernel_size=5, stride=1)


        ####
        self.fc1 = Linear(in_features=48000, out_features=730)
        self.tanh5 = Tanh()
        self.batchnorm5 = BatchNorm1d(730)

        ####
        self.fc2 = Linear(in_features=730, out_features=36)
        self.batchnorm6 = BatchNorm1d(36)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.tanh1(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        x = self.dropout(x)

        ###
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        ###
        x = self.conv3(x)
        x = self.tanh3(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)
        x = self.dropout(x)

        ###
        x = self.conv4(x)
        x = self.tanh4(x)
        x = self.batchnorm4(x)
        x = self.maxpool4(x)
        x = self.dropout(x)

        ###
        x = flatten(x, 1)

        x = self.fc1(x)
        x = self.tanh5(x)
        x = self.batchnorm5(x)



        x = self.fc2(x)
        x = self.batchnorm6(x)

        output = self.logSoftmax(x)

        #return the output predictions
        return output


class MainOAEClassifierSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = Dropout(0.5)

        self.conv1 = Conv1d(in_channels=1, out_channels=500,
                            kernel_size=3)
        self.tanh1 = Tanh()
        self.batchnorm1 = BatchNorm1d(500)
        self.maxpool1 = MaxPool1d(kernel_size=3, stride=3)

        ####
        self.conv2 = Conv1d(in_channels=500, out_channels=1000,
                            kernel_size=3)
        self.tanh2 = Tanh()
        self.batchnorm2 = BatchNorm1d(1000)
        self.maxpool2 = MaxPool1d(kernel_size=3, stride=3)

        ####
        self.conv3 = Conv1d(in_channels=1000, out_channels=1750,
                            kernel_size=3)
        self.tanh3 = Tanh()
        self.batchnorm3 = BatchNorm1d(1750)
        self.maxpool3 = MaxPool1d(kernel_size=3, stride=3)

        ####
        self.conv4 = Conv1d(in_channels=1750, out_channels=2400,
                            kernel_size=3)
        self.tanh4 = Tanh()
        self.batchnorm4 = BatchNorm1d(2400)
        self.maxpool4 = MaxPool1d(kernel_size=5, stride=1)

        ####
        self.fc1 = Linear(in_features=48000, out_features=730)
        self.tanh5 = Tanh()
        self.batchnorm5 = BatchNorm1d(730)

        ####
        self.fc2 = Linear(in_features=730, out_features=36)
        self.batchnorm6 = BatchNorm1d(36)
        self.logSigmoid = LogSigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh1(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        x = self.dropout(x)

        ###
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        ###
        x = self.conv3(x)
        x = self.tanh3(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)
        x = self.dropout(x)

        ###
        x = self.conv4(x)
        x = self.tanh4(x)
        x = self.batchnorm4(x)
        x = self.maxpool4(x)
        x = self.dropout(x)

        ###
        x = flatten(x, 1)

        x = self.fc1(x)
        x = self.tanh5(x)
        x = self.batchnorm5(x)

        x = self.fc2(x)
        x = self.batchnorm6(x)
        output = self.logSigmoid(x)

        #return the output predictions
        return output


class MainOAEClassifierSigmoidRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = Dropout(0.5)

        self.conv1 = Conv1d(in_channels=1, out_channels=500,
                            kernel_size=3)
        self.tanh1 = ReLU()
        self.batchnorm1 = BatchNorm1d(500)
        self.maxpool1 = MaxPool1d(kernel_size=3, stride=3)

        ####
        self.conv2 = Conv1d(in_channels=500, out_channels=1000,
                            kernel_size=3)
        self.tanh2 = ReLU()
        self.batchnorm2 = BatchNorm1d(1000)
        self.maxpool2 = MaxPool1d(kernel_size=3, stride=3)

        ####
        self.conv3 = Conv1d(in_channels=1000, out_channels=1750,
                            kernel_size=3)
        self.tanh3 = ReLU()
        self.batchnorm3 = BatchNorm1d(1750)
        self.maxpool3 = MaxPool1d(kernel_size=3, stride=3)

        ####
        self.conv4 = Conv1d(in_channels=1750, out_channels=2400,
                            kernel_size=3)
        self.tanh4 = ReLU()
        self.batchnorm4 = BatchNorm1d(2400)
        self.maxpool4 = MaxPool1d(kernel_size=5, stride=1)

        ####
        self.fc1 = Linear(in_features=48000, out_features=730)
        self.tanh5 = ReLU()
        self.batchnorm5 = BatchNorm1d(730)

        ####
        self.fc2 = Linear(in_features=730, out_features=24)
        self.logSigmoid = LogSigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh1(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        x = self.dropout(x)

        ###
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        ###
        x = self.conv3(x)
        x = self.tanh3(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)
        x = self.dropout(x)

        ###
        x = self.conv4(x)
        x = self.tanh4(x)
        x = self.batchnorm4(x)
        x = self.maxpool4(x)
        x = self.dropout(x)

        ###
        x = flatten(x, 1)

        x = self.fc1(x)
        x = self.tanh5(x)
        x = self.batchnorm5(x)

        x = self.fc2(x)
        #output = self.logSigmoid(x)

        #return the output predictions
        return x

