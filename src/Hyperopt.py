import os
import argparse
import logging
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.nn as nn
from cnn import torchModel
from datasets import K49
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils import AvgrageMeter, accuracy
import matplotlib
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import numpy as np
from glob import glob
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
from torch.optim import lr_scheduler


class torchModel(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), num_classes=49):
        super(torchModel, self).__init__()
        layers = []
        n_layers = 1
        n_conv_layers = 1
        kernel_size = 2
        in_channels = input_shape[0]
        out_channels = 4

        for i in range(n_conv_layers):
            c = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=2, padding=1
                          )
            a = nn.ReLU(inplace=False)
            p = nn.MaxPool2d(kernel_size=2, stride=1)
            layers.extend([c, a, p])
            in_channels = out_channels
            out_channels *= 2

        self.conv_layers = nn.Sequential(*layers)
        self.output_size = num_classes

        self.fc_layers = nn.ModuleList()
        n_in = self._get_conv_output(input_shape)
        n_out = 256
        for i in range(n_layers):
            fc = nn.Linear(int(n_in), int(n_out))
            self.fc_layers += [fc]
            n_in = n_out
            n_out /= 2

        self.last_fc = nn.Linear(int(n_in), self.output_size)
        self.dropout = nn.Dropout(p=0.2)

    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.conv_layers(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        for fc_layer in self.fc_layers:
            x = self.dropout(F.relu(fc_layer(x)))
        x = self.last_fc(x)
        return x


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device being used', device)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

space = {'batch_size': hp.uniform('batch_size', 140, 180),
         'learning_rate': hp.loguniform('x', np.log(0.00001), np.log(0.1)),
         'epochs': hp.randint('epochs', 20),
         }


def f_nn(params):
    '''
    lets say that you are making your own model from scratch,
    you could do something like this but be sure of the shapes that you get in(:number of inchannels)
    and also the the shape you output
    
    if params['choice']['layers']== 'two':
        self.fc1 = nn.Conv2d(channels, reduction, kernel_size=1, padding=0)
        # calling the model function here with obove paramters

    '''

    model = torchModel()
    model.to(device)
    print('Params testing: ', params)
    batch_size = int(params['batch_size'])

    data_augmentations = transforms.ToTensor()
    data_dir = '../data'
    train_dataset = K49(data_dir, True, data_augmentations)
    test_dataset = K49(data_dir, False, data_augmentations)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    Train_dataset_loader = train_loader
    Test_dataset_loader = test_loader
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    print('chossen learning rate', params['learning_rate'])
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    epochs = params['epochs']
    steps = 0
    train_losses, test_losses = [], []

    for e in range(epochs):
        correct = 0
        average_precision = []
        running_loss = 0
        model.train()
        exp_lr_scheduler.step()
        for images, labels in Train_dataset_loader:
            images, labels = Variable(images), Variable(labels)
            images, labels = images.to(device), labels
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()  # calculate loss for batch wise and add it to the previous value

        else:
            test_loss = 0
            accuracy = 0

            total = 0
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                model.eval()
                for images, labels in Test_dataset_loader:
                    images, labels = Variable(images), Variable(labels)
                    images, labels = images.to(device), labels.to(device)
                    ps = model(images)
                    test_loss += criterion(ps, labels.to(device))
                    _, predicted = torch.max(ps.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            train_losses.append(running_loss / len(Train_dataset_loader))
            test_losses.append(test_loss / len(Test_dataset_loader))
        if e == epochs - 1:
            print("Epoch: {}/{}.. ".format(e + 1, epochs), "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                  "Test Loss: {:.3f}.. ".format(test_losses[-1]), "Test Accuracy: {:.3f}".format(correct / total))
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    import matplotlib.pyplot as plt
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    loss = test_loss / len(Test_dataset_loader)
    return loss.detach().item()


trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=100, trials=trials)
print('best: ')
print(best)
