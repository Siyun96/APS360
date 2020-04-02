import torch
import torch.nn as nn
import torch.nn.functional as F

## No multi-scale CNN
class ModelA(nn.Module):
    def __init__(self, feature1=50, feature2=100, feature3=100, hidden1=200, hidden2=100):
        super(ModelA, self).__init__()

        self.name = 'a'
        # [48, 48, 3] => [40, 40, 50]
        self.conv1 = nn.Conv2d(3, feature1, 9)

        # [40, 40, 50] => [20, 20, 50]
        self.pool1 = nn.MaxPool2d(2, 2)


        # [20, 20, 50] => [16, 16, 100]
        self.conv2 = nn.Conv2d(feature1, feature2, 5)

        # [16, 16, 100] => [8, 8, 100]
        self.pool2 = nn.MaxPool2d(2, 2)


        # [8, 8, 100] => [6, 6, 100]
        self.conv3 = nn.Conv2d(feature2, feature3, 3)

        self.fc1 = nn.Linear(6 * 6 * feature3, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 24)

        self.feature3 = feature3
        

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        temp = F.relu(self.conv2(x))           # save a copy for fc
        x = self.pool2(temp)
        x = F.relu(self.conv3(x))

        x = x.view(-1, 6 * 6 * self.feature3)            # flatten x
        # temp = temp.view(-1, 16 * 16 * self.feature2)    # flatten temp
        # x = torch.cat((x, temp), 1)            # double check, should be dimension 1

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x