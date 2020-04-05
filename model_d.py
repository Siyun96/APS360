import torch
import torch.nn as nn
import torch.nn.functional as F

## Fewer feature and 1 less fc layer
class ModelD(nn.Module):
    def __init__(self, feature1=40, feature2=80, feature3=80, hidden=200):
        super(ModelD, self).__init__()

        self.name = 'd'
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

        self.fc1 = nn.Linear(6 * 6 * feature3 + 16 * 16 * feature2, hidden)
        self.fc2 = nn.Linear(hidden, 24)

        self.feature2 = feature2
        self.feature3 = feature3
        

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        temp = F.relu(self.conv2(x))           # save a copy for fc
        x = self.pool2(temp)
        x = F.relu(self.conv3(x))

        x = x.view(-1, 6 * 6 * self.feature3)            # flatten x
        temp = temp.view(-1, 16 * 16 * self.feature2)    # flatten temp
        x = torch.cat((x, temp), 1)            # double check, should be dimension 1

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x