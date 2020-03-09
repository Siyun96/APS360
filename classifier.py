import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # [48, 48, 3] => [40, 40, 50]
        self.conv1 = nn.Conv2d(3, 50, 9)

        # [40, 40, 50] => [20, 20, 50]
        self.pool1 = nn.MaxPool2d(2, 2)


        # [20, 20, 50] => [16, 16, 100]
        self.conv2 = nn.Conv2d(50, 100, 5)

        # [16, 16, 100] => [8, 8, 100]
        self.pool2 = nn.MaxPool2d(2, 2)


        # [8, 8, 100] => [6, 6, 100]
        self.conv3 = nn.Conv2d(100, 100, 3)

        self.fc1 = nn.Linear(6 * 6 * 100 + 16 * 16 * 100, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 24)


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        temp = F.relu(self.conv2(x))           # save a copy for fc
        x = self.pool2(temp)
        x = F.relu(self.conv3(x))

        x = x.view(-1, 6 * 6 * 100)            # flatten x
        temp = temp.view(-1, 16 * 16 * 100)    # flatten temp
        x = torch.cat((x, temp), 1)            # double check, should be dimension 1

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x