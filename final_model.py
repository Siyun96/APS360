import torch
import torch.nn as nn
import torch.nn.functional as F

# Final model for traffic sign classification
class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()

        self.name = 'final'
        # First stage
        # [48, 48, 3] => [40, 40, 30]
        self.conv1 = nn.Conv2d(3, 30, 9)

        # [40, 40, 30] => [20, 20, 30]
        self.pool1 = nn.MaxPool2d(2, 2)

        # [20, 20, 30] => [16, 16, 60]
        self.conv2 = nn.Conv2d(30, 60, 5)

        # Second stage
        # [16, 16, 60] => [8, 8, 60]
        self.pool2 = nn.MaxPool2d(2, 2)

        # [8, 8, 60] => [6, 6, 60]
        self.conv3 = nn.Conv2d(60, 60, 3)

        # Classifier
        self.fc1 = nn.Linear(6 * 6 * 60 + 16 * 16 * 60, 200)
        self.fc2 = nn.Linear(200, 24)
        

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        temp = F.relu(self.conv2(x))                     # save a copy for fc
        x = self.pool2(temp)
        x = F.relu(self.conv3(x))

        x = x.view(-1, 6 * 6 * 60)            # flatten x
        temp = temp.view(-1, 16 * 16 * 60)    # flatten temp
        x = torch.cat((x, temp), 1)                      # double check, should be dimension 1

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x