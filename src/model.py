import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, num_classes=5):
        super(VGG, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batch3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batch4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.batch5 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.batch6 = nn.BatchNorm1d(2 * 2 * 128)
        self.fc1 = nn.Linear(2 * 2 * 128, 64)
        self.fc2 = nn.Linear(64, num_classes)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.batch1(out)
        out = self.maxpool(out)

        out = self.relu(self.conv2(out))
        out = self.batch2(out)
        out = self.maxpool(out)

        out = self.relu(self.conv3(out))
        out = self.batch3(out)
        out = self.maxpool(out)

        out = self.relu(self.conv4(out))
        out = self.batch4(out)
        out = self.maxpool(out)

        out = self.relu(self.conv5(out))
        out = self.batch5(out)
        out = self.maxpool(out)

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.batch6(out)
        out = self.relu(self.fc1(out))
        out = self.softmax(self.fc2(out))

        return out
