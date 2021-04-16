import torch.nn as nn


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 2, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(4, 60, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(60, 30)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(30, 10)

    def forward(self, img, out_feature=False):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        feature = output.view(-1, 60)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        if out_feature == False:
            return output
        else:
            return output, feature


class MHLeNet5Half(nn.Module):

    def __init__(self, header):
        super(MHLeNet5Half, self).__init__()
        self.headers = header

        self.conv1 = nn.Conv2d(1, 1, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(1, 2, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(2, 30, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()

        self.fc1 = nn.ModuleList([nn.Linear(30, 15) for _ in range(header)])
        self.relu4 = nn.ReLU()
        self.fc2 = nn.ModuleList([nn.Linear(15, 10) for _ in range(header)])

    def header_forward(self, x, from_header):
        output = self.fc1[from_header](x)
        output = self.relu4(output)
        output = self.fc2[from_header](output)
        return output

    def forward(self, img):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        output = output.view(-1, 30)
        output = [self.header_forward(output, from_header=i) for i in range(self.headers)]
        return output
