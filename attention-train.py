import argparse

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.mnist import MNIST

from models import *


class Attention(nn.Module):
    def __init__(self, in_size=10):
        super(Attention, self).__init__()
        self.fc = nn.Linear(in_size, 1)

    def forward(self, x: list):
        return F.softmax(torch.cat([self.fc(header_out) for header_out in x], dim=1), dim=1)


class TrainData(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super(TrainData, self).__init__()
        self.transform = transform
        if args.dataset == 'MNIST':
            self.dataset = MNIST(root, train=True)
        else:
            self.dataset = CIFAR10(root, train=True)
        self.data = self.dataset.data[:args.data_num]
        self.targets = self.dataset.targets[:args.data_num]

    def __getitem__(self, index):
        if args.dataset == 'MNIST':
            img, target = self.data[index], int(self.targets[index])
            img = Image.fromarray(img.numpy(), mode='L')
        else:
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return args.data_num


def get_pred(outputs):
    weights = attention(outputs)
    return torch.einsum('ijk,ji->jk', outputs, weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'cifar10', 'cifar100'])
    parser.add_argument('--data', type=str, default='data/')
    parser.add_argument('--student_dir', type=str, default='ckps/')
    parser.add_argument('--header_num', type=int, default=3)
    parser.add_argument('--data_num', type=int, default=600)
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=100, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--output_dir', type=str, default='ckps/')

    args = parser.parse_args()

    attention = Attention().cuda()

    if args.dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        data_train = TrainData(args.data, transform=transform)
        data_test = MNIST(args.data, train=False, transform=transform)
        data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
        data_test_loader = DataLoader(data_test, batch_size=1024)

        net = lenet.MHLeNet5Half(args.header_num).cuda()
        net = nn.DataParallel(net)
        net.load_state_dict(torch.load(args.student_dir + 'student'))

        optimizer = torch.optim.Adam(attention.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss().cuda()

    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        data_train = TrainData(args.data, transform=transform)
        data_test = CIFAR10(args.data, train=False, transform=transform)
        data_train_loader = DataLoader(data_train, batch_size=100, shuffle=True, num_workers=0)
        data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=0)

        net = resnet.MultiHeaderResNet18(args.header_num).cuda()
        net = nn.DataParallel(net)
        net.load_state_dict(torch.load(args.student_dir + 'student'))

        optimizer = torch.optim.Adam(attention.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss().cuda()

    accr_best = 0
    net.eval()
    for epoch in range(args.n_epochs):

        total_correct = 0
        avg_loss = 0
        attention.train()
        for i, (images, labels) in enumerate(data_train_loader):
            optimizer.zero_grad()
            images, labels = images.cuda(), labels.cuda()

            outputs = torch.stack(net(images)).detach()
            output = get_pred(outputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if i == 1:
                print(f"[Epoch {epoch}/{args.n_epochs}] [loss: {loss:.4f}]")

        with torch.no_grad():
            attention.eval()
            for i, (images, labels) in enumerate(data_test_loader):
                images = images.cuda()
                labels = labels.cuda()

                outputs = torch.stack(net(images))
                output = get_pred(outputs)

                avg_loss += criterion(output, labels).sum()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
            avg_loss /= len(data_test)
            accr = round(float(total_correct) / len(data_test), 4)
            print(f'Test Avg. Loss: {avg_loss:.4f}, Accuracy: {accr}')
        if accr > accr_best:
            accr_best = accr
            torch.save(attention.state_dict(), args.output_dir + 'attention')
