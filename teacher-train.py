import argparse
import os

import torch
import torch.optim as optim

from models import *
from utils import get_loader


def train(net, epochs, optimizer, criterion, data_train_loader, data_test_loader):
    acc_best = 0
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [80, 120], 0.1)
    for epoch in range(epochs):
        net.train()
        for i, (images, labels) in enumerate(data_train_loader):
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            if args.dataset != 'MNIST':
                scheduler.step()

            if i == 1:
                print(f'Train - Epoch {epoch}, Batch: {i}, Loss: {loss:.6f}')

        total_correct = 0
        avg_loss = 0
        with torch.no_grad():
            net.eval()
            for i, (images, labels) in enumerate(data_test_loader):
                images, labels = images.cuda(), labels.cuda()
                output = net(images)
                avg_loss += criterion(output, labels).sum()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

        avg_loss /= len(data_train_loader.dataset)
        acc = float(total_correct) / len(data_train_loader.dataset)
        if acc_best < acc:
            acc_best = acc
        print(f'Test Avg. Loss: {avg_loss:.6f}, Accuracy: {acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train-teacher-network')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'cifar10'])
    parser.add_argument('--data', type=str, default='data/')
    parser.add_argument('--output_dir', type=str, default='ckps/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--teacher_num', type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data_train_loader, data_test_loader = get_loader(args.data, args.dataset, args.batch_size)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for i in range(args.teacher_num):
        print(f'start train teacher-{i + 1}')
        if args.dataset == 'MNIST':
            epochs = 10
            net = LeNet5().cuda()
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        else:
            epochs = 200
            net = resnet.ResNet34().cuda()
            optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        train(net, epochs, optimizer, criterion, data_train_loader, data_test_loader)
        torch.save(net.state_dict(), args.output_dir + f'teacher{i + 1}')
