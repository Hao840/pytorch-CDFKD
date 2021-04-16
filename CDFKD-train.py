'''
reference: https://github.com/huawei-noah/Data-Efficient-Model-Compression/tree/master/DAFL
'''
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

from models import *
from utils import get_loader


def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) / y.shape[0]
    return l_kl


def generator_loss(outputs_T, features_T):
    pred = outputs_T.data.max(1)[1]
    loss_activation = -features_T.abs().mean()
    loss_one_hot = criterion(outputs_T, pred)
    softmax_o_T = F.softmax(outputs_T, dim=1).mean(dim=0)
    loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
    loss = loss_one_hot * args.oh + loss_information_entropy * args.ie + loss_activation * args.a
    return loss_activation, loss_one_hot, loss_information_entropy, loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'cifar10'])
    parser.add_argument('--data', type=str, default='data/')
    parser.add_argument('--teacher_dir', type=str, default='ckps/')
    parser.add_argument('--teacher_num', type=int, default=3)
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
    parser.add_argument('--lr_G', type=float, default=0.2, help='learning rate')
    parser.add_argument('--lr_S', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--oh', type=float, default=1, help='one hot loss')
    parser.add_argument('--ie', type=float, default=5, help='information entropy loss')
    parser.add_argument('--a', type=float, default=0.1, help='activation loss')
    parser.add_argument('--output_dir', type=str, default='ckps/')

    args = parser.parse_args()

    generator = Generator(args).cuda()
    generator = nn.DataParallel(generator)

    _, data_test_loader = get_loader(args.data, args.dataset, args.batch_size)

    teachers = []
    for i in range(args.teacher_num):
        if args.dataset == 'MNIST':
            teacher = lenet.LeNet5().cuda()
        elif args.dataset == 'cifar10':
            teacher = resnet.ResNet34().cuda()
        else:
            raise NotImplementedError
        teacher.load_state_dict(torch.load(args.teacher_dir + f'teacher{i + 1}'))
        teacher.eval()
        teacher = nn.DataParallel(teacher)
        teachers.append(teacher)

    if args.dataset == 'MNIST':
        net = lenet.MHLeNet5Half(args.teacher_num).cuda()
        net = nn.DataParallel(net)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G)
        optimizer_S = torch.optim.Adam(net.parameters(), lr=args.lr_S)
    elif args.dataset == 'cifar10':
        net = resnet.MultiHeaderResNet18(args.teacher_num).cuda()
        net = nn.DataParallel(net)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G)
        optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)
    else:
        raise NotImplementedError

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer_S, [400, 600], 0.1)

    accr_best = 0
    for epoch in range(args.n_epochs):
        net.train()
        for i in range(120):
            optimizer_S.zero_grad()
            optimizer_G.zero_grad()

            z = torch.randn(args.batch_size, args.latent_dim).cuda()
            gen_imgs = generator(z)

            teacher_outputs = [teacher(gen_imgs, out_feature=True) for teacher in teachers]
            outputs_Ts = [out[0] for out in teacher_outputs]
            features_Ts = [out[1] for out in teacher_outputs]

            loss_Gs = [generator_loss(outputs_Ts[j], features_Ts[j]) for j in range(3)]  # 0: a, 1: oh, 2: ie, 3: sum
            loss_G = torch.mean(torch.stack([loss[3] for loss in loss_Gs]))
            loss_G.backward()

            outputs_S = net(gen_imgs.detach())
            loss_kd = [kdloss(outputs_S[j], outputs_Ts[j].detach()) for j in range(3)]
            loss_headers = torch.mean(torch.stack(loss_kd))
            outputs_T = torch.mean(torch.stack(outputs_Ts, dim=0), dim=0)
            loss_ensemble = kdloss(torch.mean(torch.stack(outputs_S), dim=0), outputs_T.detach())
            (loss_headers + loss_ensemble).backward()

            optimizer_G.step()
            optimizer_S.step()

            scheduler.step()

            if i == 1:
                print(f"[Epoch {epoch}/{args.n_epochs}] "
                      f"[loss_oh: {sum([loss_Gs[i][1] for i in range(3)]) / 3:.4f}] "
                      f"[loss_ie: {sum([loss_Gs[i][2] for i in range(3)]) / 3:.4f}] "
                      f"[loss_a: {sum([loss_Gs[i][0] for i in range(3)]) / 3:.4f}] "
                      f"[loss_headers: {loss_headers:.4f}] "
                      f"[loss_ensemble: {loss_ensemble:.4f}]")

        total_correct = 0
        avg_loss = 0
        with torch.no_grad():
            net.eval()

            for i, (images, labels) in enumerate(data_test_loader):
                images = images.cuda()
                labels = labels.cuda()

                output = torch.stack(net(images))
                output = torch.mean(output, dim=0)

                avg_loss += criterion(output, labels).sum()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
        avg_loss /= len(data_test_loader.dataset)
        accr = round(float(total_correct) / len(data_test_loader.dataset), 4)

        print(f'Test Avg. Loss: {avg_loss:.4f}, '
              f'Accuracy: {accr}')

        if accr > accr_best:
            accr_best = accr
            torch.save(net.state_dict(), args.output_dir + f'student')
