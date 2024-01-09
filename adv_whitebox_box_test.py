from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import utils
from models.resnet import ResNet18
from datetime import datetime
import aapm

import os
from art.classifiers.pytorch import PyTorchClassifier
from art.attacks import ProjectedGradientDescent

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Neural Network Training')
parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'Fashionmnist'])
parser.add_argument('--dataset-root', type=str, default='./data', help='Dataset root')
parser.add_argument('--batch-size','-b', default=64, type=int, help='batch size')
parser.add_argument('-j', '--num-workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--save-dir', type=str, default='adv_log', help='save log and model')
parser.add_argument('--gpu-ids', type=str, default='0', help='gpu id eg. 0 or  0,1')
parser.add_argument('--exp-name', type=str, default='test', help='experiment-name')
parser.add_argument('--arc', type=str, default='resnet18', help='model')
parser.add_argument('--start', type=int, default=1, help='start')
parser.add_argument('--times', type=int, default=5, help='times')
parser.add_argument('--num-classes', type=int, default=10, help='num of datatset classes')
parser.add_argument('--DOWNLOAD-DATASET', '-d', action='store_true', help='Download datatset')
parser.add_argument('--RGB-Image', action='store_true', help='RGB image data')
parser.add_argument('--drop-type', type=str, default='ACAD', help='None or ACAD')
parser.add_argument('--method', type=str, default='ACAD-CIFAR100.pth', help='evaluated model')
parser.add_argument('--adv', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'Fashionmnist'])


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["NUMEXPR_MAX_THREADS"] = '16'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
use_cuda = True

path = os.path.join(opt.save_dir, opt.dataset)
if not os.path.isdir(path):
    os.mkdir(path)



if 'CIFAR10' == opt.dataset:
    opt.num_classes = 10
    opt.image_size = 32
    opt.RGB_Image = True
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).cuda()

    test_data = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=opt.DOWNLOAD_DATASET, transform=transform_test)
    clip_values = (-2.429, 2.754)
    input_shape = (100, 3, 32, 32)

elif 'CIFAR100' == opt.dataset:
    opt.num_classes = 100
    opt.image_size = 32
    opt.RGB_Image = True
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])

    mean = torch.tensor([0.5070751592371323, 0.48654887331495095, 0.4409178433670343]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.2673342858792401, 0.2564384629170883, 0.27615047132568404]).view(1, 3, 1, 1).cuda()

    test_data = torchvision.datasets.CIFAR100(
        root=data_path, train=False, download=opt.DOWNLOAD_DATASET, transform=transform_test)
    clip_values = (-1.903, 2.026)
    input_shape = (100, 3, 32, 32)


elif 'Fashionmnist' == opt.dataset:
    opt.num_classes = 10
    opt.image_size = 28
    opt.RGB_Image = False
    transform_data = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_data = torchvision.datasets.FashionMNIST(
        root=data_path, train=False, transform=transform_data, download=opt.DOWNLOAD_DATASET)
    clip_values = (0, 1)
    input_shape = (100, 1, 28, 28)
else:
    raise Exception("Unknown Dataset！")

print("test_dataset:", opt.dataset)
print("test_data:", len(test_data))
testloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size, shuffle=False,
                                         num_workers=opt.num_workers)

device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
results_log_csv_name ='log.cv'

def test(model, device, test_loader, eps, method, e_num, opt):
    pretrained_model = 'result/'+ method[:-4] +'/result'+str(e_num)+'/' + method
    # Load the pretrained model
    model.load_state_dict((torch.load(pretrained_model))['state_dict'])
    acc = (torch.load(pretrained_model))['acc']
    print('The accuracy of ' + method + ': %.3f%%' % (acc))
    model.eval()

    correct = 0
    total = 0
    total_sum = 0
    common_id = []
    acc1 = 0.0
    acc2 = 0.0

    pytorch_classifier = PyTorchClassifier(clip_values=clip_values, model=model, loss=criterion, optimizer=None, input_shape=input_shape, nb_classes=opt.num_classes)
    if opt.adv == 'FGSM':
        adv_crafter = FastGradientMethod(pytorch_classifier, norm=np.inf, eps=eps, eps_step=eps, targeted=False,
                                         num_random_init=0, batch_size=100)
    elif opt.adv == 'PGD':
        adv_crafter = ProjectedGradientDescent(pytorch_classifier, norm=np.inf, eps=eps, eps_step=eps / 10,
                                               max_iter=10, targeted=False, num_random_init=1, batch_size=100)
    elif opt.adv == 'NewtonFool':
        adv_crafter = NewtonFool(pytorch_classifier, max_iter=5, batch_size=100)
    elif opt.adv == 'JSMA':
        adv_crafter = SaliencyMapMethod(SM_classifier,theta=0.1, gamma=0.1,batch_size=100)

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = pytorch_classifier.predict(data.cpu().numpy(), batch_size=100)
        output = torch.tensor(output)
        output = output.to(device)
        init_pred = output.max(1, keepdim=False)[1]

        common_id = np.where(init_pred.cpu() == target.cpu())[0]
        x_test_adv = adv_crafter.generate(x=data.cpu().numpy())
        perturbed_output = pytorch_classifier.predict(x_test_adv.cpu().numpy(), batch_size=100)
        perturbed_output = torch.tensor(perturbed_output)
        perturbed_output = perturbed_output.to(device)
        final_pred = perturbed_output.max(1, keepdim=False)[1]
        total += len(common_id)
        total_sum += target.size(0)
        correct += final_pred[common_id].eq(target[common_id].data).cpu().sum()

        acc1 =  float(correct) / total
        acc2 =  float(correct) / total_sum
        utils.progress_bar(batch_idx, len(test_loader), 'Acc('+opt.adv+'_' + str(eps) + '): %.3f%% (%d/%d) %.3f%%' %
                           (100. * acc1, correct, total, 100. * acc2))
    with open(os.path.join(path, results_log_csv_name), 'a') as f:
        f.write('%0.3f,%0.3f,%0.3f,%0.3f,%s,\n' % (eps, acc, 100. * acc1, 100. * acc2, datetime.now().strftime('%b%d-%H:%M:%S')))
    return correct


for j in range(1,5):

    pathstring='adv_log/new_CIFAR10-acad'+str(j)+'/'
    path = os.path.join(pathstring)
    if not os.path.isdir(path):
        os.mkdir(path)
    methods = opt.
    print('test model: ' + method)
    epsilons = [8/255]
    results_log_csv_name = method+'log.csv'
    model = ResNet18(num_classes=10, RGB_Image=True, drop='ACAD')
    criterion = nn.CrossEntropyLoss()
    with open(os.path.join(path, results_log_csv_name), 'w') as f:
        f.write(' epsilons,acc_ori,acc1 , acc2, time\n')
    for eps in epsilons:
        test(model, device, test_loader, eps, method, j, opt)