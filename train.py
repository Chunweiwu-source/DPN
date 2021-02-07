from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import model_zoo

import data_loader
from model import DPN
import objective
from lr_schedule import inv_lr_scheduler

import math
import numpy as np

import argparse
from IPython import embed
import tqdm


# Training settings
parser = argparse.ArgumentParser(description='PyTorch DPN')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 500)')

# optimization
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--log-interval', type=int, default=10, metavar='log',
                    help='how many batches to wait before logging training status')
parser.add_argument('--l2_decay', type=float, default=5e-4,
                    help='the L2  weight decay')
parser.add_argument('--diff_lr', type=bool, default=True,
                    help='the fc layer and the sharenet have different or same learning rate')

# model dataset
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')                   
parser.add_argument('--root_path', type=str, default="./OfficeHomeDataset/",
                    help='the path to load the data')
parser.add_argument('--source_dir', type=str, default="Art",
                    help='the name of the source dir')
parser.add_argument('--test_dir', type=str, default="Product",
                    help='the name of the test dir')
parser.add_argument('--num_class', default=65, type=int,
                    help='the number of classes')

# method
parser.add_argument('--z', type=float, default=8,
                    help='number of pseudo label samples on each class')

# temperature
parser.add_argument('--temp', type=float, default=0.05,
                    help='temperature for classifier') 

# other setting                    
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--seed', type=int, default=233, metavar='S',
                    help='random seed (default: 1)')               


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}


# Load data
def load_data():
    source_train_loader = data_loader.load_training(args.root_path, args.source_dir, args.batch_size, kwargs)
    
    target_train_loader = data_loader.load_training(args.root_path, args.test_dir, args.batch_size, kwargs)
    target_test_loader  = data_loader.load_testing(args.root_path, args.test_dir, args.batch_size, kwargs)
    
    return source_train_loader, target_train_loader, target_test_loader

# Print learning rate
def print_learning_rate(optimizer):
    for p in optimizer.param_groups:
        outputs = ''
        for k, v in p.items():
            if k == 'params':
                outputs += (k + ': ' + str(v[0].shape).ljust(30) + ' ')
            else:
                outputs += (k + ': ' + str(v).ljust(10) + ' ')
        print(outputs)

###################################

# For every epoch training
def train(epoch, model, source_loader, target_loader, optimizer, memory_bank, Invariance_criterion):
    
    # Set loss
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    # Print learning rate
    print_learning_rate(optimizer)

    len_dataloader = len(source_loader)
    bsz = args.batch_size

    len_train_target = len(target_loader) - 1
    
    # Set domain label
    dlabel_src = Variable(torch.ones(bsz).long().to(DEVICE))
    dlabel_tgt = Variable(torch.zeros(bsz).long().to(DEVICE))
   
    # Train
    for batch_idx, (_, source_data, source_label) in tqdm.tqdm(enumerate(source_loader),
                                    total=len_dataloader,
                                    desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):
        model.train()       
        
        # the parameter for reversing gradients
        p = float(batch_idx+1 + epoch * len_dataloader) / args.epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        
        # for the source domain batch
        source_data, source_label = Variable(source_data).to(DEVICE), Variable(source_label).to(DEVICE)
        
        _, clabel_src, dlabel_pred_src = model(source_data, alpha = alpha)
        label_loss = loss_class(clabel_src, source_label)
        domain_loss_src = loss_domain(dlabel_pred_src, dlabel_src)
        
        
        # for the target domain batch
        if batch_idx % len_train_target == 0:
            iter_target = iter(target_loader)

        target_index, target_data, target_label = iter_target.next()
        target_data, target_label = Variable(target_data).to(DEVICE), Variable(target_label).to(DEVICE)
        
        feat_tgt, clabel_tgt, dlabel_pred_tgt = model(target_data, alpha = alpha)
        domain_loss_tgt = loss_domain(dlabel_pred_tgt, dlabel_tgt)
        
        
        # EM loss
        clabel_tgt = F.softmax(clabel_tgt, dim=1)
        em_loss_tgt = objective.entropy_loss(clabel_tgt)
              
        
        # Domain Loss
        domain_loss_total = domain_loss_src + domain_loss_tgt
        
        
        # Invariance Criterion
        prototypes = model.class_classifier.weight.data
        L_inv = Invariance_criterion(prototypes, memory_bank)

        # Overall Loss Function
        join_loss = domain_loss_total + 100 * alpha * L_inv + .05 * em_loss_tgt
        loss_total = label_loss + join_loss
        
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        # Update memory bank
        with torch.no_grad():
            memory_bank.update_points(feat_tgt.detach(), target_index)

        if batch_idx % args.log_interval == 0:
            print('\nLoss: {:.6f},  label_Loss: {:.6f},  join_Loss: {:.6f}, domain_loss:{:.6f}, em_loss_tgt:{:.6f}, L_inv:{:.6f}'.format(
                loss_total.item(), label_loss.item(), join_loss.item(), domain_loss_total.item(), em_loss_tgt, L_inv.item()))


# For every epoch evaluation
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    for data, target in test_loader:
        data, target = Variable(data).to(DEVICE), Variable(target).to(DEVICE)
        _, s_output, _ = model(data)
            
        test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target, reduction='sum').item()
        pred = s_output.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        args.test_dir, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return correct


if __name__ == '__main__':

    model = DPN.DPN(num_classes=args.num_class, base_net='ResNet50', temperature=args.temp).to(DEVICE)
    train_loader, unsuptrain_loader, test_loader = load_data()
    correct = 0
    s_max_correct = 0

    # Set optimizer
    if args.diff_lr:
        optimizer = torch.optim.SGD([
            {'params': model.feature.parameters(), "lr_mult":1, 'decay_mult':2},
            {'params': model.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2},
            {'params': model.domain_classifier.parameters(), "lr_mult":1, 'decay_mult':2},
            {'params': model.class_classifier.parameters(), "lr_mult":10, 'decay_mult':2},
        ], lr=args.lr, momentum=args.momentum, weight_decay=args.l2_decay, nesterov=True)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.l2_decay, nesterov=True)
        
    # Create Memory Bank
    memory_bank = objective.MemoryBank(len(unsuptrain_loader.dataset), device=DEVICE)
    memory_bank.random_init_bank()
    
    # Create Invariance Criterion
    Invariance_criterion = objective.InvariancePropagationLoss(temperature=args.temp, z=args.z)
    
    # Training
    for epoch in range(1, args.epochs + 1):
        optimizer = inv_lr_scheduler(optimizer, epoch, args.epochs, args.lr)
        train(epoch, model, train_loader, unsuptrain_loader, optimizer, memory_bank, Invariance_criterion)
        
        # Evaluation
        t_correct = test(model, test_loader)
        if t_correct > correct:
            correct = t_correct

        print("%s max correct:" % args.test_dir, correct)
        print(args.source_dir, "to", args.test_dir)
    