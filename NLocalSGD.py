import os
import numpy as np
import time
import argparse
import sys

from math import ceil
from random import Random

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

from synthetic import partition_dataset, Synthetic_Dataset
import FedProxSGD as FedProx
import util_v4 as util
from comm_helpers import SyncAllreduce, unbalanced_SyncAllreduce, NormalSGDALLreduce, FedProx_SyncAllreduce

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(description='CIFAR-10 baseline')
parser.add_argument('--name','-n', 
                    default="default", 
                    type=str, 
                    help='experiment name, used for saving results')
parser.add_argument('--backend',
                    default="nccl",
                    type=str,
                    help='experiment name, used for saving results')
parser.add_argument('--dataset', 
                    default="cifar10", 
                    type=str, 
                    help='dataset name')
parser.add_argument('--model', 
                    default="res", 
                    type=str, 
                    help='neural network model')
parser.add_argument('--alpha', 
                    default=0, 
                    type=float, 
                    help='alpha')
parser.add_argument('--beta', 
                    default=0, 
                    type=float, 
                    help='beta')
parser.add_argument('--slowRatio', 
                    default=0, 
                    type=float, 
                    help='slowRatio')
parser.add_argument('--gmf', 
                    default=0, 
                    type=float, 
                    help='global momentum factor')
parser.add_argument('--lr', 
                    default=0.1, 
                    type=float, 
                    help='learning rate')
parser.add_argument('--bs', 
                    default=128, 
                    type=int, 
                    help='batch size on each worker')
parser.add_argument('--cp', nargs='+',
                    default=None, 
                    type=int, 
                    help='communication period')
parser.add_argument('--globalCp',
                    default=2, 
                    type=float, 
                    help='global communication period')
parser.add_argument('--mu',
                    default=0, 
                    type=float, 
                    help='mu')
parser.add_argument('--cr', 
                    default=4000, 
                    type=int, 
                    help='communication round')
# parser.add_argument('--print_freq', 
#                     default=100, 
#                     type=int, 
#                     help='print info frequency')
parser.add_argument('--save_freq', 
                    default=10, 
                    type=int, 
                    help='save info frequency')
# parser.add_argument('--rank', 
#                     default=0, 
#                     type=int, 
#                     help='the rank of worker')
parser.add_argument('--size', 
                    default=8, 
                    type=int, 
                    help='number of workers')
parser.add_argument('--total_size', 
                    default=100, 
                    type=int, 
                    help='number of total workers')
parser.add_argument('--seed', 
                    default=1, 
                    type=int, 
                    help='random seed')
parser.add_argument('--save', '-s', 
                    action='store_true', 
                    help='whether save the training results')
parser.add_argument('--all_reduce',
                    action='store_true', 
                    help='whether use AR-SGD')
parser.add_argument('--warmup', default='False', type=str,
                    help='whether to warmup learning rate for first 5 epochs')
parser.add_argument('--p', '-p', 
                    action='store_true', 
                    help='whether the dataset is partitioned or not')
parser.add_argument('--NIID',
                    action='store_true',
                    help='whether the dataset is partitioned or not')
parser.add_argument('--Unbalanced',
                    action='store_true',
                    help='whether to use Dirichlet distribution or not')
parser.add_argument('--NSGD',
                    action='store_true',
                    help='whether to use NSGD or LocalSGD')
parser.add_argument('--FedProx',
                    action='store_true',
                    help='whether to use NSGD or LocalSGD')
parser.add_argument('--constant_cp',
                    action='store_true',
                    help='whether to use NSGD or LocalSGD')
parser.add_argument('--persistent',
                    action='store_true',
                    help='whether to use NSGD or LocalSGD')

args = parser.parse_args()

print(args)


def run(size):
    models = []
    anchor_models = []
    optimizers = []
    ratios = []
    iters = []
    cps = args.cp
    save_names = []
    loss_Meters = []
    top1_Meters = []
    best_test_accs = []

    if args.constant_cp:
        cps = args.cp * args.size
    elif args.persistent:
        cps = [5,5,5,5,5,5,5,20,20,20]
    else:
        local_cps = args.cp * np.ones(size, dtype=int)
        num_slow_nodes = int(size * args.slowRatio)
        np.random.seed(2020)
        random_cps = 5 + np.random.randn(num_slow_nodes) * 2
        for i in range(len(random_cps)):
            random_cps[i] = round(random_cps[i])
        local_cps[:num_slow_nodes] = random_cps
        # local_iterations = local_cps[rank]
        cps = local_cps

    for rank in range(args.size):
    # initiate experiments folder
        save_path = 'new_results/'
        folder_name = save_path+args.name
        if rank == 0 and os.path.isdir(folder_name)==False and args.save:
            os.mkdir(folder_name)
        # initiate log files
        tag = '{}/lr{:.3f}_bs{:d}_cr{:d}_avgcp{:.3f}_e{}_r{}_n{}.csv'
        saveFileName = tag.format(folder_name, args.lr, args.bs, args.cr, 
                                  np.mean(args.cp), args.seed, rank, size)
        args.out_fname = saveFileName
        save_names.append(saveFileName)
        with open(args.out_fname, 'w+') as f:
            print(
                'BEGIN-TRAINING\n'
                'World-Size,{ws}\n'
                'Batch-Size,{bs}\n'
                'itr,'
                'Loss,avg:Loss,Prec@1,avg:Prec@1,val'.format(
                    ws=args.size,
                    bs=args.bs),
                file=f)

        globalCp = args.globalCp
        total_size = args.total_size

        # seed for reproducibility
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

        # load datasets
        train_loader, test_loader, dataRatio, x, y = partition_dataset(rank, total_size, 1, args.alpha, args.beta, args)
        ratios.append(dataRatio)
        print(sum([len(i) for i in x]))
        data_iter = iter(train_loader)
        iters.append(data_iter)

        # define neural nets model, criterion, and optimizer
        model = util.select_model(args.model, args)
        anchor_model = util.select_model(args.model, args)

        models.append(model)
        anchor_models.append(anchor_model)

        criterion = nn.CrossEntropyLoss()
        if args.FedProx:
            optimizer = FedProx.FedProxSGD(model.parameters(),
                              lr=args.lr,
                              momentum=0,
                              nesterov = False,
                              weight_decay=1e-4)
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=args.lr,
                                  momentum=0,
                                  nesterov = False,
                                  weight_decay=1e-4)
        optimizers.append(optimizer)

        
        batch_idx = 0
        best_test_accuracy = 0
        best_test_accs.append(best_test_accuracy)

        losses = util.Meter(ptag='Loss')
        top1 = util.Meter(ptag='Prec@1')
        loss_Meters.append(losses)
        top1_Meters.append(top1)

        model.train()
        tic = time.time()
        print(dataRatio, len(train_loader), len(test_loader))

    round_communicated = 0
    while round_communicated < args.cr:
        for rank in range(args.size):
            model = models[rank]
            anchor_model = anchor_models[rank]
            data_iter = iters[rank]
            optimizer = optimizers[rank]
            losses = loss_Meters[rank]
            top1 = top1_Meters[rank]

            for cp in range(cps[rank]):
                try:
                    data, target = data_iter.next()
                except StopIteration:
                    data_iter = iter(train_loader)
                    data, target = data_iter.next()

                # data loading
                data = data
                target = target

                # forward pass
                output = model(data)
                loss = criterion(output, target)

                # backward pass
                loss.backward()
                if args.FedProx:
                    optimizer.step(anchor_model, args.mu)
                else:
                    optimizer.step()
                optimizer.zero_grad()

                train_acc = util.comp_accuracy(output, target)
                losses.update(loss.item(), data.size(0))
                top1.update(train_acc[0].item(), data.size(0))

                # batch_idx += 1
            # change the worker
            train_loader, dataRatio = get_next_trainloader(round_communicated, x, y, rank, args)
            data_iter = iter(train_loader)
            iters[rank] = data_iter
            ratios[rank] = dataRatio

        if args.NSGD:
            NormalSGDALLreduce(models, anchor_models, cps, globalCp, ratios)
        elif args.FedProx:
            FedProx_SyncAllreduce(models, ratios, anchor_models)
        else:
            unbalanced_SyncAllreduce(models, ratios)
        round_communicated += 1
            # update_lr(optimizer, round_communicated)


        if round_communicated % 4 == 0:
            for rank in range(args.size):
                name = save_names[rank]
                losses = loss_Meters[rank]
                top1 = top1_Meters[rank]

                with open(name, '+a') as f:
                    print('{itr},'
                          '{loss.val:.4f},{loss.avg:.4f},'
                          '{top1.val:.3f},{top1.avg:.3f},-1'
                          .format(itr=round_communicated,
                                  loss=losses, top1=top1), file=f)

        if round_communicated % 12 == 0:
            for rank in range(args.size):
                name = save_names[rank]
                model = models[rank]
                losses = loss_Meters[rank]
                top1 = top1_Meters[rank]
                name = save_names[rank]

                test_acc, global_loss = evaluate(model, test_loader, criterion)
                
                if test_acc > best_test_accs[rank]:
                    best_test_accs[rank]= test_acc

                print('itr {}, '
                      'rank {}, loss value {:.4f}, '
                      'train accuracy {:.3f}, test accuracy {:.3f}, '
                      'elasped time {:.3f}'.format(
                        round_communicated, rank, losses.avg, top1.avg, test_acc, time.time()-tic))

                with open(name, '+a') as f:
                    print('{itr},{filler},{filler},'
                          '{filler},{loss:.4f},'
                          '{val:.4f}'
                          .format(itr=-1,
                                  filler=-1, loss=global_loss, val=test_acc), 
                          file=f)

                losses.reset()
                top1.reset()
                tic = time.time()
                # return

    for rank in range(args.size):
        name = save_names[rank]
        with open(name, '+a') as f:
            print('{itr} best test accuracy: {val:.4f}'
                  .format(itr=-2, val=best_test_accs[rank]), 
                  file=f)



def evaluate(model, test_loader, criterion):
    model.eval()
    top1 = util.AverageMeter()
    losses = util.AverageMeter()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            outputs = model(data)
            loss = criterion(outputs, target)
            acc1 = util.comp_accuracy(outputs, target)
            top1.update(acc1[0].item(), data.size(0))
            losses.update(loss.item(), data.size(0))

    model.train()

    return top1.avg, losses.avg

def get_next_trainloader(seed, x, y, rank,args):
    np.random.seed(seed)
    a = np.arange(args.total_size)
    np.random.shuffle(a)
    current_workers = a[:args.size]
    current_x = [x[i] for i in current_workers]
    worker_idx = current_workers[rank]

    sub_x, sub_y = x[worker_idx], y[worker_idx]
    myDataset = Synthetic_Dataset(sub_x, sub_y)
    train_loader = DataLoader(myDataset,
                      batch_size=args.bs,
                      shuffle=True,
                      pin_memory=True)

    ratio = len(sub_x) / sum([len(i) for i in current_x])

    return train_loader, ratio

def update_learning_rate(optimizer, epoch, itr=None, itr_per_epoch=None,
                         scale=1):
    """
    1) Linearly warmup to reference learning rate (5 epochs)
    2) Decay learning rate exponentially (epochs 30, 60, 80)
    ** note: args.lr is the reference learning rate from which to scale up
    ** note: minimum global batch-size is 256
    """
    target_lr = args.lr * args.bs * scale * args.size / 128

    lr = None
    if args.warmup and epoch < 5:  # warmup to scaled lr
        if target_lr <= args.lr:
            lr = target_lr
        else:
            assert itr is not None and itr_per_epoch is not None
            count = epoch * itr_per_epoch + itr + 1
            incr = (target_lr - args.lr) * (count / (5 * itr_per_epoch))
            lr = args.lr + incr
    else:
        lr = target_lr
        for e in args.lr_schedule:
            if epoch >= e:
                lr *= args.lr_schedule[e]

    if lr is not None:
        # print('Updating learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def update_lr(optimizer, iter_id):
    if iter_id == int(args.cr/2):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

    if iter_id == int(args.cr*0.75):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def init_weights(epoch, train_loader, model, criterion, optimizer):
    model.train()
    for e in range(epoch):
        for data, target in train_loader:

            # data loading
            data = data.cuda(non_blocking = True)
            target = target.cuda(non_blocking = True)

            # forward pass
            output = model(data)
            loss = criterion(output, target)

            # backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def init_processes(rank, size, fn):
    """ Initialize the distributed environment. """
    # dist.init_process_group(backend=args.backend, 
    #                         init_method='tcp://h0:22000', 
    #                         rank=rank, 
    #                         world_size=size)
    fn(size)

if __name__ == "__main__":
    # rank = args.rank
    size = args.size
    # print(rank)
    init_processes(-1, size, run)


