#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================
import csv
import datetime
import time
from copy import copy
from operator import itemgetter
import os
import shutil

import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap
import tqdm
from .util import Logger
from .model import *
from .torch_blocks import *
from .confusion_support import plot_confusion_support, avg_sim_confusion
import os.path
import pdb
from torchvision import transforms
from .PCA import *
from .sinkhorn_distance import SinkhornDistance
from .dataloader.FSCIL.data_utils import *

from .mixup_DA import *



def pretrain_baseFSCIL(verbose, **parameters):
    '''
    Pre-training on base session
    '''
    args = DotMap(parameters)

    writer = SummaryWriter(args.log_dir)

    # Initialize the dataset generator and the model
    args = set_up_datasets(args)
    trainset, train_loader, val_loader = get_base_dataloader(args)

    model = KeyValueNetwork(args)

    model.mode = 'pretrain'
    # Store all parameters in a variable
    parameters_list, parameters_table = process_dictionary(parameters)
    logs_dir = os.path.join(args.log_dir + '/' + 'train_log.txt')

    # Print all parameters
    if verbose:
        print("Parameters:")
        for key, value in parameters_list:
            print("\t{}".format(key).ljust(40) + "{}".format(value))
            with open(logs_dir, 'a', encoding='utf-8') as f1:
                f1.write("\t{}".format(key).ljust(40) + "{}".format(value)+'\n')


    criterion = nn.CrossEntropyLoss()


    if args.gpu is not None:
        t.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)

    for param in model.embedding.conv1.parameters():
        param.requires_grad = False

    optimizer = t.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.learning_rate,nesterov=args.SGDnesterov,
                            weight_decay=args.SGDweight_decay, momentum=args.SGDmomentum)


    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)


    best_acc1 = 0

    for epoch in tqdm.tqdm(range(1, args.max_train_iter), desc='Epoch'):
        global_count = 0
        losses = AverageMeter('Loss')
        acc = AverageMeter('Acc@1')
        model.train(True)


        for i, batch in enumerate(train_loader):
            global_count = global_count + 1
            data, train_label = [_.cuda(args.gpu, non_blocking=True) for _ in batch]
            # data, data_aug1, data_aug2, train_label = [_.cuda(args.gpu, non_blocking=True) for _ in batch]
            # forward pass
            optimizer.zero_grad()
            aug_p = torch.rand(1)


            if epoch <50:
                output = model(data)
                loss_cls = criterion(output, train_label)
                proxy = model.classifier
                features = model.fea_rep
                loss_pcl = PCLoss(num_classes=args.base_class, scale=12)(features, train_label, proxy)

            else:
                if aug_p < 0.1:
                    inputs_aug, targets_a_aug, targets_b_aug, lam = mixup_data(data, train_label, device_idx=args.gpu)
                    all_x = torch.cat([inputs_aug, data])
                    all_y = model(all_x)
                    loss_cls = lam * criterion(all_y[:data.size()[0]], targets_a_aug) + (1 - lam) * criterion(
                    all_y[:data.size()[0]], targets_b_aug) + criterion(all_y[data.size()[0]:], train_label)
                    proxy = model.classifier
                    features = model.fea_rep[data.size()[0]:]
                    loss_pcl = PCLoss(num_classes=args.base_class, scale=12)(features, train_label, proxy)

                else:
                    output = model(data)
                    loss_cls = criterion(output, train_label)
                    proxy = model.classifier
                    features = model.fea_rep
                    loss_pcl = PCLoss(num_classes=args.base_class, scale=12)(features, train_label, proxy)

            loss = loss_cls + args.pcl_weight * loss_pcl
            # Backpropagation
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), data.size(0))


        scheduler.step()

        # write to tensorboard
        writer.add_scalar('training_loss/pretrain_CEL', losses.avg, epoch)
        writer.add_scalar('accuracy/pretrain_train', acc.avg, epoch)

        val_loss, val_acc_mean, _ = validation(model, criterion, val_loader, args)
        writer.add_scalar('validation_loss/pretrain_CEL', val_loss, epoch)
        writer.add_scalar('accuracy/pretrain_val', val_acc_mean, epoch)

        is_best = val_acc_mean > best_acc1
        best_acc1 = max(val_acc_mean, best_acc1)

        if is_best and epoch > 80:
            model.eval()
            trainset_clean, train_loader_clean, val_loader_clean = get_clean_dataloader(args)
            prototype, cov, classlabel = model.protoSave(train_loader_clean)
        else:
            prototype, cov, classlabel = None, None, None

        with open(logs_dir, 'a', encoding='utf-8') as f1:
            f1.write(f'epoch: {epoch} current mean ac: {val_acc_mean} best acc: {best_acc1:0.5f} loss: {losses.avg}\n')
        print('epoch:', epoch, 'current mean acc', val_acc_mean, 'best acc:', best_acc1, 'loss', losses.avg)

        save_checkpoint({
            'train_iter': epoch + 1,
            'arch': args.block_architecture,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'prototype': prototype,
            'cov': cov,
            'classlabel': classlabel,

        }, is_best, savedir=args.log_dir)


    writer.close()








def train_FSCIL(verbose=False, **parameters):
    '''
    Main FSCIL evaluation on all sessions
    '''
    args = DotMap(parameters)
    args = set_up_datasets(args)
    # args.gpu = 6

    model = KeyValueNetwork(args, mode="pretrain")

    # Store all parameters in a variable
    parameters_list, parameters_table = process_dictionary(parameters)

    # Print all parameters
    if verbose:
        print("Parameters:")
        for key, value in parameters_list:
            print("\t{}".format(key).ljust(40) + "{}".format(value))

    # Write parameters to file
    if not args.inference_only:
        filename = args.log_dir + '/parameters.csv'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # retrain
        with open(filename, 'w') as csv_file:
            writer = csv.writer(csv_file)
            keys, values = zip(*parameters_list)
            writer.writerow(keys)
            writer.writerow(values)

    writer = SummaryWriter(args.log_dir)

    criterion = nn.CrossEntropyLoss()

    if args.gpu is not None:
        t.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)

    # set all parameters except FC to trainable false
    for param in model.parameters():
        param.requires_grad = False
    for param in model.embedding.fc.parameters():
        param.requires_grad = True

    model.classifier.requires_grad = False

    optimizer = t.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.learning_rate, nesterov=args.SGDnesterov,
                            weight_decay=args.SGDweight_decay, momentum=args.SGDmomentum)

    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)

    # model, optimizer, scheduler, start_train_iter, best_acc1= load_checkpoint(model, optimizer, scheduler, args)
    model, optimizer, scheduler, start_train_iter, best_acc1, prototype, cov, classlabel = load_checkpoint2(model,
                                                                                                            optimizer,
                                                                                                            scheduler,
                                                                                                            args)



    logs_dir = os.path.join(args.log_dir + '/' + 'test_log.txt')
    all_acc = []
    acc_novel = []
    prototype = model.classifier.data.cpu().numpy()
    for session in range(args.sessions):
        nways_session = args.base_class + session * args.way

        if session > 0:
            model.mode ='meta'

        train_set, train_loader, test_loader, test_loader_new = get_dataloader(args, session)


        # update model
        batch = next(iter(train_loader))


        proto_align_final_cub(model, batch, optimizer, args, writer, session, nways_session, prototype, cov)



        loss, acc, class_sample_count, class_correct_count = validation_for_incremental(model, criterion, test_loader,
                                                                                        args, nways_session)



        print("Session {:}: {:.2f}%".format(session, acc))
        all_acc.append(acc)
        writer.add_scalar('accuracy/cont', acc, session)

        acc_up2now=[]
        for i in range(session + 1):
            if i == 0:
                classes = np.arange(args.num_classes)[:args.base_class]
            else:
                classes = np.arange(args.num_classes)[(args.base_class + (i - 1) * args.way):(args.base_class + i * args.way)]
            Acc_Each_Session = caculate_session_acc(classes, class_sample_count, class_correct_count)
            acc_up2now.append(Acc_Each_Session)


        if session >0 :
            novel_classes_so_far = np.arange(args.num_classes)[args.base_class:(args.base_class + session * args.way)]
            Acc_All_Novel = caculate_session_acc(novel_classes_so_far, class_sample_count, class_correct_count)
            acc_novel.append(Acc_All_Novel)

        print(f'{acc_up2now}')
        print(f'Novel classes Avg Acc:{acc_novel}\n')


        if session == 0:
            with open(logs_dir, 'a', encoding='utf-8') as f1:
                f1.write(f'\nSource Model:{args.resume}\n')
                f1.write(f'{acc_up2now}\t{acc}\n')
        else:
            with open(logs_dir, 'a', encoding='utf-8') as f1:
                f1.write(f'{acc_up2now} Current Avg Acc:{acc} Novel classes Avg Acc:{acc_novel}\n')

        if session == args.sessions - 1:
            mean_acc = np.mean(all_acc)
            with open(logs_dir, 'a', encoding='utf-8') as f1:
                f1.write(f'Mean Acc for this run is: {mean_acc}\tEach Session Acc is{all_acc}\n')
                print((f'Mean Acc for this run is: {mean_acc}\nEach Session Acc is{all_acc}\n'))
    writer.close()




def proto_align_final_cub(model, data, optimizer, args, writer, session, nways_session, base_prototype, base_cov):

    losses = AverageMeter('Loss')
    criterion = myCosineLoss(args.retrain_act)

    dataset = myRetrainDataset(data[0], data[1])


    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size_training)
    sinkhorn = SinkhornDistance(eps=0.01, max_iter=200, args=args, reduction=None).cuda(args.gpu)

    sinkhorn_multi = SinkhornDistance(eps=0.01, max_iter=200, args=args, reduction=None).cuda(args.gpu)

    model.eval()
    with t.no_grad():
        for x, target in dataloader:
            x = x.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            model.update_feat_replay(x, target)


    feat, label = model.get_feat_replay()
    model.reset_prototypes(args)
    model.update_prototypes_feat(feat, label, nways_session)
    model.embedding.fc.train()


    if session > 0:
        nways_session = args.base_class + session *args.way
        oways_session = args.base_class + (session - 1) * args.way


        base_torch = torch.from_numpy(base_prototype).cuda(args.gpu)
        c_proto = model.key_mem.data
        cost, Pi, C = sinkhorn(base_torch, c_proto[oways_session:nways_session])

        c_proto = c_proto.cpu().numpy()

        for i in range(oways_session,nways_session):

            mean, cov = distribution_calibration(c_proto[i], Pi[:, i-oways_session], base_prototype, base_cov,
                                                     n_lsamples=args.way)

            proto_temp = np.random.multivariate_normal(mean=mean, cov=cov, size=args.sample_num)

            proto_temp3 = torch.from_numpy(proto_temp).float().cuda(args.gpu)

            cost2, Pi2, C2 = sinkhorn_multi(base_torch,proto_temp3)

            new_temp = torch.matmul(Pi[:, i-oways_session], torch.matmul(Pi2, proto_temp3))

            c_proto[i] = new_temp.cpu().numpy()

        c_proto = torch.from_numpy(c_proto).float().cuda(args.gpu)


        model.key_mem.data = c_proto

        model.nudge_prototypes(nways_session, writer, session, args.gpu)


        for epoch in range(args.retrain_iter):

            optimizer.zero_grad()
            support = model.get_support_feat(feat)
            loss = criterion(support[:nways_session], model.key_mem.data[:nways_session])
            # Backpropagation
            loss.backward()
            optimizer.step()
            writer.add_scalar('retraining/loss_sess{:}'.format(session), loss.item(), epoch)


    model.eval()
    model.reset_prototypes(args)
    model.update_prototypes_feat(feat, label, nways_session)








def distribution_calibration(prototype, probabi, base_means, base_cov, n_lsamples, alpha=0.21, lambd=0.3, k=10):
    # index = np.argsort(-probabi.numpy())
    dim = base_means[0].shape[0]
    calibrated_mean = 0
    calibrated_cov = 0

    probabi = probabi.cpu()

    proab_reshape = np.repeat(n_lsamples * probabi.numpy(), dim, axis=0).reshape(len(base_means), dim)
    calibrated_mean = (1 - lambd) * np.sum(proab_reshape * np.concatenate([base_means[:]]), axis=0) + lambd * prototype
    #
    proab_reshape_conv = np.repeat(n_lsamples * probabi.numpy(), dim * dim, axis=0).reshape(len(base_means), dim, dim)
    calibrated_cov = np.sum(proab_reshape_conv * np.concatenate([base_cov[:]]), axis=0) + alpha
    return calibrated_mean, calibrated_cov





def validation(model,criterion,dataloader, args,nways_session=None):
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')

    sim_conf = avg_sim_confusion(args.num_classes,nways_session)
    model.eval()
    with t.no_grad():
        for i, batch in enumerate(dataloader):
            data, label = [_.cuda(args.gpu,non_blocking=True) for _ in batch]

            output = model(data)
            loss = criterion(output,label)

            # print(output)
            # print(output[:5])
            accuracy = top1accuracy(output.argmax(dim=1),label)
            losses.update(loss.item(),data.size(0))
            acc.update(accuracy.item(),data.size(0))
            # if nways_session is not None:
            #     sim_conf.update(model.similarities.detach().cpu(),
            #                 F.one_hot(label.detach().cpu(), num_classes = args.num_classes).float())
    # Plot figure if needed
    fig = sim_conf.plot() if nways_session is (not None) else None
    return losses.avg, acc.avg, fig


def validation_for_incremental(model, criterion, dataloader, args, nways_session=None):
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')


    class_sample_count = {}
    class_correct_count = {}

    model.eval()
    with t.no_grad():
        for i, batch in enumerate(dataloader):
            data, label = [_.cuda(args.gpu, non_blocking=True) for _ in batch]

            output = model(data)
            loss = criterion(output, label)
            # accuracy = top1accuracy(output.argmax(dim=1), label)

            correct = output.argmax(dim=1).eq(label).float()
            accuracy = correct.sum(0).mul_(100.0 / label.size(0))

            losses.update(loss.item(), data.size(0))
            acc.update(accuracy.item(), data.size(0))


            for l in label:
                if l.item() in class_sample_count:
                    class_sample_count[l.item()] += 1
                else:
                    class_sample_count[l.item()] = 1

            for l, c in zip(label, correct):
                if l.item() in class_correct_count:
                    class_correct_count[l.item()] += c.item()
                else:
                    class_correct_count[l.item()] = c.item()

    return losses.avg, acc.avg, class_sample_count, class_correct_count





def caculate_session_acc(classes, class_sample_count, class_correct_count):


    test_data_num, correct_data_num = 0,0
    for itm in classes:
        test_data_num += class_sample_count[itm]
        correct_data_num += class_correct_count[itm]

    return (correct_data_num/test_data_num) * 100






def validation_onehot(model,criterion,dataloader, args, num_classes):
    #  

    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')

    model.eval()

    with t.no_grad(): 
        for i, batch in enumerate(dataloader):
            data, label = [_.cuda(args.gpu,non_blocking=True) for _ in batch]
            label = F.one_hot(label, num_classes = num_classes).float()

            output = model(data)
            loss = criterion(output,label)
            
            _, _, _, _, accuracy = process_result(
                output,label)

            losses.update(loss.item(),data.size(0))
            acc.update(accuracy.item()*100,data.size(0))
    
    return losses.avg, acc.avg

# --------------------------------------------------------------------------------------------------
# Interpretation
# --------------------------------------------------------------------------------------------------
def process_result(predictions, actual):
    predicted_labels = t.argmax(predictions, dim=1)
    actual_labels = t.argmax(actual, dim=1)

    accuracy = predicted_labels.eq(actual_labels).float().mean(0,keepdim=True)
    # TBD implement those uncertainties
    predicted_certainties =0#
    actual_certainties = 0 #
    return predicted_labels, predicted_certainties, actual_labels, actual_certainties, accuracy


def process_dictionary(dict):
    # Convert the dictionary to a sorted list
    dict_list = sorted(list(dict.items()))

    # Convert the dictionary into a table
    keys, values = zip(*dict_list)
    values = [repr(value) for value in values]
    dict_table = np.vstack((np.array(keys), np.array(values))).T

    return dict_list, dict_table

# --------------------------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------------------------
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar',savedir=''):
    t.save(state, savedir+'/'+filename)
    if is_best:
        shutil.copyfile(savedir+'/'+filename, savedir+'/'+'model_best.pth.tar')



def load_checkpoint(model,optimizer,scheduler,args):        

    # First priority: load checkpoint from log_dir 
    if os.path.isfile(args.log_dir+ '/checkpoint.pth.tar'):
        resume = args.log_dir+ '/checkpoint.pth.tar'
        print("=> loading checkpoint '{}'".format(resume))
        if args.gpu is None:
            checkpoint = t.load(resume)
        else:
            # Map model to be loaded to specified single args.gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = t.load(resume, map_location=loc)
        start_train_iter = int(checkpoint['train_iter'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])

        print("=> loaded checkpoint '{}' (train_iter {})"
              .format(args.log_dir, checkpoint['train_iter']))
        print('previous acc', best_acc1)
        prototype, cov, classlabel = None, None, None


    # Second priority: load from pretrained model
    # No scheduler and no optimizer loading here.  
    elif os.path.isfile(args.resume+'/model_best.pth.tar'):
        resume = args.resume+'/model_best.pth.tar'
        print("=> loading pretrain checkpoint '{}'".format(resume))
        if args.gpu is None:
            checkpoint = t.load(resume)
        else:
            # Map model to be loaded to specified single args.gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = t.load(resume, map_location=loc)
        start_train_iter = 0 
        best_acc1 = 0
        model.load_state_dict(checkpoint['state_dict'])
        best_acc2 = checkpoint['best_acc1']
        print('previous best acc',best_acc2)
        print("=> loaded pretrained checkpoint '{}' (train_iter {})"
              .format(args.log_dir, checkpoint['train_iter']))

        # prototype = checkpoint['prototype']
        # cov = checkpoint['cov']
        # classlabel = checkpoint['classlabel']


    else:
        start_train_iter=0
        best_acc1 = 0
        prototype, cov, classlabel = None, None, None
        print("=> no checkpoint found at '{}'".format(args.log_dir))
        print("=> no pretrain checkpoint found at '{}'".format(args.resume))




    return model, optimizer, scheduler, start_train_iter, best_acc1,
    # return model, optimizer, scheduler, start_train_iter, best_acc1, prototype,cov,classlabel




def load_checkpoint2(model,optimizer,scheduler,args):

    # First priority: load checkpoint from log_dir
    if os.path.isfile(args.log_dir+ '/checkpoint.pth.tar'):
        resume = args.log_dir+ '/checkpoint.pth.tar'
        print("=> loading checkpoint '{}'".format(resume))
        if args.gpu is None:
            checkpoint = t.load(resume)
        else:
            # Map model to be loaded to specified single args.gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = t.load(resume, map_location=loc)
        start_train_iter = int(checkpoint['train_iter'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])

        print("=> loaded checkpoint '{}' (train_iter {})"
              .format(args.log_dir, checkpoint['train_iter']))
        print('previous acc', best_acc1)
        prototype, cov, classlabel = None, None, None


    # Second priority: load from pretrained model
    # No scheduler and no optimizer loading here.
    elif os.path.isfile(args.resume+'/model_best.pth.tar'):
        resume = args.resume+'/model_best.pth.tar'
        print("=> loading pretrain checkpoint '{}'".format(resume))
        if args.gpu is None:
            checkpoint = t.load(resume)
        else:
            # Map model to be loaded to specified single args.gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = t.load(resume, map_location=loc)
        start_train_iter = 0
        best_acc1 = 0
        model.load_state_dict(checkpoint['state_dict'])
        best_acc2 = checkpoint['best_acc1']
        best_acc1 = best_acc2

        print('previous best acc',best_acc2)
        print("=> loaded pretrained checkpoint '{}' (train_iter {})"
              .format(args.log_dir, checkpoint['train_iter']))

        prototype = checkpoint['prototype']
        cov = checkpoint['cov']
        classlabel = checkpoint['classlabel']


    else:
        start_train_iter=0
        best_acc1 = 0
        prototype, cov, classlabel = None, None, None
        print("=> no checkpoint found at '{}'".format(args.log_dir))
        print("=> no pretrain checkpoint found at '{}'".format(args.resume))




    # return model, optimizer, scheduler, start_train_iter, best_acc1,
    return model, optimizer, scheduler, start_train_iter, best_acc1, prototype,cov,classlabel

# --------------------------------------------------------------------------------------------------
# Some Pytorch helper functions (might be removed from this file at some point)
# --------------------------------------------------------------------------------------------------





def convert_toonehot(label): 
    '''
    Converts index to one-hot. Removes rows with only zeros, such that 
    the tensor has shape (B,num_ways)
    '''
    label_onehot = F.one_hot(label)
    label_onehot = label_onehot[:,label_onehot.sum(dim=0)!=0]
    return label_onehot.type(t.FloatTensor)

def top1accuracy(pred, target):
    """Computes the precision@1"""
    batch_size = target.size(0)

    correct = pred.eq(target).float().sum(0)
    return correct.mul_(100.0 / batch_size)



def accuracy_for_each_task(pred, target, ):
    """Computes the precision@1"""
    batch_size = target.size(0)

    correct = pred.eq(target).float().sum(0)
    return correct.mul_(100.0 / batch_size)



class myRetrainDataset(Dataset):
    def __init__(self, x,y):
        self.x = x
        self.y = y
       
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

