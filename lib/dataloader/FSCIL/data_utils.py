#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import numpy as np
import torch
from .sampler import CategoriesSampler
from .augmix import *


def set_up_datasets(args):
    if args.dataset == 'mini_imagenet':
        import lib.dataloader.FSCIL.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9
        import lib.dataloader.FSCIL.augmentations as augmentations
        augmentations.IMAGE_SIZE = 84


    elif args.dataset == 'cifar100':
        import lib.dataloader.FSCIL.cifar100.cifar100 as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9
        import lib.dataloader.FSCIL.augmentations as augmentations
        augmentations.IMAGE_SIZE = 32

    elif args.dataset == 'cub200':
        import lib.dataloader.FSCIL.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11

        import lib.dataloader.FSCIL.augmentations as augmentations
        augmentations.IMAGE_SIZE = 224
    args.Dataset = Dataset
    return args


def get_dataloader(args, session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader_meta(args, do_augment=False)

        testloader_new = None

    else:
        trainset, trainloader, testloader, testloader_new = get_new_dataloader(args, session, do_augment=False)

    return trainset, trainloader, testloader, testloader_new


def get_base_dataloader(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)

    if args.dataset == 'cifar100':
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])

        trainset = args.Dataset.CIFAR100(root=args.data_folder, train=True, download=True,
                                         index=class_index, base_sess=True, mode='pretrain')

        testset = args.Dataset.CIFAR100(root=args.data_folder, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':

        preprocess = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        trainset = args.Dataset.CUB200(root=args.data_folder, train=True,
                                       index=class_index, base_sess=True, crop_transform=None, secondary_transform=None,
                                       mode='pretrain')

        testset = args.Dataset.CUB200(root=args.data_folder, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        preprocess = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        trainset = args.Dataset.MiniImageNet(root=args.data_folder, train=True,
                                             index=class_index, base_sess=True, mode='pretrain')

        testset = args.Dataset.MiniImageNet(root=args.data_folder, train=False, index=class_index)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_training, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.batch_size_inference, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader


def get_clean_dataloader(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(0, args.base_class)

    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.data_folder, train=True, download=False,
                                         index=class_index, base_sess=True, mode='None')

        testset = args.Dataset.CIFAR100(root=args.data_folder, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        class_index = np.arange(0, args.base_class)
        trainset = args.Dataset.CUB200(root=args.data_folder, train=True,
                                       index=class_index, base_sess=True, mode='None')
        testset = args.Dataset.CUB200(root=args.data_folder, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.data_folder, train=True,
                                             index=class_index, base_sess=True, mode='None')

        testset = args.Dataset.MiniImageNet(root=args.data_folder, train=False, index=class_index)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_training, shuffle=False,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.batch_size_inference, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader




def get_new_dataloader(args, session, do_augment=True):
    # crop_transform, secondary_transform = get_transform(args)

    # Load support set (don't do data augmentation here )
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.data_folder, train=True, download=False,
                                         index=class_index, base_sess=False,
                                         mode='incre_train')  #, do_augment=do_augment)

    if args.dataset == 'mini_imagenet':
        preprocess = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        trainset = args.Dataset.MiniImageNet(root=args.data_folder, train=True,
                                             index_path=txt_path, do_augment=True, mode='incremental')


    if args.dataset == 'cub200':
        preprocess = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        txt_path = "data/index_list/" + "cub200" + "/session_" + str(session + 1) + '.txt'

        trainset = args.Dataset.CUB200(root=args.data_folder, train=True, index_path=txt_path,
                                       base_sess=False, mode='incremental')


        # always load entire dataset in one batch
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=trainset.__len__(), shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    only_new_class = get_new_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.data_folder, train=False, download=False,
                                        index=class_new, base_sess=False)

        testset_new_only = args.Dataset.CIFAR100(root=args.data_folder, train=False, download=False,
                                                 index=only_new_class, base_sess=False)

    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.data_folder, train=False,
                                            index=class_new)

        testset_new_only = args.Dataset.MiniImageNet(root=args.data_folder, train=False,
                                                     index=only_new_class, base_sess=False)

    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.data_folder, train=False, index=class_new)

        testset_new_only = args.Dataset.CUB200(root=args.data_folder, train=False,
                                               index=only_new_class, base_sess=False)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size_inference, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    testloader_new = torch.utils.data.DataLoader(dataset=testset_new_only, batch_size=args.batch_size_inference,
                                                 shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader, testloader_new



def get_base_dataloader_meta(args,do_augment=True):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)

    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.data_folder, train=False, download=False,
                                         index=class_index, base_sess=True) #, do_augment=do_augment)



        testset = args.Dataset.CIFAR100(root=args.data_folder, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.data_folder, train=True,
                                             index_path=txt_path, do_augment=False)
        testset = args.Dataset.MiniImageNet(root=args.data_folder, train=False,
                                            index=class_index)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.data_folder, train=True,index=class_index, base_sess=True, mode='meta')  # , do_augment=do_augment)

        testset = args.Dataset.CUB200(root=args.data_folder, train=False,index=class_index, base_sess=True)


    sampler = CategoriesSampler(trainset.targets, args.max_train_iter, args.num_ways_training,
                                 args.num_shots_training + args.num_query_training)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.batch_size_inference, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_session_classes(args, session):
    class_list = np.arange(args.base_class + session * args.way)
    return class_list


def get_new_classes(args, session):
    class_list = np.arange(args.base_class, args.base_class + session * args.way)
    return class_list
