
#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import pdb



def plot_confusion_support(prototypes, savepath=None):
    '''
    Parameters:
        prototypes: torch tensor (ways, dim_features)

    Returns:
        fig: pyplot figure

    '''
    cm =  get_confusion(prototypes).numpy()
    fig = plt.figure(figsize=(70, 70), dpi=200, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, vmin=-1, vmax=1, cmap='seismic')

    fig.set_tight_layout(True)
    fig.colorbar(im)

    # Add text labels to each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', color='black')

    if savepath is not None:
        fig.savefig(savepath + ".pdf")
        np.savez(savepath + ".npz", cm=cm)

    return fig



def get_confusion(support): 

    nways = support.shape[0]
    cm = t.zeros(nways,nways)
    cos = t.nn.CosineSimilarity()
    for way in range(nways): 
        cm[way] = cos(support[way:way+1],support)

    return cm

class avg_sim_confusion: 

    def __init__(self,nways,nways_session): 
        self.confusion_sum = t.zeros(nways,nways)
        self.nways_session = nways_session
        eps = 1e-8
        self.cnt = t.ones(1,nways)*eps

    def update(self,sim,onehot_label): 
        '''
        Parameters 
        ----------
        sim: Tensor (B,n_ways)
        onehot_label: Tensor (B,n_ways)
        '''
        acos_sim = t.acos(sim[:,:self.nways_session])
        self.confusion_sum[:self.nways_session] +=  t.matmul(t.transpose(acos_sim,0,1),onehot_label)
        self.cnt += t.sum(onehot_label,dim=0, keepdim=True)

    def plot(self):        
        cm = (self.confusion_sum/(self.cnt+1e-8))
        cm_diag = t.diagonal(cm).unsqueeze(0)
        interf_risk = cm_diag*t.div(1,cm+1e-8)
        mask = t.eye(interf_risk.shape[0],interf_risk.shape[1]).bool()
        interf_risk.masked_fill_(mask, 0)
        interf_risk[self.nways_session:]=0

        np.set_printoptions(precision=2)
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,7)) 
        ax1.imshow(t.transpose(cm,1,0), vmin=0, vmax=3.14,
               cmap='Blues')
        ax1.set_xlabel("Class vector")
        ax1.set_ylabel("Class data")

        ax2.imshow(t.transpose(interf_risk,1,0), vmin=0, vmax=1.5,
               cmap='Reds')
        ax2.set_xlabel("Class vector")
        ax2.set_ylabel("Class data")

        fig.set_tight_layout(True)

        return fig 


cifar100_name_dict = {'apple': 0, 'aquarium_fish': 1, 'baby': 2, 'bear': 3, 'beaver': 4, 'bed': 5, 'bee': 6,
                   'beetle': 7, 'bicycle': 8, 'bottle': 9, 'bowl': 10, 'boy': 11, 'bridge': 12, 'bus': 13,
                   'butterfly': 14, 'camel': 15, 'can': 16, 'castle': 17, 'caterpillar': 18, 'cattle': 19,
                   'chair': 20, 'chimpanzee': 21, 'clock': 22, 'cloud': 23, 'cockroach': 24, 'couch': 25, 'crab': 26, 'crocodile': 27, 'cup': 28, 'dinosaur': 29,
                   'dolphin': 30, 'elephant': 31,'flatfish': 32, 'forest': 33, 'fox': 34, 'girl': 35, 'hamster': 36, 'house': 37, 'kangaroo': 38,
                   'keyboard': 39, 'lamp': 40, 'lawn_mower': 41, 'leopard': 42, 'lion': 43, 'lizard': 44, 'lobster': 45,
                   'man': 46, 'maple_tree': 47, 'motorcycle': 48, 'mountain': 49, 'mouse': 50,
                   'mushroom': 51, 'oak_tree': 52, 'orange': 53, 'orchid': 54, 'otter': 55, 'palm_tree': 56, 'pear': 57,
                   'pickup_truck': 58, 'pine_tree': 59, 'plain': 60, 'plate': 61, 'poppy': 62,'porcupine': 63,
                    'possum': 64, 'rabbit': 65, 'raccoon': 66, 'ray': 67, 'road': 68, 'rocket': 69, 'rose': 70, 'sea': 71,
                   'seal': 72, 'shark': 73, 'shrew': 74, 'skunk': 75, 'skyscraper': 76, 'snail': 77, 'snake': 78,
                   'spider': 79, 'squirrel': 80, 'streetcar': 81, 'sunflower': 82, 'sweet_pepper': 83, 'table': 84,
                   'tank': 85, 'telephone': 86, 'television': 87, 'tiger': 88, 'tractor': 89, 'train': 90, 'trout': 91,
                   'tulip': 92, 'turtle': 93, 'wardrobe': 94, 'whale': 95, 'willow_tree': 96, 'wolf': 97, 'woman': 98, 'worm': 99}
