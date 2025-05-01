#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================
import numpy as np
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from .nudging import nudge_prototypes
from .embeddings.ResNet12 import ResNet12
from .embeddings.ResNet18 import resnet18
from .embeddings.ResNet20 import ResNet20
from .torch_blocks import fixCos, softstep, step, softabs, softrelu, cosine_similarity_multi, scaledexp, \
    linear_similarity_multi, Tanh10x
t.manual_seed(0)
import math



# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------


class KeyValueNetwork(nn.Module):
    # ----------------------------------------------------------------------------------------------
    # Special Functions & Properties
    # ----------------------------------------------------------------------------------------------

    def __init__(self, args, mode="meta"):
        super().__init__()

        self.args = args
        self.mode = mode
        self.fea_rep = None
        # Modules
        if args.block_architecture == "mini_resnet12":
            self.embedding = ResNet12(args)
        elif args.block_architecture == "mini_resnet18":
            if args.trainstage == 'pretrain_baseFSCIL':
                self.embedding = resnet18(pretrained=True, num_classes=args.dim_features)
            else:
                self.embedding = resnet18(pretrained=False, num_classes=args.dim_features)
        elif args.block_architecture == "mini_resnet20":
            self.embedding = ResNet20(num_classes=args.dim_features)

        # Load pretrain FC module
        if args.pretrainFC == "spherical":  # use cosine similarity
            self.fc_pretrain = fixCos(args.dim_features, args.base_class)
        else:

            self.classifier = nn.Parameter(t.FloatTensor(args.base_class, args.dim_features))
            nn.init.kaiming_uniform_(self.classifier, mode='fan_out', a=math.sqrt(5))


            self.fc_pretrain = nn.Linear(args.dim_features, args.base_class, bias=False)

        # Activations
        activation_functions = {
            'softabs': (lambda x: softabs(x, steepness=args.sharpening_strength)),
            'softrelu': (lambda x: softrelu(x, steepness=args.sharpening_strength)),
            'relu': nn.ReLU,
            'abs': t.abs,
            'scaledexp': (lambda x: scaledexp(x, s=args.sharpening_strength)),
            'exp': t.exp,
            'real': nn.Identity()

        }
        approximations = {
            'softabs': 'abs',
            'softrelu': 'relu'
        }

        self.sharpening_activation = activation_functions[args.sharpening_activation]

        # Access to intermediate activations
        self.intermediate_results = dict()

        self.feat_replay = t.zeros((args.num_classes, self.embedding.n_interm_feat)).cuda(args.gpu)
        self.label_feat_replay = t.diag(t.ones(self.args.num_classes)).cuda(args.gpu)


    # ----------------------------------------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------------------------------------

    def forward(self, inputs):
        '''
        Forward pass of main model

        Parameters:
        -----------
        inputs:  Tensor (B,H,W)
            Input data
        Return:
        -------
        output:  Tensor (B,ways)
        '''
        # Embed batch

        query_vectors = self.embedding(inputs)

        if self.mode == "pretrain":
            self.fea_rep = query_vectors

            output = F.linear(query_vectors, self.classifier)


        else:

            ##################### Cosine similarities #########################################################
            query_vectors = F.normalize(query_vectors, dim=-1)
            self.fea_rep = query_vectors

            self.similarities = cosine_similarity_multi(query_vectors, self.key_mem, rep=self.args.representation)


            ################# Sharpen the similarities with a soft absolute activation ############################
            similarities_sharpened = self.sharpening_activation(self.similarities)

            # Normalize the similarities in order to turn them into weightings
            if self.args.normalize_weightings:
                denom = t.sum(similarities_sharpened, dim=1, keepdim=True)
                weightings = t.div(similarities_sharpened, denom)
            else:

                weightings = similarities_sharpened

            # Return weighted sum of labels
            if self.args.average_support_vector_inference:
                output = weightings

            else:
                output = t.matmul(weightings, self.val_mem)

        return output

    def pseduo_feature_inference(self, p_features):
        # p_features = self.embedding.fc(p_feat)
        self.similarities = cosine_similarity_multi(p_features, self.key_mem.data[:self.args.base_class],
                                                    rep=self.args.representation)
        similarities_sharpened = nn.Softmax(dim=1)(self.similarities)

        # Normalize the similarities in order to turn them into weightings
        if self.args.normalize_weightings:
            denom = t.sum(similarities_sharpened, dim=1, keepdim=True)
            weightings = t.div(similarities_sharpened, denom)
        else:

            weightings = similarities_sharpened

        # Return weighted sum of labels
        if self.args.average_support_vector_inference:
            output = weightings

        else:
            output = t.matmul(weightings, self.val_mem)

        return output

    def reset_proto_cov(self, args):
        self.prototype = np.zeros([args.num_classes, self.embedding.n_interm_feat])
        self.cov = np.zeros([args.num_classes, self.embedding.n_interm_feat, self.embedding.n_interm_feat])
        self.class_label = []

    def write_mem(self, x, y):
        '''
        Rewrite key and value memory

        Parameters:
        -----------
        x:  Tensor (B,D)
            Input data
        y:  Tensor (B,w)
            One-hot encoded classes
        '''
        self.key_mem = self.embedding(x)
        self.val_mem = y

        if self.args.average_support_vector_inference:
            self.key_mem = t.matmul(t.transpose(self.val_mem, 0, 1), self.key_mem)
            # print(self.key_mem.shape)
        return

    def reset_prototypes(self, args):
        if hasattr(self, 'key_mem'):
            self.key_mem.data.fill_(0.0)
        else:
            self.key_mem = nn.parameter.Parameter(t.zeros(self.args.num_classes, self.args.dim_features),
                                                  requires_grad=False).cuda(args.gpu)
            self.val_mem = nn.parameter.Parameter(t.diag(t.ones(self.args.num_classes)), requires_grad=False).cuda(
                args.gpu)

    def create_dummy_classifier(self, args):
        self.dummy_classifier = nn.parameter.Parameter(t.zeros(self.args.num_classes, self.args.dim_features),
                                                       requires_grad=False).cuda(args.gpu)

    def initialize_new_classifier(self, args, seen_class_num, session, old_class_num):

        self.new_classifier = nn.Parameter(t.FloatTensor(seen_class_num, self.args.dim_features),
                                           requires_grad=True).cuda(args.gpu)
        nn.init.kaiming_uniform_(self.new_classifier, mode='fan_out', a=math.sqrt(5))

        if session == 0:
            self.new_classifier.data[:old_class_num] = self.classifier.data[:old_class_num]
        else:
            self.new_classifier.data[:old_class_num] = self.dummy_classifier.data[:old_class_num]

    def save_old_classifier(self, seen_class_num):
        self.dummy_classifier.data[:seen_class_num] = self.new_classifier.data[:seen_class_num]

    def update_prototypes(self, x, y):
        '''
        Update key memory

        Parameters:
        -----------
        x:  Tensor (B,D)
            Input data
        y:  Tensor (B)
            lables
        '''

        support_vec = self.embedding(x)
        y_onehot = F.one_hot(y, num_classes=self.args.num_classes).float()
        prototype_vec = t.matmul(t.transpose(y_onehot, 0, 1), support_vec)
        self.key_mem.data += prototype_vec


    def get_sum_support(self, x, y):
        '''
        Compute prototypes

        Parameters:
        -----------
        x:  Tensor (B,D)
            Input data
        y:  Tensor (B)
            lables
        '''
        support_vec = self.embedding(x)
        y_onehot = F.one_hot(y, num_classes=self.args.num_classes).float()
        sum_cnt = t.sum(y_onehot, dim=0).unsqueeze(1)
        sum_support = t.matmul(t.transpose(y_onehot, 0, 1), support_vec)
        return sum_support, sum_cnt

    def update_feat_replay(self, x, y):
        '''
        Compute feature representatin of new data and update
        Parameters:
        -----------
        x   t.Tensor(B,in_shape)
            Input raw images
        y   t.Tensor (B)
            Input labels

        Return:
        -------
        '''
        feat_vec = self.embedding.forward_conv(x)

        y_onehot = F.one_hot(y, num_classes=self.args.num_classes).float()

        sum_cnt = t.sum(y_onehot, dim=0).unsqueeze(1)

        sum_feat_vec = t.matmul(t.transpose(y_onehot, 0, 1), feat_vec)

        avg_feat_vec = t.div(sum_feat_vec, sum_cnt + 1e-8)


        self.feat_replay += avg_feat_vec

    def get_feat_replay(self):
        return self.feat_replay, self.label_feat_replay

    def update_prototypes_feat(self, feat, y_onehot, nways=None):
        '''
        Update key

        Parameters:
        -----------
        feat:  Tensor (B,d_f)
            Input data
        y:  Tensor (B)
        nways: int
            If none: update all prototypes, if int, update only nwyas prototypes
        '''
        support_vec = self.get_support_feat(feat)
        prototype_vec = t.matmul(t.transpose(y_onehot, 0, 1), support_vec)

        if nways is not None:

            self.key_mem.data[:nways] += prototype_vec[:nways]
            # self.key_mem.data[:self.args.base_class] = self.classifier.data

            if self.args.dataset=='cifar100':
                self.key_mem.data[:self.args.base_class] = 0.2 * self.key_mem.data[:self.args.base_class] + 0.8 * self.classifier.data[:self.args.base_class]

            elif self.args.dataset=='mini_imagenet':

                self.key_mem.data[:self.args.base_class] = 0.05 * self.key_mem.data[:self.args.base_class] + 0.95*self.classifier.data[:self.args.base_class]

            else:
                self.key_mem.data[:self.args.base_class] = self.classifier.data


            self.key_mem.data[:nways] = F.normalize(self.key_mem.data[:nways])




        else:
            self.key_mem.data += prototype_vec


    def protoreplay(self):
        self.key_mem.data[:self.args.base_class] = self.classifier.data

    def get_support_feat(self, feat):

        '''
        Pass activations through final FC

        Parameters:
        -----------
        feat:  Tensor (B,d_f)
            Input data
        Return:
        ------
        support_vec:  Tensor (B,d)
            Mapped support vectors
        '''
        support_vec = self.embedding.fc(feat)
        return support_vec

    def nudge_prototypes(self, num_ways, writer, session, gpu):
        '''
        Prototype nudging
        Parameters:
        -----------
        num_ways:   int
        writer:     Tensorboard writer
        session:    int
        gpu:        int

        '''
        prototypes_orig = self.key_mem.data[:num_ways]
        self.key_mem.data[:num_ways] = nudge_prototypes(prototypes_orig, writer, session,
                                                        gpu=self.args.gpu, num_epochs=self.args.nudging_iter,
                                                        bipolarize_prototypes=self.args.bipolarize_prototypes,
                                                        act=self.args.nudging_act,
                                                        act_exp=self.args.nudging_act_exp)
        return

    def protoSave(self, loader):
        features = []
        labels = []
        with torch.no_grad():
            for i, (data, target) in enumerate(loader):

                # data, target = [_.cuda(self.args.gpu, non_blocking=True) for _ in batch]

                feature = self.embedding(data.cuda(self.args.gpu))

                if feature.shape[0] == self.args.batch_size_training:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())

        # print(len(labels),type(labels[0]))
        # print(np.array(labels).shape)
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))

        prototype = []
        cov = []
        class_label = []
        # radius = []

        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)

            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
            cov_class = np.cov(feature_classwise.T)
            cov.append(cov_class)
            # radius.append(np.trace(cov) / features.shape[1])


        cov = np.concatenate(cov, axis=0).reshape([-1, self.args.dim_features, self.args.dim_features])
        prototype = np.array(prototype)
        class_label = class_label
        # radius = np.sqrt(np.mean(radius))

        return prototype, cov, class_label



