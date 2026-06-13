# General
import os
import numpy as np
from pathlib import Path
import typing
import time
import json
import copy
from typing import Dict, List
import multiprocessing
from multiprocessing import Pool
from itertools import accumulate, product
from collections import OrderedDict
import pickle
import sys
from functools import partial


#Sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# Pandas
import pandas as pd

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

# Pytorch lightning
import pytorch_lightning as pl
import wandb

# Pytorch Geometric
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import MessagePassing, GINConv

# Similarity calculations
from fastdtw import fastdtw

# Networkx
import networkx as nx


# Our Methods
sys.path.insert(0, '..') # add config to path
import config
import subgraph_utils
from subgraph_mpn import SG_MPN
from datasets import SubgraphDataset
import anchor_patch_samplers
from anchor_patch_samplers import *
import gamma
import attention


def _calc_dtw_pair(args):
    component_degree, patch_degree = args
    return gamma.calc_dtw(component_degree, patch_degree)


def _calc_dtw_indexed(args):
    flat_idx, anchor_idx, component_degree, patch_degree = args
    return flat_idx, anchor_idx, gamma.calc_dtw(component_degree, patch_degree)


class LSTM(nn.Module):
    '''
    bidirectional LSTM with linear head
    '''
    def __init__(self, n_features, h, dropout=0.0, num_layers=1, batch_first=True, aggregator='last'):
        super().__init__()

        # number of LSTM layers
        self.num_layers = num_layers

        # type of aggregation('sum' or 'last')
        self.aggregator = aggregator

        self.lstm = nn.LSTM(n_features, h, num_layers=num_layers, batch_first=batch_first, dropout=dropout, bidirectional=True)
        self.linear = nn.Linear(h * 2, n_features)
    
    def forward(self, input):
        #input: (batch_sz, seq_len, hidden_dim )
        lstm_out, last_hidden = self.lstm(input)
        batch, seq_len, _ = lstm_out.shape

        # either take last hidden state or sum all hidden states
        if self.aggregator == 'last':
            lstm_agg = lstm_out[:,-1,:]
        elif self.aggregator == 'sum':
            lstm_agg = torch.sum(lstm_out, dim=1)
        else:
            raise NotImplementedError
        return self.linear(lstm_agg)

class SubGNN(pl.LightningModule):
    '''
        Pytorch lightning class for SubGNN
    '''
    def __init__(self, hparams: Dict, graph_path: str, subgraph_path: str, 
        embedding_path: str, similarities_path: str, shortest_paths_path:str,
        degree_dict_path: str, ego_graph_path: str):
        super(SubGNN, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #dictionary of hyperparameters
        self.hparams = hparams  
        
        # paths where data is stored 
        self.graph_path = graph_path
        self.subgraph_path = subgraph_path
        self.embedding_path = embedding_path
        self.similarities_path = Path(similarities_path)
        self.shortest_paths_path = shortest_paths_path
        self.degree_dict_path = degree_dict_path
        self.ego_graph_path = ego_graph_path

        # read in data
        self.read_data()

        self.use_aml_edge_features = bool(self.hparams.get('use_aml_edge_features', False)) \
            and int(self.hparams.get('aml_edge_feature_dim', 0)) > 0

        # wandb logging control
        self.wandb_epoch_logging = bool(self.hparams.get('wandb_epoch_logging', False))

        # initialize MPN layers for each channel (neighborhood, structure, position; internal, border) 
        # and each layer (up to 'n_layers')
        hid_dim = self.hparams['node_embed_size'] 
        if self.use_aml_edge_features:
            hid_dim += int(self.hparams.get('aml_edge_feature_dim', 0))
        self.neighborhood_mpns = nn.ModuleList()
        if self.hparams['use_neighborhood']:
            hid_dim += self.hparams['n_layers'] * 2 * self.hparams['node_embed_size'] #automatically infer hidden dimension
            for l in range(self.hparams['n_layers']):
                curr_layer = nn.ModuleDict() 
                curr_layer['internal'] = SG_MPN(self.hparams)
                curr_layer['border'] = SG_MPN(self.hparams)
                # optionally add batch_norm
                if 'batch_norm' in self.hparams and self.hparams['batch_norm']:
                    curr_layer['batch_norm'] = nn.BatchNorm1d(self.hparams['node_embed_size']).to(self.device)
                    curr_layer['batch_norm_out'] = nn.BatchNorm1d(self.hparams['node_embed_size']).to(self.device)
                self.neighborhood_mpns.append(curr_layer)

        self.position_mpns = nn.ModuleList()
        if self.hparams['use_position']:
            hid_dim = hid_dim + (self.hparams['n_anchor_patches_pos_in'] + self.hparams['n_anchor_patches_pos_out']) * self.hparams['n_layers']
            for l in range(self.hparams['n_layers']):
                curr_layer = nn.ModuleDict() 
                curr_layer['internal'] = SG_MPN(self.hparams)
                curr_layer['border'] = SG_MPN(self.hparams)
                # optionally add batch_norm
                if 'batch_norm' in self.hparams and self.hparams['batch_norm']:
                    curr_layer['batch_norm'] = nn.BatchNorm1d(self.hparams['node_embed_size']).to(self.device)
                    curr_layer['batch_norm_out'] = nn.BatchNorm1d(self.hparams['node_embed_size']).to(self.device)
                self.position_mpns.append(curr_layer)

        self.structure_mpns = nn.ModuleList()
        if self.hparams['use_structure']:
            hid_dim += 2 * self.hparams['n_anchor_patches_structure'] * self.hparams['n_layers']
            for l in range(self.hparams['n_layers']):
                curr_layer = nn.ModuleDict() 
                curr_layer['internal'] = SG_MPN(self.hparams)
                curr_layer['border'] = SG_MPN(self.hparams)
                # optionally add batch_norm
                if 'batch_norm' in self.hparams and self.hparams['batch_norm']:
                    curr_layer['batch_norm'] = nn.BatchNorm1d(self.hparams['node_embed_size']).to(self.device)
                    curr_layer['batch_norm_out'] = nn.BatchNorm1d(self.hparams['node_embed_size']).to(self.device)

                self.structure_mpns.append(curr_layer)

        # initialize 3 FF layers on top of MPN layers
        self.lin =  nn.Linear(hid_dim, self.hparams['linear_hidden_dim_1'])
        self.lin2 =  nn.Linear(self.hparams['linear_hidden_dim_1'], self.hparams['linear_hidden_dim_2'])
        self.lin3 =  nn.Linear(self.hparams['linear_hidden_dim_2'], self.num_classes)

        # optional dropout on the linear layers
        self.lin_dropout = nn.Dropout(p=self.hparams['lin_dropout'])
        self.lin_dropout2 = nn.Dropout(p=self.hparams['lin_dropout'])

        # initialize loss
        if self.multilabel:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            # CrossEntropyLoss uses `weight[i]` for class i.
            # Prefer an explicitly tuned illicit-class weight, then manual full weights,
            # then deriving weights from the *training* label distribution unless overridden.
            use_auto_weights = bool(self.hparams.get('auto_class_weights', True))
            manual_weights = self.hparams.get('ce_class_weights', None)
            illicit_class_weight = self.hparams.get('illicit_class_weight', None)

            weight = None
            if illicit_class_weight is not None:
                weight = torch.ones(int(self.num_classes), dtype=torch.float)
                illicit_idx = self.hparams.get('illicit_class_idx', None)
                if illicit_idx is None and hasattr(self, 'label_mapping'):
                    illicit_idx = self.label_mapping.get('illicit_component', None)
                class_counts = getattr(self, 'class_counts', None)
                if illicit_idx is None and isinstance(class_counts, torch.Tensor):
                    illicit_idx = int(torch.argmin(class_counts).item())
                if illicit_idx is None:
                    illicit_idx = 0
                weight[int(illicit_idx)] = float(illicit_class_weight)
            elif manual_weights is not None:
                weight = torch.tensor(manual_weights, dtype=torch.float)
            elif use_auto_weights and getattr(self, 'class_weights', None) is not None:
                weight = self.class_weights

            self.loss = nn.CrossEntropyLoss(weight=weight)
            # pos_weight = torch.tensor([1 / 7])
            # self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # initialize LSTM - this is used in the structure channel for embedding anchor patches
        self.lstm = LSTM(self.hparams['node_embed_size'], self.hparams['node_embed_size'], \
            dropout=self.hparams['lstm_dropout'], num_layers=self.hparams['lstm_n_layers'], \
            aggregator=self.hparams['lstm_aggregator'])

        # optionally, use feedforward attention 
        if 'ff_attn' in self.hparams and self.hparams['ff_attn']:
            self.attn_vector = torch.nn.Parameter(torch.zeros((hid_dim,1), dtype=torch.float).to(self.device), requires_grad=True)   
            nn.init.xavier_uniform_(self.attn_vector)
            self.attention = attention.AdditiveAttention(hid_dim, hid_dim)
     
        # default similarity function for the structure channel is dynamic time warping
        if 'structure_similarity_fn' not in self.hparams:
            self.hparams['structure_similarity_fn'] = 'dtw'

        # track metrics (used for optuna)
        self.metric_scores = []
        self.train_metric_scores = []
        self._keep_outputs = False
        self._valid_outputs = []

    @staticmethod
    def _wandb_prefix_metrics(metrics: Dict, split: str) -> Dict:
        prefixed = {}
        split_prefix = f"{split}_"
        for key, value in metrics.items():
            if key.startswith(split_prefix):
                mapped_key = key[len(split_prefix):]
            elif key.startswith("avg_" + split_prefix):
                mapped_key = "avg_" + key[len("avg_" + split_prefix):]
            elif key.startswith("avg_"):
                mapped_key = key
            else:
                mapped_key = key
            prefixed[f"{split}/{mapped_key}"] = value
        return prefixed

##################################################
# forward pass
    
    def run_mpn_layer(self, dataset_type, mpn_fn, subgraph_ids, subgraph_idx, cc_ids, \
            cc_embeds, cc_embed_mask, sims, layer_num, channel, inside=True):
        '''
        Perform a single message-passing layer for the specified 'channel' and internal/border

        Returns:
            - cc_embed_matrix: updated connected component embedding matrix
            - position_struc_out: property aware embedding matrix (for position & structure channels)

        '''
        # batch_sz, max_n_cc, max_size_cc = cc_ids.shape
        # self.graph.x  (n_nodes, hidden dim)

        # Get Anchor Patches
        anchor_patches, anchor_mask, anchor_embeds = get_anchor_patches(dataset_type, self.hparams, \
            self.networkx_graph, self.node_embeddings, subgraph_idx, cc_ids, cc_embed_mask, self.lstm, 
            self.anchors_neigh_int, self.anchors_neigh_border, self.anchors_pos_int, \
            self.anchors_pos_ext, self.anchors_structure, layer_num, channel, inside, self.device)

        # for the structure channel, we need to also pass in indices into larger matrix of pre-sampled structure AP
        if channel == 'structure': anchors_sim_index = self.anchors_structure[layer_num][1] 
        else: anchors_sim_index = None
        
        # one layer of message passing
        cc_embed_matrix, position_struc_out = mpn_fn(self.networkx_graph, sims, cc_ids, 
            cc_embeds, cc_embed_mask, anchor_patches, anchor_embeds, 
            anchor_mask, anchors_sim_index)

        return cc_embed_matrix, position_struc_out

    def forward(self, dataset_type, N_I_cc_embed, N_B_cc_embed, \
        S_I_cc_embed, S_B_cc_embed, P_I_cc_embed, P_B_cc_embed, \
        subgraph_ids, cc_ids, subgraph_idx, NP_sim, \
        I_S_sim, B_S_sim):

        '''
        subgraph_ids: (batch_sz, max_len_sugraph)
        cc_ids: (batch_sz, max_n_cc, max_len_cc)
        '''

        # create cc_embeds matrix for each channel: (batch_sz, max_n_cc, hidden_dim)
        init_cc_embeds = self.initialize_cc_embeddings(cc_ids, self.hparams['cc_aggregator'])
        if not self.hparams['trainable_cc']: # if the cc embeddings, aren't trainable, we clone them
            N_in_cc_embeds = init_cc_embeds.clone()
            N_out_cc_embeds = init_cc_embeds.clone()
            P_in_cc_embeds = init_cc_embeds.clone()
            P_out_cc_embeds = init_cc_embeds.clone()
            S_in_cc_embeds = init_cc_embeds.clone()
            S_out_cc_embeds = init_cc_embeds.clone()
        else: # otherwise, we index into the intialized cc embeddings for each channel using the subgraph ids for the given batch
            N_in_cc_embeds = torch.index_select(N_I_cc_embed, 0, subgraph_idx.squeeze(-1))
            N_out_cc_embeds = torch.index_select(N_B_cc_embed, 0, subgraph_idx.squeeze(-1))
            P_in_cc_embeds = torch.index_select(P_I_cc_embed, 0, subgraph_idx.squeeze(-1))
            P_out_cc_embeds = torch.index_select(P_B_cc_embed, 0, subgraph_idx.squeeze(-1))
            S_in_cc_embeds = torch.index_select(S_I_cc_embed, 0, subgraph_idx.squeeze(-1))
            S_out_cc_embeds = torch.index_select(S_B_cc_embed, 0, subgraph_idx.squeeze(-1))

        batch_sz, max_n_cc, _ = init_cc_embeds.shape
        
        #get mask for cc_embeddings
        cc_embed_mask = (cc_ids != config.PAD_VALUE)[:,:,0] # only take first element bc only need mask over n_cc, not n_nodes in cc
        
        
        # for each layer in SubGNN:
        outputs = []
        for l in range(self.hparams['n_layers']):

            # neighborhood channel
            if self.hparams['use_neighborhood']:
                # message passing layer for N internal and border
                N_in_cc_embeds, _ = self.run_mpn_layer(dataset_type, self.neighborhood_mpns[l]['internal'], subgraph_ids, subgraph_idx, cc_ids, N_in_cc_embeds, cc_embed_mask, NP_sim, layer_num=l, channel='neighborhood', inside=True)
                N_out_cc_embeds, _ = self.run_mpn_layer(dataset_type, self.neighborhood_mpns[l]['border'], subgraph_ids, subgraph_idx, cc_ids, N_out_cc_embeds, cc_embed_mask, NP_sim, layer_num=l, channel='neighborhood', inside=False)
                if 'batch_norm' in self.hparams and self.hparams['batch_norm']: #optional batch norm
                    N_in_cc_embeds = self.neighborhood_mpns[l]['batch_norm'](N_in_cc_embeds.view(batch_sz*max_n_cc,-1)).view(batch_sz,max_n_cc, -1 )
                    N_out_cc_embeds = self.neighborhood_mpns[l]['batch_norm_out'](N_out_cc_embeds.view(batch_sz*max_n_cc,-1)).view(batch_sz,max_n_cc, -1 )

                outputs.extend([N_in_cc_embeds, N_out_cc_embeds])
            
            # position channel
            if self.hparams['use_position']:
                # message passing layer for P internal and border
                P_in_cc_embeds, P_in_position_embed = self.run_mpn_layer(dataset_type, self.position_mpns[l]['internal'], subgraph_ids, subgraph_idx,  cc_ids, P_in_cc_embeds, cc_embed_mask, NP_sim, layer_num=l, channel='position', inside=True)
                P_out_cc_embeds, P_out_position_embed = self.run_mpn_layer(dataset_type, self.position_mpns[l]['border'], subgraph_ids, subgraph_idx, cc_ids, P_out_cc_embeds, cc_embed_mask, NP_sim, layer_num=l, channel='position', inside=False)
                if 'batch_norm' in self.hparams and self.hparams['batch_norm']:  #optional batch norm
                    P_in_cc_embeds = self.position_mpns[l]['batch_norm'](P_in_cc_embeds.view(batch_sz*max_n_cc,-1)).view(batch_sz,max_n_cc, -1 )
                    P_out_cc_embeds = self.position_mpns[l]['batch_norm_out'](P_out_cc_embeds.view(batch_sz*max_n_cc,-1)).view(batch_sz,max_n_cc, -1 )
                outputs.extend([P_in_position_embed, P_out_position_embed])
            
            # structure channel
            if self.hparams['use_structure']:
                # message passing layer for S internal and border
                S_in_cc_embeds, S_in_struc_embed = self.run_mpn_layer(dataset_type, self.structure_mpns[l]['internal'], subgraph_ids, subgraph_idx, cc_ids, S_in_cc_embeds, cc_embed_mask, I_S_sim, layer_num=l, channel='structure', inside=True)
                S_out_cc_embeds, S_out_struc_embed = self.run_mpn_layer(dataset_type, self.structure_mpns[l]['border'], subgraph_ids, subgraph_idx, cc_ids, S_out_cc_embeds, cc_embed_mask, B_S_sim, layer_num=l, channel='structure', inside=False)
                if 'batch_norm' in self.hparams and self.hparams['batch_norm']:  #optional batch norm
                    S_in_cc_embeds = self.structure_mpns[l]['batch_norm'](S_in_cc_embeds.view(batch_sz*max_n_cc,-1)).view(batch_sz,max_n_cc, -1 )
                    S_out_cc_embeds = self.structure_mpns[l]['batch_norm_out'](S_out_cc_embeds.view(batch_sz*max_n_cc,-1)).view(batch_sz,max_n_cc, -1 )
                outputs.extend([S_in_struc_embed, S_out_struc_embed])

        
        # concatenate all layers
        all_cc_embeds = torch.cat([init_cc_embeds] + outputs, dim=-1)
        
        # sum across all CC
        if 'ff_attn' in self.hparams and self.hparams['ff_attn']:
            batched_attn = self.attn_vector.squeeze().unsqueeze(0).repeat(all_cc_embeds.shape[0],1)
            attn_weights = self.attention(batched_attn, all_cc_embeds, cc_embed_mask)
            subgraph_embedding = subgraph_utils.weighted_sum(all_cc_embeds, attn_weights)
        else:
            subgraph_embedding = subgraph_utils.masked_sum(all_cc_embeds, cc_embed_mask.unsqueeze(-1), dim=1, keepdim=False)

        aml_edge_features = self._get_aml_edge_features(dataset_type, subgraph_idx)
        if aml_edge_features is not None:
            subgraph_embedding = torch.cat([subgraph_embedding, aml_edge_features], dim=-1)
        
        # Fully Con Layers + Optional Dropout
        subgraph_embedding_out = F.relu(self.lin(subgraph_embedding))
        subgraph_embedding_out = self.lin_dropout(subgraph_embedding_out) 
        subgraph_embedding_out = F.relu(self.lin2(subgraph_embedding_out))
        subgraph_embedding_out = self.lin_dropout2(subgraph_embedding_out) 
        subgraph_embedding_out = self.lin3(subgraph_embedding_out) 

        return subgraph_embedding_out

    def _get_aml_edge_features(self, dataset_type, subgraph_idx):
        if not getattr(self, 'use_aml_edge_features', False):
            return None

        attr = {
            'train': 'train_aml_edge_features',
            'val': 'val_aml_edge_features',
            'test': 'test_aml_edge_features',
        }.get(dataset_type)
        if attr is None:
            return None

        features = getattr(self, attr, None)
        if features is None:
            return None

        idx = subgraph_idx.squeeze(-1).long().cpu()
        selected = torch.index_select(features, 0, idx)
        return selected.to(subgraph_idx.device)

    ##################################################
     # training, val, test steps 

    def training_step(self, train_batch, batch_idx):
        '''
        Runs a single training step over the batch
        '''
        # get subgraphs and labels
        subgraph_ids = train_batch['subgraph_ids']
        cc_ids = train_batch['cc_ids']
        subgraph_idx = train_batch['subgraph_idx']
        labels = train_batch['label'].squeeze(-1)

        # get similarities for batch
        NP_sim = train_batch['NP_sim']
        I_S_sim = train_batch['I_S_sim']
        B_S_sim = train_batch['B_S_sim']

        # forward pass
        logits = self.forward('train', self.train_N_I_cc_embed, self.train_N_B_cc_embed, \
            self.train_S_I_cc_embed, self.train_S_B_cc_embed, self.train_P_I_cc_embed, self.train_P_B_cc_embed, \
            subgraph_ids, cc_ids, subgraph_idx, NP_sim, I_S_sim, B_S_sim)

        # calculate loss
        if len(labels.shape) == 0: labels = labels.unsqueeze(-1)
        if self.multilabel:
            loss = self.loss(logits.squeeze(1), labels.type_as(logits)) 
        else:
            loss = self.loss(logits, labels)

        # calculate accuracy
        pred_threshold = float(self.hparams.get('pred_threshold', 0.5))
        acc = subgraph_utils.calc_accuracy(logits, labels, multilabel_binarizer=self.multilabel_binarizer, threshold=pred_threshold)

        logs = {'train_loss': loss, 'train_acc': acc} # used for tensorboard
        if self.wandb_epoch_logging:
            return {'loss': loss, 'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'log': logs}

    def training_epoch_end(self, outputs):
        if len(outputs) == 0:
            return
        if 'train_loss' not in outputs[0] or 'train_acc' not in outputs[0]:
            return
        train_loss = torch.stack([x['train_loss'] for x in outputs]).mean().cpu()
        train_acc = torch.stack([x['train_acc'] for x in outputs]).mean().cpu()
        self.train_metric_scores.append({'train_loss': train_loss, 'train_acc': train_acc})
        if self.wandb_epoch_logging and wandb.run is not None:
            wandb.log(self._wandb_prefix_metrics({'train_loss': train_loss, 'train_acc': train_acc}, 'train'), step=self.current_epoch)
        
    def val_test_step(self, batch, batch_idx, is_test = False):
        '''
        Runs a single validation or test step over the batch
        '''

        # get subgraphs and labels
        subgraph_ids = batch['subgraph_ids']
        cc_ids = batch['cc_ids']
        subgraph_idx = batch['subgraph_idx']
        labels = batch['label'].squeeze(-1)

        # get similarities for batch
        NP_sim = batch['NP_sim']
        I_S_sim = batch['I_S_sim']
        B_S_sim = batch['B_S_sim']
        
        # forward pass
        if not is_test:
            logits = self.forward('val', self.val_N_I_cc_embed, self.val_N_B_cc_embed, \
                self.val_S_I_cc_embed, self.val_S_B_cc_embed, self.val_P_I_cc_embed, self.val_P_B_cc_embed, \
                subgraph_ids, cc_ids, subgraph_idx, NP_sim, I_S_sim, B_S_sim)
        else:
            logits = self.forward('test', self.test_N_I_cc_embed, self.test_N_B_cc_embed, \
                self.test_S_I_cc_embed, self.test_S_B_cc_embed, self.test_P_I_cc_embed, self.test_P_B_cc_embed, \
                subgraph_ids, cc_ids, subgraph_idx, NP_sim, I_S_sim, B_S_sim)

        # calc loss
        if len(labels.shape) == 0: labels = labels.unsqueeze(-1)
        if self.multilabel:
            loss = self.loss(logits.squeeze(1), labels.type_as(logits))
        else:
            loss = self.loss(logits, labels)

        # calc accuracy and macro F1
        pred_threshold = float(self.hparams.get('pred_threshold', 0.5))
        acc = subgraph_utils.calc_accuracy(logits, labels, multilabel_binarizer=self.multilabel_binarizer, threshold=pred_threshold)
        macro_f1 = subgraph_utils.calc_f1(logits, labels, avg_type='macro', multilabel_binarizer=self.multilabel_binarizer, threshold=pred_threshold)


        if not is_test: # for tensorboard
            return {'val_loss': loss, 'val_acc': acc, 'val_macro_f1': macro_f1, 'val_logits': logits, 'val_labels': labels} 
        else:
            return {'test_loss': loss, 'test_acc': acc, 'test_macro_f1': macro_f1, 'test_logits': logits, 'test_labels': labels}

    def validation_step(self, val_batch, batch_idx):
        '''
        wrapper for self.val_test_step
        '''
        out = self.val_test_step(val_batch, batch_idx, is_test = False)
        if getattr(self, "_keep_outputs", False):
            self._valid_outputs.append({
                "logits": out["val_logits"].detach().cpu(),
                "labels": out["val_labels"].detach().cpu(),
            })
        return out

    def test_step(self, test_batch, batch_idx):
        '''
        wrapper for self.val_test_step
        '''
        return self.val_test_step(test_batch, batch_idx, is_test = True) 

    def _add_binary_diagnostics(self, tensorboard_logs, prefix, logits, labels, pred_threshold):
        if self.multilabel or logits.dim() != 2 or logits.shape[1] != 2:
            return

        labels_cpu = labels.detach().cpu().view(-1)
        preds_cpu = subgraph_utils.binary_threshold_predictions(logits, labels, pred_threshold).detach().cpu().view(-1)
        probs_cpu = F.softmax(logits.detach(), dim=1).cpu()
        minority_class = subgraph_utils.get_minority_class(labels_cpu)
        majority_class = 1 - minority_class
        minority_probs = probs_cpu[:, minority_class]
        minority_mask = labels_cpu == minority_class
        majority_mask = labels_cpu == majority_class
        true_minority_probs = minority_probs[minority_mask]
        false_minority_probs = minority_probs[majority_mask]

        def _safe_mean(values):
            return float(values.mean().item()) if values.numel() > 0 else 0.0

        def _safe_max(values):
            return float(values.max().item()) if values.numel() > 0 else 0.0

        def _safe_quantile(values, q):
            return float(torch.quantile(values, q).item()) if values.numel() > 0 else 0.0

        tensorboard_logs.update({
            f'{prefix}_minority_class': minority_class,
            f'{prefix}_label_class_0': int((labels_cpu == 0).sum().item()),
            f'{prefix}_label_class_1': int((labels_cpu == 1).sum().item()),
            f'{prefix}_pred_class_0': int((preds_cpu == 0).sum().item()),
            f'{prefix}_pred_class_1': int((preds_cpu == 1).sum().item()),
            f'{prefix}_minority_prob_mean': float(minority_probs.mean().item()),
            f'{prefix}_minority_prob_true_mean': _safe_mean(true_minority_probs),
            f'{prefix}_minority_prob_false_mean': _safe_mean(false_minority_probs),
            f'{prefix}_minority_prob_true_p95': _safe_quantile(true_minority_probs, 0.95),
            f'{prefix}_minority_prob_false_p95': _safe_quantile(false_minority_probs, 0.95),
            f'{prefix}_minority_prob_true_max': _safe_max(true_minority_probs),
            f'{prefix}_minority_prob_false_max': _safe_max(false_minority_probs),
            f'{prefix}_minority_true_above_threshold': int((true_minority_probs > pred_threshold).sum().item()),
            f'{prefix}_minority_false_above_threshold': int((false_minority_probs > pred_threshold).sum().item()),
            f'{prefix}_f1_class_0': f1_score(labels_cpu.numpy(), preds_cpu.numpy(), labels=[0], average='macro', zero_division=0),
            f'{prefix}_f1_class_1': f1_score(labels_cpu.numpy(), preds_cpu.numpy(), labels=[1], average='macro', zero_division=0),
        })

##################################################
# validation & test epoch end

    def validation_epoch_end(self, outputs):
        '''
        called at the end of the validation epoch
        
        Input:
            - outputs: is an array with what you returned in validation_step for each batch
              outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 
        '''

        # aggregate the logits, labels, and metrics for all batches
        logits = torch.cat([x['val_logits'] for x in outputs], dim=0)
        labels = torch.cat([x['val_labels'] for x in outputs], dim=0)

        pred_threshold = float(self.hparams.get('pred_threshold', 0.5))
        self.best_val_threshold = pred_threshold
        macro_f1 = subgraph_utils.calc_f1(logits, labels, avg_type='macro', multilabel_binarizer=self.multilabel_binarizer, threshold=pred_threshold).squeeze()
        micro_f1 = subgraph_utils.calc_f1(logits, labels, avg_type='micro', multilabel_binarizer=self.multilabel_binarizer, threshold=pred_threshold).squeeze()
        acc = subgraph_utils.calc_accuracy(logits, labels, multilabel_binarizer=self.multilabel_binarizer, threshold=pred_threshold).squeeze()

        # calc AUC/PR
        if self.multilabel:
            auroc = roc_auc_score(labels.cpu(), torch.sigmoid(logits).cpu(), multi_class = 'ovr')
        elif len(torch.unique(labels)) == 2: #binary case
            auroc = roc_auc_score(labels.cpu(), F.softmax(logits, dim=1).cpu()[:,1])
        else: #multiclass
            auroc = roc_auc_score(labels.cpu(), F.softmax(logits, dim=1).cpu(), multi_class = 'ovr')
        auprc = subgraph_utils.calc_auprc(logits, labels, multilabel_binarizer=self.multilabel_binarizer)
        illicit_pr_auc = subgraph_utils.calc_minor_pr_auc(logits, labels, multilabel_binarizer=self.multilabel_binarizer)
        minor_f1 = subgraph_utils.calc_minor_f1(logits, labels, multilabel_binarizer=self.multilabel_binarizer, threshold=pred_threshold).squeeze()

        # get average loss, acc, and macro F1 over batches
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().cpu()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_macro_f1 = torch.stack([x['val_macro_f1'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_micro_f1': micro_f1, 'val_macro_f1': macro_f1, \
            'val_acc': acc, 'avg_val_acc': avg_acc, 'avg_macro_f1':avg_macro_f1, 'val_auroc':auroc, \
                            'val_auprc': auprc, 'val_minor_f1': minor_f1, 'val_illicit_pr_auc': illicit_pr_auc,
                            'val_threshold': pred_threshold }

        self._add_binary_diagnostics(tensorboard_logs, 'val', logits, labels, pred_threshold)
        if self.multilabel:
            for c in range(logits.shape[1]): #n_classes
                tensorboard_logs['val_auroc_class_' + str(c)] = roc_auc_score(labels[:, c].cpu(), torch.sigmoid(logits)[:, c].cpu())
        else:
            one_hot_labels = one_hot(labels, num_classes = logits.shape[1])
            for c in range(logits.shape[1]): #n_classes
                tensorboard_logs['val_auroc_class_' + str(c)] = roc_auc_score(one_hot_labels[:, c].cpu(), logits[:, c].cpu())

        # Re-Initialize cc_embeds 
        if not self.hparams['trainable_cc']:
            self.init_all_embeddings(split = 'train_val', trainable = self.hparams['trainable_cc'])

        # Optionally re initialize anchor patches each epoch (defaults to false)
        if self.hparams['resample_anchor_patches']:
            if self.hparams['use_neighborhood']:
                self.anchors_neigh_int, self.anchors_neigh_border = init_anchors_neighborhood('train_val', self.hparams, self.networkx_graph, self.device, self.train_cc_ids, self.val_cc_ids, self.test_cc_ids)
            if self.hparams['use_position']:
                self.anchors_pos_int = init_anchors_pos_int('train_val', self.hparams, self.networkx_graph, self.device, self.train_sub_G, self.val_sub_G, self.test_sub_G) 
                self.anchors_pos_ext = init_anchors_pos_ext(self.hparams, self.networkx_graph, self.device)
            if self.hparams['use_structure']:
                self.anchors_structure = init_anchors_structure(self.hparams,  self.structure_anchors, self.int_structure_anchor_random_walks, self.bor_structure_anchor_random_walks)
            
        self.metric_scores.append(tensorboard_logs) # keep track for optuna

        if self.wandb_epoch_logging and wandb.run is not None:
            wandb_logs = self._wandb_prefix_metrics(tensorboard_logs, 'val')
            wandb.log(wandb_logs, step=self.current_epoch)
        
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def on_save_checkpoint(self, checkpoint: Dict[str, typing.Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["best_val_threshold"] = getattr(self, "best_val_threshold", 0.5)

    def test_epoch_end(self, outputs):
        '''
        Called at end of the test epoch
        '''
        
        # aggregate the logits, labels, and metrics for all batches
        logits = torch.cat([x['test_logits'] for x in outputs], dim=0)
        labels = torch.cat([x['test_labels'] for x in outputs], dim=0)
        pred_threshold = float(getattr(self, 'best_val_threshold', 0.5))
        macro_f1 = subgraph_utils.calc_f1(logits, labels, avg_type='macro', multilabel_binarizer=self.multilabel_binarizer, threshold=pred_threshold).squeeze()
        micro_f1 = subgraph_utils.calc_f1(logits, labels, avg_type='micro', multilabel_binarizer=self.multilabel_binarizer, threshold=pred_threshold).squeeze()
        acc = subgraph_utils.calc_accuracy(logits, labels, multilabel_binarizer=self.multilabel_binarizer, threshold=pred_threshold).squeeze()
        
        # calc AUC/PR
        if self.multilabel:
            auroc = roc_auc_score(labels.cpu(), torch.sigmoid(logits).cpu(), multi_class = 'ovr')
        elif len(torch.unique(labels)) == 2: #binary case
            auroc = roc_auc_score(labels.cpu(), F.softmax(logits, dim=1).cpu()[:,1])
        else: #multiclass
            auroc = roc_auc_score(labels.cpu(), F.softmax(logits, dim=1).cpu(), multi_class = 'ovr')
        auprc = subgraph_utils.calc_auprc(logits, labels, multilabel_binarizer=self.multilabel_binarizer)
        illicit_pr_auc = subgraph_utils.calc_minor_pr_auc(logits, labels, multilabel_binarizer=self.multilabel_binarizer)
        minor_f1 = subgraph_utils.calc_minor_f1(logits, labels, multilabel_binarizer=self.multilabel_binarizer, threshold=pred_threshold).squeeze()

        # get average loss, acc, and macro F1 over batches
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean().cpu()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        avg_macro_f1 = torch.stack([x['test_macro_f1'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss, 'test_micro_f1': micro_f1, 'test_macro_f1': macro_f1, \
            'test_acc': acc, 'avg_test_acc': avg_acc, 'test_avg_macro_f1':avg_macro_f1, 'test_auroc':auroc, \
            'test_auprc': auprc, 'test_minor_f1': minor_f1, 'test_illicit_pr_auc': illicit_pr_auc,
            'test_threshold': pred_threshold }

        self._add_binary_diagnostics(tensorboard_logs, 'test', logits, labels, pred_threshold)

        # add ROC for each class to tensorboard
        if self.multilabel:
            for c in range(logits.shape[1]): #n_classes
                tensorboard_logs['test_auroc_class_' + str(c)] = roc_auc_score(labels[:, c].cpu(), torch.sigmoid(logits)[:, c].cpu())
                tensorboard_logs['test_auprc_class_' + str(c)] = average_precision_score(labels[:, c].cpu(), torch.sigmoid(logits)[:, c].cpu())
        else:
            one_hot_labels = one_hot(labels, num_classes = logits.shape[1])
            softmax_logits = F.softmax(logits, dim=1).cpu()
            for c in range(logits.shape[1]): #n_classes
                tensorboard_logs['test_auroc_class_' + str(c)] = roc_auc_score(one_hot_labels[:, c].cpu(), softmax_logits[:, c].cpu())
                tensorboard_logs['test_auprc_class_' + str(c)] = average_precision_score(one_hot_labels[:, c].cpu(), softmax_logits[:, c].cpu())

        self.test_results = tensorboard_logs

        if self.wandb_epoch_logging and wandb.run is not None:
            wandb_logs = self._wandb_prefix_metrics(tensorboard_logs, 'test')
            wandb.log(wandb_logs, step=self.current_epoch)

        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

##################################################
# Read in data

    def reindex_data(self, data):
        '''
        Relabel node indices in the train/val/test sets to be 1-indexed instead of 0 indexed
        so that we can use 0 for padding
        '''
        new_subg = []
        for subg in data:
            new_subg.append([c + 1 for c in subg])
        return new_subg

    def _resolve_path(self, path_value):
        if path_value is None:
            return None
        path = Path(path_value)
        if path.is_absolute():
            return path
        return config.PROJECT_ROOT / path

    def _load_embedding_tensor(self, embedding_path: Path, node_order=None):
        suffix = embedding_path.suffix.lower()
        if suffix in {'.pt', '.pth'}:
            loaded = torch.load(str(embedding_path), torch.device('cpu'))
            if isinstance(loaded, dict):
                for key in ['embeddings', 'node_embeddings', 'x']:
                    if key in loaded:
                        loaded = loaded[key]
                        break
                else:
                    raise ValueError(
                        f"Embedding file {embedding_path} is a dict, but none of "
                        "['embeddings', 'node_embeddings', 'x'] were found."
                    )
            embeddings = loaded
        elif suffix == '.npy':
            embeddings = torch.tensor(np.load(str(embedding_path)))
        elif suffix == '.csv':
            df = pd.read_csv(embedding_path)
            if 'clId' in df.columns and node_order is not None:
                df['clId'] = df['clId'].astype(str)
                missing = set(node_order) - set(df['clId'])
                if missing:
                    raise ValueError(
                        f"Embedding CSV {embedding_path} is missing {len(missing)} clId values "
                        "from background_nodes.csv."
                    )
                df = df.set_index('clId').loc[node_order].reset_index()
            cols = [c for c in df.columns if c != 'clId']
            embeddings = torch.tensor(df[cols].values)
        else:
            raise ValueError(
                f"Unsupported embedding file extension for {embedding_path}. "
                "Use .pth, .pt, .npy, or .csv."
            )

        if not torch.is_tensor(embeddings):
            embeddings = torch.tensor(embeddings)
        if embeddings.dim() != 2:
            raise ValueError(f"Expected a 2D embedding matrix, got shape {tuple(embeddings.shape)}.")
        return embeddings.float()

    def _load_aml_global_node_embeddings(self, embedding_path: Path, expected_nodes: int, node_order=None):
        if embedding_path is None:
            raise ValueError(
                "use_global_node_embeddings is true, but no embedding path was configured. "
                "Set hyperparams_fix.global_node_embedding_path or use embedding_type to choose "
                "the default gin_embeddings.pth/graphsaint_gcn_embeddings.pth path."
            )
        if not embedding_path.exists():
            raise FileNotFoundError(
                f"use_global_node_embeddings is true, but {embedding_path} does not exist. "
                "Generate global node embeddings first, with rows in background_nodes.csv order."
            )

        embeddings = self._load_embedding_tensor(embedding_path, node_order=node_order)
        if embeddings.shape[0] == expected_nodes + 1:
            print(
                f"--- Loaded AML global node embeddings from {embedding_path}; "
                "dropping leading padding row ---",
                flush=True,
            )
            embeddings = embeddings[1:]
        if embeddings.shape[0] != expected_nodes:
            raise ValueError(
                f"AML global node embeddings at {embedding_path} have {embeddings.shape[0]} rows, "
                f"but background_nodes.csv has {expected_nodes} rows."
            )

        print(
            f"--- Loaded AML global node embeddings from {embedding_path} "
            f"({embeddings.shape[1]} dims) ---",
            flush=True,
        )
        return embeddings

    def _set_aml_temporal_edge_splits(
        self,
        data_root: Path,
        valid_edges: pd.DataFrame,
        train_cc_ids,
        val_cc_ids,
        train_ratio: float,
        val_ratio: float,
        node_count: int,
    ):
        if "feat0" not in valid_edges.columns:
            return

        edge_time = valid_edges["feat0"].astype(float).reset_index(drop=True)
        component_edges_path = data_root / "component_edges.csv"
        component_edges = None
        if component_edges_path.exists():
            component_edges = pd.read_csv(component_edges_path, usecols=["ccId", "feat0"])
            component_edges["ccId"] = component_edges["ccId"].astype(str)
            component_edges["feat0"] = component_edges["feat0"].astype(float)

        if component_edges is not None and len(component_edges) > 0:
            train_ratio = float(component_edges["ccId"].isin(train_cc_ids).mean())
            val_ratio = float(component_edges["ccId"].isin(val_cc_ids).mean())

        edge_time_np = edge_time.to_numpy()
        total_edges = edge_time_np.shape[0]
        order = np.argsort(edge_time_np, kind="mergesort")
        train_cut = max(0, min(int(train_ratio * total_edges), total_edges))
        val_cut = max(train_cut, min(int(val_ratio * total_edges), total_edges))

        train_np = np.zeros(total_edges, dtype=bool)
        val_np = np.zeros(total_edges, dtype=bool)
        test_np = np.ones(total_edges, dtype=bool)
        if train_cut > 0:
            train_np[order[:train_cut]] = True
        if val_cut > 0:
            val_np[order[:val_cut]] = True

        self.edge_train_mask = torch.from_numpy(train_np)
        self.edge_val_mask = torch.from_numpy(val_np)
        self.edge_test_mask = torch.from_numpy(test_np)

        src = valid_edges["clId1"].map(self._aml_node_mapping).to_numpy(dtype=np.int64)
        dst = valid_edges["clId2"].map(self._aml_node_mapping).to_numpy(dtype=np.int64)
        edge_pairs = list(zip(src.tolist(), dst.tolist()))

        def build_graph(mask_np):
            graph = nx.Graph()
            graph.add_nodes_from(range(node_count))
            graph.add_edges_from([edge_pairs[i] for i in np.flatnonzero(mask_np)])
            return graph

        self.train_networkx_graph = build_graph(train_np)
        self.val_networkx_graph = build_graph(val_np)
        self.test_networkx_graph = build_graph(test_np)

    def _set_empty_aml_edge_features(self):
        self.train_aml_edge_features = None
        self.val_aml_edge_features = None
        self.test_aml_edge_features = None
        self.aml_edge_feature_names = []
        self.hparams['aml_edge_feature_dim'] = 0

    def _edge_summary_from_group(self, group, continuous_cols, categorical_cols):
        n_edges = 0 if group is None else len(group)
        values = [float(np.log1p(n_edges))]

        for col in continuous_cols:
            if n_edges == 0:
                values.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            arr = pd.to_numeric(group[col], errors='coerce').dropna().to_numpy(dtype=float)
            if arr.size == 0:
                values.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                values.extend([
                    float(np.mean(arr)),
                    float(np.std(arr)),
                    float(np.min(arr)),
                    float(np.max(arr)),
                    float(np.sum(arr)),
                ])

        for col in categorical_cols:
            if n_edges == 0:
                values.extend([0.0, 0.0, 0.0])
                continue
            counts = group[col].dropna().value_counts()
            total = float(counts.sum())
            if total == 0:
                values.extend([0.0, 0.0, 0.0])
            else:
                probs = counts.to_numpy(dtype=float) / total
                values.extend([
                    float(len(counts)),
                    float(np.max(probs)),
                    -float(np.sum(probs * np.log(probs + 1e-12))),
                ])
        return values

    def _build_aml_edge_feature_matrix(self, entries, grouped_edges, continuous_cols, categorical_cols):
        rows = [
            self._edge_summary_from_group(grouped_edges.get(entry['cc_id']), continuous_cols, categorical_cols)
            for entry in entries
        ]
        if len(rows) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def _init_aml_edge_features(self, data_root: Path, train_entries, val_entries, test_entries):
        self._set_empty_aml_edge_features()
        if not bool(self.hparams.get('use_aml_edge_features', False)):
            return

        component_edges_csv = data_root / "component_edges.csv"
        if not component_edges_csv.exists():
            print(f"--- AML edge features requested but {component_edges_csv} does not exist; disabling ---", flush=True)
            return

        component_edges_df = pd.read_csv(component_edges_csv)
        if 'ccId' not in component_edges_df.columns:
            print("--- AML edge features requested but component_edges.csv has no ccId column; disabling ---", flush=True)
            return

        component_edges_df['ccId'] = component_edges_df['ccId'].astype(str)
        feat_cols = [c for c in component_edges_df.columns if c.startswith('feat')]
        continuous_cols = [c for c in ['feat0', 'feat1', 'feat4'] if c in feat_cols]
        categorical_cols = [c for c in ['feat2', 'feat3', 'feat5'] if c in feat_cols]
        if len(continuous_cols) == 0 and len(categorical_cols) == 0:
            print("--- AML edge features requested but no feat* columns were found; disabling ---", flush=True)
            return

        grouped_edges = {
            cc_id: group
            for cc_id, group in component_edges_df.groupby('ccId', sort=False)
        }
        train_raw = self._build_aml_edge_feature_matrix(train_entries, grouped_edges, continuous_cols, categorical_cols)
        val_raw = self._build_aml_edge_feature_matrix(val_entries, grouped_edges, continuous_cols, categorical_cols)
        test_raw = self._build_aml_edge_feature_matrix(test_entries, grouped_edges, continuous_cols, categorical_cols)

        if train_raw.size == 0:
            print("--- AML edge features requested but no train edge summaries were created; disabling ---", flush=True)
            return

        mean = train_raw.mean(axis=0, keepdims=True)
        std = train_raw.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0

        self.train_aml_edge_features = torch.tensor((train_raw - mean) / std, dtype=torch.float)
        self.val_aml_edge_features = torch.tensor((val_raw - mean) / std, dtype=torch.float)
        self.test_aml_edge_features = torch.tensor((test_raw - mean) / std, dtype=torch.float)

        feature_names = ['edge_count_log1p']
        for col in continuous_cols:
            feature_names.extend([f'{col}_mean', f'{col}_std', f'{col}_min', f'{col}_max', f'{col}_sum'])
        for col in categorical_cols:
            feature_names.extend([f'{col}_nunique', f'{col}_top_freq', f'{col}_entropy'])
        self.aml_edge_feature_names = feature_names
        self.hparams['aml_edge_feature_dim'] = int(self.train_aml_edge_features.shape[1])
        print(f"--- Loaded AML edge feature summaries ({self.hparams['aml_edge_feature_dim']} dims) ---", flush=True)

    def _read_aml_data(self, data_root: Path):
        nodes_csv = data_root / "background_nodes.csv"
        edges_csv = data_root / "background_edges.csv"
        cc_csv = data_root / "connected_components.csv"
        sub_nodes_csv = data_root / "nodes.csv"

        if not nodes_csv.exists() or not edges_csv.exists() or not cc_csv.exists() or not sub_nodes_csv.exists():
            raise FileNotFoundError(
                f"AML data files not found under {data_root}. Expected: background_nodes.csv, "
                "background_edges.csv, connected_components.csv, nodes.csv"
            )

        bg_nodes_df = pd.read_csv(nodes_csv)
        bg_nodes_df['clId'] = bg_nodes_df['clId'].astype(str)
        node_mapping = {clId: idx for idx, clId in enumerate(bg_nodes_df['clId'])}
        self._aml_node_mapping = node_mapping

        feat_cols = [c for c in bg_nodes_df.columns if c.startswith('feat')]
        if len(feat_cols) == 0:
            bg_nodes_df['feat0'] = 1.0
            feat_cols = ['feat0']

        use_global_node_embeddings = bool(self.hparams.get('use_global_node_embeddings', False))
        if use_global_node_embeddings:
            global_embedding_path = self.hparams.get('global_node_embedding_path', None)
            if global_embedding_path is None:
                global_embedding_path = self.embedding_path
            global_embedding_path = self._resolve_path(global_embedding_path)
            pretrained_node_embeds = self._load_aml_global_node_embeddings(
                global_embedding_path,
                expected_nodes=len(bg_nodes_df),
                node_order=bg_nodes_df['clId'].tolist(),
            )
        else:
            pretrained_node_embeds = torch.tensor(bg_nodes_df[feat_cols].values, dtype=torch.float)

        bg_edges_df = pd.read_csv(edges_csv)
        bg_edges_df['clId1'] = bg_edges_df['clId1'].astype(str)
        bg_edges_df['clId2'] = bg_edges_df['clId2'].astype(str)
        valid_edges = bg_edges_df[bg_edges_df['clId1'].isin(node_mapping) & bg_edges_df['clId2'].isin(node_mapping)]

        self.networkx_graph = nx.Graph()
        self.networkx_graph.add_nodes_from(range(len(node_mapping)))
        self.networkx_graph.add_edges_from(
            zip(
                valid_edges['clId1'].map(node_mapping).to_list(),
                valid_edges['clId2'].map(node_mapping).to_list(),
            )
        )

        cc_df = pd.read_csv(cc_csv)
        cc_df['ccId'] = cc_df['ccId'].astype(str)
        cc_df['ccLabel'] = cc_df['ccLabel'].astype(str)
        unique_labels = sorted(cc_df['ccLabel'].unique())
        label_mapping = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.label_mapping = label_mapping
        self.label_names = unique_labels
        cc_label_dict = {row['ccId']: label_mapping[row['ccLabel']] for _, row in cc_df.iterrows()}

        cc_time_dict = {}
        if 'ccStartTime' in cc_df.columns:
            cc_df['ccStartTime'] = pd.to_datetime(cc_df['ccStartTime'], errors='coerce')
            cc_time_dict = {
                row['ccId']: row['ccStartTime']
                for _, row in cc_df.iterrows()
                if pd.notna(row['ccStartTime'])
            }

        sub_df = pd.read_csv(sub_nodes_csv)
        sub_df['clId'] = sub_df['clId'].astype(str)
        sub_df['ccId'] = sub_df['ccId'].astype(str)

        all_entries = []
        for cc_id, group in sub_df.groupby('ccId'):
            mapped_nodes = [node_mapping[c] for c in group['clId'] if c in node_mapping]
            if len(mapped_nodes) > 0:
                all_entries.append({
                    "cc_id": cc_id,
                    "nodes": mapped_nodes,
                    "label": cc_label_dict.get(cc_id, 0),
                    "time": cc_time_dict.get(cc_id, pd.NaT),
                })

        if len(all_entries) == 0:
            raise ValueError(
                "No AML subgraphs could be mapped to background node IDs. "
                "Check ID type consistency between background_nodes.csv and nodes.csv."
            )

        split_type = str(self.hparams.get('aml_split', 'temporal')).strip().lower().replace('-', '_')
        train_ratio = float(self.hparams.get('aml_train_ratio', 0.7))
        val_ratio = float(self.hparams.get('aml_val_ratio', 0.15))
        seed = int(self.hparams.get('seed', 42))

        if split_type == 'temporal' and any(pd.notna(e['time']) for e in all_entries):
            ordered = sorted(all_entries, key=lambda e: (pd.isna(e['time']), e['time'], e['cc_id']))
            n = len(ordered)
            t_end = int(train_ratio * n)
            v_end = t_end + int(val_ratio * n)
            train_entries = ordered[:t_end]
            val_entries = ordered[t_end:v_end]
            test_entries = ordered[v_end:]
        elif split_type in {'class_temporal', 'classwise_temporal', 'stratified_temporal'}:
            ordered = sorted(all_entries, key=lambda e: (pd.isna(e['time']), e['time'], e['cc_id']))
            train_entries, val_entries, test_entries = [], [], []
            for label in sorted({e['label'] for e in ordered}):
                class_entries = [e for e in ordered if e['label'] == label]
                n = len(class_entries)
                t_end = int(train_ratio * n)
                v_end = t_end + int(val_ratio * n)
                train_entries.extend(class_entries[:t_end])
                val_entries.extend(class_entries[t_end:v_end])
                test_entries.extend(class_entries[v_end:])

            train_entries = sorted(train_entries, key=lambda e: (pd.isna(e['time']), e['time'], e['cc_id']))
            val_entries = sorted(val_entries, key=lambda e: (pd.isna(e['time']), e['time'], e['cc_id']))
            test_entries = sorted(test_entries, key=lambda e: (pd.isna(e['time']), e['time'], e['cc_id']))
        elif split_type == 'random':
            rng = random.Random(seed)
            ordered = list(all_entries)
            rng.shuffle(ordered)
            n = len(ordered)
            t_end = int(train_ratio * n)
            v_end = t_end + int(val_ratio * n)
            train_entries = ordered[:t_end]
            val_entries = ordered[t_end:v_end]
            test_entries = ordered[v_end:]
        else:
            raise ValueError(
                "aml_split must be one of temporal, random, class_temporal, "
                f"or stratified_temporal; got {split_type!r}."
            )

        self._set_aml_temporal_edge_splits(
            data_root,
            valid_edges,
            {e['cc_id'] for e in train_entries},
            {e['cc_id'] for e in train_entries} | {e['cc_id'] for e in val_entries},
            train_ratio,
            train_ratio + val_ratio,
            len(node_mapping),
        )

        self.train_sub_G = [e['nodes'] for e in train_entries]
        self.val_sub_G = [e['nodes'] for e in val_entries]
        self.test_sub_G = [e['nodes'] for e in test_entries]

        self.train_sub_G_label = torch.tensor([e['label'] for e in train_entries]).long()
        self.val_sub_G_label = torch.tensor([e['label'] for e in val_entries]).long()
        self.test_sub_G_label = torch.tensor([e['label'] for e in test_entries]).long()

        self._init_aml_edge_features(data_root, train_entries, val_entries, test_entries)

        self.multilabel = False
        self.multilabel_binarizer = None

        return pretrained_node_embeds

    def read_data(self):
        '''
        Read in the subgraphs & their associated labels
        '''

        graph_path = self._resolve_path(self.graph_path)
        subgraph_path = self._resolve_path(self.subgraph_path)
        embedding_path = self._resolve_path(self.embedding_path)

        data_format = self.hparams.get('data_format')
        pretrained_node_embeds = None

        if data_format in ['ibm_aml', 'aml'] or (graph_path is not None and not graph_path.exists()):
            data_root = graph_path.parent if graph_path is not None else config.PROJECT_ROOT
            if (data_root / "background_edges.csv").exists():
                pretrained_node_embeds = self._read_aml_data(data_root)
            else:
                raise FileNotFoundError(f"Graph path not found: {graph_path}")
        else:
            # read networkx graph from edge list
            self.networkx_graph = nx.read_edgelist(graph_path)

            # readin list of node ids for each subgraph & their labels
            self.train_sub_G, self.train_sub_G_label, self.val_sub_G, \
                self.val_sub_G_label, self.test_sub_G, self.test_sub_G_label \
                = subgraph_utils.read_subgraphs(subgraph_path)

        # check if the dataset is multilabel (e.g. HPO-NEURO)
        if type(self.train_sub_G_label) == list: 
            self.multilabel=True
            all_labels = self.train_sub_G_label + self.val_sub_G_label + self.test_sub_G_label
            self.multilabel_binarizer = MultiLabelBinarizer().fit(all_labels)
        else: 
            self.multilabel = False
            self.multilabel_binarizer = None
        
        # Optionally subset the data for debugging purposes to the batch size
        if 'subset_data' in self.hparams and self.hparams['subset_data']:
            print("****WARNING: SUBSETTING DATA*****")
            self.train_sub_G, self.train_sub_G_label, self.val_sub_G, \
                self.val_sub_G_label, self.test_sub_G, self.test_sub_G_label = self.train_sub_G[0:self.hparams['batch_size']], self.train_sub_G_label[0:self.hparams['batch_size']], self.val_sub_G[0:self.hparams['batch_size']], \
                self.val_sub_G_label[0:self.hparams['batch_size']], self.test_sub_G[0:self.hparams['batch_size']], self.test_sub_G_label[0:self.hparams['batch_size']]

        # get the number of classes for prediction
        if type(self.train_sub_G_label) == list: # if multi-label
            self.num_classes = max([max(l) for l in self.train_sub_G_label + self.val_sub_G_label + self.test_sub_G_label]) + 1
        else:
            self.num_classes = int(torch.max(torch.cat((self.train_sub_G_label, self.val_sub_G_label, self.test_sub_G_label)))) + 1

        # Derive class distribution / weights for imbalanced classification (single-label only).
        # These are consumed by CrossEntropyLoss when `auto_class_weights` is enabled.
        self.class_counts = None
        self.class_weights = None
        if (not self.multilabel) and isinstance(self.train_sub_G_label, torch.Tensor):
            try:
                counts = torch.bincount(self.train_sub_G_label.view(-1).cpu(), minlength=int(self.num_classes)).float()
                self.class_counts = counts

                # "balanced" weights: w_c = N / (C * n_c)
                denom = counts.clone()
                denom[denom == 0] = 1.0
                weights = counts.sum() / (float(self.num_classes) * denom)

                max_w = float(self.hparams.get('max_class_weight', 100.0))
                self.class_weights = torch.clamp(weights, min=0.0, max=max_w)
            except Exception:
                self.class_counts = None
                self.class_weights = None

        # renumber nodes to start with index 1 instead of 0
        mapping = {n:int(n)+1 for n in self.networkx_graph.nodes()}
        self.networkx_graph = nx.relabel_nodes(self.networkx_graph, mapping)
        self.train_sub_G = self.reindex_data(self.train_sub_G)
        self.val_sub_G = self.reindex_data(self.val_sub_G)
        self.test_sub_G = self.reindex_data(self.test_sub_G)

        # Initialize pretrained node embeddings
        if pretrained_node_embeds is None:
            if embedding_path is None:
                raise ValueError("embedding_path is not set and no AML features were loaded.")
            pretrained_node_embeds = torch.load(embedding_path, torch.device('cpu')) # feature matrix should be initialized to the node embeddings
        self.hparams['node_embed_size'] = pretrained_node_embeds.shape[1]
        zeros = torch.zeros(1, pretrained_node_embeds.shape[1])
        embeds = torch.cat((zeros, pretrained_node_embeds), 0) #there's a zeros in the first index for padding
        
        # optionally freeze the node embeddings
        self.node_embeddings = nn.Embedding.from_pretrained(embeds, freeze=self.hparams['freeze_node_embeds'], padding_idx=config.PAD_VALUE).to(self.device)

        print('--- Finished reading in data ---')

##################################################
# Initialize connected components & associated embeddings for each channel in SubGNN

    def initialize_cc_ids(self, subgraph_ids):
        '''
        Initialize the 3D matrix of (n_subgraphs X max number of cc X max length of cc)

        Input:
            - subgraph_ids: list of subgraphs where each subgraph is a list of node ids 

        Output:
            - reshaped_cc_ids_pad: padded tensor of shape (n_subgraphs, max_n_cc, max_len_cc)
        '''
        n_subgraphs = len(subgraph_ids) # number of subgraphs

        # Create connected component ID list from subgraphs
        cc_id_list = []
        for curr_subgraph_ids in subgraph_ids:
            subgraph = nx.subgraph(self.networkx_graph, curr_subgraph_ids) #networkx version of subgraph
            con_components = list(nx.connected_components(subgraph)) # get connected components in subgraph
            cc_id_list.append([torch.LongTensor(list(cc_ids)) for cc_ids in con_components])

        # pad number of connected components
        max_n_cc = max([len(cc) for cc in cc_id_list]) #max number of cc across all subgraphs
        for cc_list in cc_id_list:
            while True:
                if len(cc_list) == max_n_cc: break
                cc_list.append(torch.LongTensor([config.PAD_VALUE]))

        # pad number of nodes in connected components
        all_pad_cc_ids = [cc for cc_list in cc_id_list for cc in cc_list]
        assert len(all_pad_cc_ids) % max_n_cc == 0
        con_component_ids_pad = pad_sequence(all_pad_cc_ids, batch_first=True, padding_value=config.PAD_VALUE) # (batch_sz * max_n_cc, max_cc_len)
        reshaped_cc_ids_pad = con_component_ids_pad.view(n_subgraphs, max_n_cc, -1) # (batch_sz, max_n_cc, max_cc_len)

        return reshaped_cc_ids_pad # (n_subgraphs, max_n_cc, max_len_cc)

    def initialize_cc_embeddings(self, cc_id_list, aggregator='sum'):
        '''
        Initialize connected component embeddings as either the sum or max of node embeddings in the connected component

        Input:
            - cc_id_list: 3D tensor of shape (n subgraphs, max n CC, max length CC)

        Output:
            - 3D tensor of shape (n_subgraphs, max n_cc, node embedding dim)
        '''
        if aggregator == 'sum':
            return torch.sum(self.node_embeddings(cc_id_list.to(self.device)), dim=2)
        elif aggregator == 'max':
            return torch.max(self.node_embeddings(cc_id_list.to(self.device)), dim=2)[0]

    def initialize_channel_embeddings(self, cc_embeddings, trainable = False):
        '''
        Initialize CC embeddings for each channel (N, S, P X internal, border)
        '''

        if trainable: # if the embeddings are trainable, make them a parameter
            N_I_cc_embeds = Parameter(cc_embeddings.detach().clone())
            N_B_cc_embeds = Parameter(cc_embeddings.detach().clone())
            S_I_cc_embeds = Parameter(cc_embeddings.detach().clone())
            S_B_cc_embeds = Parameter(cc_embeddings.detach().clone())
            P_I_cc_embeds = Parameter(cc_embeddings.detach().clone())
            P_B_cc_embeds = Parameter(cc_embeddings.detach().clone())
        else:
            N_I_cc_embeds = cc_embeddings
            N_B_cc_embeds = cc_embeddings
            S_I_cc_embeds = cc_embeddings
            S_B_cc_embeds = cc_embeddings
            P_I_cc_embeds = cc_embeddings
            P_B_cc_embeds = cc_embeddings

        return (N_I_cc_embeds, N_B_cc_embeds, S_I_cc_embeds, S_B_cc_embeds, P_I_cc_embeds, P_B_cc_embeds)

    def init_all_embeddings(self, split = 'all', trainable = False):
        '''
        Initialize the CC and channel-specific CC embeddings for the subgraphs in the specified split
         ('all', 'train_val', 'train', 'val', or 'test')
        '''
        if split in ['all','train_val','train']:
            # initialize CC embeddings
            train_cc_embeddings = self.initialize_cc_embeddings(self.train_cc_ids, self.hparams['cc_aggregator'])
            
            # initialize  CC embeddings for each channel
            self.train_N_I_cc_embed, self.train_N_B_cc_embed, self.train_S_I_cc_embed, \
                self.train_S_B_cc_embed, self.train_P_I_cc_embed, self.train_P_B_cc_embed \
                    = self.initialize_channel_embeddings(train_cc_embeddings, trainable)
        if split in ['all','train_val','val']:
            val_cc_embeddings = self.initialize_cc_embeddings( self.val_cc_ids, self.hparams['cc_aggregator'])
            self.val_N_I_cc_embed, self.val_N_B_cc_embed, self.val_S_I_cc_embed, \
                self.val_S_B_cc_embed, self.val_P_I_cc_embed, self.val_P_B_cc_embed \
                    = self.initialize_channel_embeddings(val_cc_embeddings, trainable=False)
        if split in ['all','test']:
            test_cc_embeddings = self.initialize_cc_embeddings( self.test_cc_ids, self.hparams['cc_aggregator'])
            self.test_N_I_cc_embed, self.test_N_B_cc_embed, self.test_S_I_cc_embed, \
                self.test_S_B_cc_embed, self.test_P_I_cc_embed, self.test_P_B_cc_embed \
                    = self.initialize_channel_embeddings(test_cc_embeddings, trainable=False)

##################################################
# Initialize node border sets surrounding each CC for each subgraph

    def initialize_border_sets(self, fname, cc_ids, radius, ego_graph_dict=None):
        '''
        Creates and saves to file a matrix containing the node ids in the k-hop border set of each CC for each subgraph
        The shape of the resulting matrix, which is padded to the max border set size, is (n_subgraphs, max_n_cc, max_border_set_sz)
        '''
        n_subgraphs, max_n_cc, _ = cc_ids.shape
        all_border_sets = []

        # for each component in each subgraph, calculate the k-hop node border of the connected component
        for s, subgraph in enumerate(cc_ids):
            border_sets = []
            for c, component in enumerate(subgraph):
                # radius specifies the size of the border set - i.e. the k number of hops away the node can be from any node in the component to be in the border set 
                component_border = subgraph_utils.get_component_border_neighborhood_set(self.networkx_graph, component, radius, ego_graph_dict)
                border_sets.append(component_border)
            all_border_sets.append(border_sets)

        #fill in matrix with padding
        max_border_set_len = max([len(s) for l in all_border_sets for s in l])
        border_set_matrix = torch.zeros((n_subgraphs, max_n_cc, max_border_set_len), dtype=torch.long).fill_(config.PAD_VALUE)
        for s, subgraph in enumerate(all_border_sets):
            for c,component in enumerate(subgraph):
                fill_len = max_border_set_len - len(component)
                border_set_matrix[s,c,:] = torch.cat([torch.LongTensor(list(component)),torch.LongTensor((fill_len)).fill_(config.PAD_VALUE)])
        
        # save border set to file 
        np.save(fname, border_set_matrix.cpu().numpy())
        return border_set_matrix # n_subgraphs, max_n_cc, max_border_set_sz

    def get_border_sets(self, split):
        '''
            Returns the node ids in the k-hop border of each subgraph (where k = neigh_sample_border_size) for the train, val, and test subgraphs
        '''

        # location where similarities are stored
        sim_path = self._resolve_path(self.similarities_path)
        if sim_path is not None and not sim_path.exists():
            sim_path.mkdir(parents=True, exist_ok=True)
        
        self.train_P_border = None
        self.val_P_border = None
        self.test_P_border = None

        # We need the border sets if we're using the neighborhood channel or if we're using the edit distance similarity function in the structure channel
        if self.hparams['use_neighborhood'] or (self.hparams['use_structure'] and self.hparams['structure_similarity_fn'] == 'edit_distance'):
            
            # load ego graphs dictionary
            ego_graph_path = self._resolve_path(self.ego_graph_path)
            if ego_graph_path is not None and ego_graph_path.exists():
                with open(str(ego_graph_path), 'r') as f:
                    ego_graph_dict = json.load(f)
                ego_graph_dict = {int(key): value for key, value in ego_graph_dict.items()}
            else: ego_graph_dict = None

            # either load in the border sets from file or recompute the border sets
            train_neigh_path = sim_path / (str(self.hparams["neigh_sample_border_size"]) + '_' + str(config.PAD_VALUE) + '_train_border_set.npy') 
            val_neigh_path = sim_path / (str(self.hparams["neigh_sample_border_size"]) + '_' + str(config.PAD_VALUE) + '_val_border_set.npy') 
            test_neigh_path = sim_path / (str(self.hparams["neigh_sample_border_size"]) + '_' + str(config.PAD_VALUE) + '_test_border_set.npy')
            if split == 'test':
                expected = int(self.test_cc_ids.shape[0])
                if test_neigh_path.exists() and not self.hparams['compute_similarities']:
                    loaded = torch.tensor(np.load(test_neigh_path, allow_pickle=True))
                    if loaded.shape[0] != expected:
                        print(f"--- Border cache mismatch for test (got {loaded.shape[0]}, expected {expected}); recomputing ---", flush=True)
                        self.test_N_border = self.initialize_border_sets(test_neigh_path, self.test_cc_ids, self.hparams["neigh_sample_border_size"], ego_graph_dict)
                    else:
                        self.test_N_border = loaded
                else:
                    self.test_N_border = self.initialize_border_sets(test_neigh_path, self.test_cc_ids, self.hparams["neigh_sample_border_size"], ego_graph_dict)
            elif split == 'train_val':
                expected_train = int(self.train_cc_ids.shape[0])
                expected_val = int(self.val_cc_ids.shape[0])

                if train_neigh_path.exists() and not self.hparams['compute_similarities']:
                    loaded = torch.tensor(np.load(train_neigh_path, allow_pickle=True))
                    if loaded.shape[0] != expected_train:
                        print(f"--- Border cache mismatch for train (got {loaded.shape[0]}, expected {expected_train}); recomputing ---", flush=True)
                        self.train_N_border = self.initialize_border_sets(train_neigh_path, self.train_cc_ids, self.hparams["neigh_sample_border_size"], ego_graph_dict)
                    else:
                        self.train_N_border = loaded
                else:
                    self.train_N_border = self.initialize_border_sets(train_neigh_path, self.train_cc_ids,  self.hparams["neigh_sample_border_size"], ego_graph_dict)

                if val_neigh_path.exists() and not self.hparams['compute_similarities']:
                    loaded = torch.tensor(np.load(val_neigh_path, allow_pickle=True))
                    if loaded.shape[0] != expected_val:
                        print(f"--- Border cache mismatch for val (got {loaded.shape[0]}, expected {expected_val}); recomputing ---", flush=True)
                        self.val_N_border = self.initialize_border_sets(val_neigh_path, self.val_cc_ids, self.hparams["neigh_sample_border_size"], ego_graph_dict)
                    else:
                        self.val_N_border = loaded
                else:
                    self.val_N_border = self.initialize_border_sets(val_neigh_path, self.val_cc_ids, self.hparams["neigh_sample_border_size"], ego_graph_dict)
         
        else: # otherwise, we can just set these to None
            self.train_N_border = None
            self.val_N_border = None
            self.test_N_border = None

##################################################
# Compute similarities between the anchor patches & the subgraphs

    def compute_shortest_path_similarities(self, fname, shortest_paths, cc_ids):
        '''
        Creates a similarity matrix with shape (n_subgraphs, max num cc, number of nodes in graph) that stores the shortest 
        path between each cc (for each subgraph) and all nodes in the graph. 
        '''

        print('---- Precomputing Shortest Path Similarities ----')
        n_subgraphs, max_n_cc, _ = cc_ids.shape
        n_nodes_in_graph = len(self.networkx_graph.nodes()) #get number of nodes in the underlying base graph

        cc_id_mask = (cc_ids[:,:,0] != config.PAD_VALUE)
        similarities = torch.zeros((n_subgraphs, max_n_cc, n_nodes_in_graph)) \
            .fill_(config.PAD_VALUE)
        
        #NOTE: could use multiprocessing to speed up this calculation
        for s, subgraph in enumerate(cc_ids):
            for c, component in enumerate(subgraph):
                non_padded_component = component[component != config.PAD_VALUE].cpu().numpy() #remove padding
                if len(non_padded_component) > 0:
                    # NOTE: indexing is off by 1 bc node ids are indexed starting at 1
                    similarities[s,c,:] = torch.tensor(np.min(shortest_paths[non_padded_component - 1,:], axis=0))

        
        # add padding (because each subgraph has variable # CC) & save to file
        if not fname.parent.exists(): fname.parent.mkdir(parents=True)
        print('---- Saving Shortest Path Similarities ----')
        similarities[~cc_id_mask] = config.PAD_VALUE 
        np.save(fname, similarities.cpu().numpy())

        return similarities

    def compute_structure_patch_similarities(self, degree_dict, fname, internal, cc_ids, sim_path, dataset_type, border_set=None):
        '''
        Calculate the similarity between the sampled anchor patches and the connected components

        The default structure similarity function is DTW over the patch and component degree sequences.

        Returns tensor of similarities of shape (n_subgraphs, max_n_cc, n anchor patches)
        '''

        print('---Computing Structure Patch Similarities---')
        n_anchors = self.structure_anchors.shape[0]
        n_subgraphs, max_n_cc, _ = cc_ids.shape
        cc_id_mask = (cc_ids[:,:,0] != config.PAD_VALUE)

        # the default structure similarity function is dynamic time warping (DTW) over the degree sequences of the anchor patches & connected components
        if self.hparams['structure_similarity_fn'] == 'dtw':

            # store the degree sequence for each anchor patch into a dict
            anchor_degree_seq_dict = {}
            for a, anchor_patch in enumerate(self.structure_anchors):
                anchor_degree_seq_dict[a] = gamma.get_degree_sequence(self.networkx_graph, anchor_patch, degree_dict, internal=internal)

            # Store degree sequences only for real connected components. Padding can
            # dominate SAML-D tensors and should not generate DTW jobs.
            component_degree_seq_dict = {}
            cc_ids_reshaped = cc_ids.view(n_subgraphs*max_n_cc, -1)
            valid_component_indices = []
            for c, component in enumerate(cc_ids_reshaped):
                if component[0] == config.PAD_VALUE:
                    continue
                component_degree_seq_dict[c] = gamma.get_degree_sequence(
                    self.networkx_graph, component, degree_dict, internal=internal)
                valid_component_indices.append(c)

            input_iter = (
                (c, a, component_degree_seq_dict[c], anchor_degree_seq_dict[a])
                for c in valid_component_indices
                for a in range(n_anchors)
            )

            n_processes = max(1, min(int(self.hparams['n_processes']), os.cpu_count() or 1))
            similarities_np = np.full(
                (n_subgraphs * max_n_cc, n_anchors),
                config.PAD_VALUE,
                dtype=np.float32,
            )
            if n_processes == 1:
                for flat_idx, anchor_idx, score in map(_calc_dtw_indexed, input_iter):
                    similarities_np[flat_idx, anchor_idx] = score
            else:
                with multiprocessing.Pool(processes=n_processes) as pool:
                    for flat_idx, anchor_idx, score in pool.imap(
                        _calc_dtw_indexed, input_iter, chunksize=1000):
                        similarities_np[flat_idx, anchor_idx] = score

            # reshape similarities to a matrix of shape (n_subgraphs, max_n_cc, n anchor patches)
            similarities = torch.from_numpy(
                similarities_np.reshape(n_subgraphs, max_n_cc, n_anchors))

        else:
            # other structure similarity functions can be added here
            raise NotImplementedError
        
        # add padding & save to file
        print('---- Saving Similarities ----')
        if not fname.parent.exists(): fname.parent.mkdir(parents=True)
        similarities[~cc_id_mask] = config.PAD_VALUE
        np.save(fname, similarities.cpu().numpy())
        return similarities

    def get_similarities(self, split):
        '''
        For the N/P channels: precomputes the shortest paths between all connected components (for all subgraphs) and all nodes in the graph
        For the S channel: precomputes structure anchor patches & random walks as well as structure similarity calculations between the anchor patches and all connected components
        '''
        # path where similarities are stored
        sim_path = self._resolve_path(self.similarities_path)
        if sim_path is not None and not sim_path.exists():
            sim_path.mkdir(parents=True, exist_ok=True)
     
        # If we're using the position or neighborhood channels, we need to calculate the relevant shortest path similarities
        if self.hparams['use_position'] or self.hparams['use_neighborhood']:

            # read in precomputed shortest paths between all nodes in the graph
            pairwise_shortest_paths_path = self._resolve_path(self.shortest_paths_path)
            pairwise_shortest_paths = np.load(pairwise_shortest_paths_path, allow_pickle=True)

            # Read in precomputed similarities if they exist. If they don't, calculate them
            
            train_np_path = sim_path / (str(config.PAD_VALUE) + '_train_similarities.npy') 
            val_np_path = sim_path / (str(config.PAD_VALUE) + '_val_similarities.npy') 
            test_np_path = sim_path / (str(config.PAD_VALUE) + '_test_similarities.npy')

            if split == 'test':
                expected = int(self.test_cc_ids.shape[0])
                if test_np_path.exists() and not self.hparams['compute_similarities']:
                    print('--- Loading Position Similarities from File ---')
                    loaded = torch.tensor(np.load(test_np_path, allow_pickle=True))
                    if loaded.shape[0] != expected:
                        print(f"--- NP cache mismatch for test (got {loaded.shape[0]}, expected {expected}); recomputing ---", flush=True)
                        self.test_neigh_pos_similarities = self.compute_shortest_path_similarities(test_np_path, pairwise_shortest_paths, self.test_cc_ids)
                    else:
                        self.test_neigh_pos_similarities = loaded
                else:
                    self.test_neigh_pos_similarities = self.compute_shortest_path_similarities(test_np_path, pairwise_shortest_paths, self.test_cc_ids)
            elif split == 'train_val':
                expected_train = int(self.train_cc_ids.shape[0])
                expected_val = int(self.val_cc_ids.shape[0])
                if train_np_path.exists() and not self.hparams['compute_similarities']:
                    print('--- Loading Train Position Similarities from File ---')
                    loaded = torch.tensor(np.load(train_np_path, allow_pickle=True))
                    if loaded.shape[0] != expected_train:
                        print(f"--- NP cache mismatch for train (got {loaded.shape[0]}, expected {expected_train}); recomputing ---", flush=True)
                        self.train_neigh_pos_similarities = self.compute_shortest_path_similarities(train_np_path, pairwise_shortest_paths, self.train_cc_ids)
                    else:
                        self.train_neigh_pos_similarities = loaded
                else:
                    self.train_neigh_pos_similarities = self.compute_shortest_path_similarities(train_np_path, pairwise_shortest_paths, self.train_cc_ids)

                if val_np_path.exists() and not self.hparams['compute_similarities']:
                    print('--- Loading Val Position Similarities from File ---')
                    loaded = torch.tensor(np.load(val_np_path, allow_pickle=True))
                    if loaded.shape[0] != expected_val:
                        print(f"--- NP cache mismatch for val (got {loaded.shape[0]}, expected {expected_val}); recomputing ---", flush=True)
                        self.val_neigh_pos_similarities = self.compute_shortest_path_similarities(val_np_path, pairwise_shortest_paths, self.val_cc_ids)
                    else:
                        self.val_neigh_pos_similarities = loaded
                else:
                    self.val_neigh_pos_similarities = self.compute_shortest_path_similarities(val_np_path, pairwise_shortest_paths, self.val_cc_ids)
        else: # if we're only using the structure channel, we can just set these to None
            self.train_neigh_pos_similarities = None
            self.val_neigh_pos_similarities  = None
            self.test_neigh_pos_similarities = None
        
        if self.hparams['use_structure']:
    
            # load in degree dictionary {node id: degree}
            degree_path = self._resolve_path(self.degree_dict_path)
            if degree_path is not None and degree_path.exists():
                with open(str(degree_path), 'r') as f:
                    degree_dict = json.load(f)
                degree_dict = {int(key): value for key, value in degree_dict.items()}
            else: degree_dict = None

            # (1) sample structure anchor patches
            # sample walk len: length of the random walk used to sample the anchor patches
            # structure_patch_type: either 'triangular_random_walk' (default) or 'ego_graph'
            # MAX_SIM_EPOCHS: 
            struc_anchor_patches_path = sim_path / ('struc_patches_' + str(self.hparams['sample_walk_len']) +  '_' + self.hparams['structure_patch_type'] + '_' + str(self.hparams['max_sim_epochs']) + '.npy') 

            if struc_anchor_patches_path.exists() and not self.hparams['compute_similarities']:
                self.structure_anchors = torch.tensor(np.load(struc_anchor_patches_path, allow_pickle=True))
            else:
                self.structure_anchors = sample_structure_anchor_patches(self.hparams, self.networkx_graph, self.device, self.hparams['max_sim_epochs'])
                np.save(struc_anchor_patches_path, self.structure_anchors.cpu().numpy())

            # (2) perform internal and border random walks over sampled anchor patches

            #border
            bor_struc_patch_random_walks_path = sim_path / ('bor_struc_patch_random_walks_' + str(self.hparams['n_triangular_walks']) +  '_' + str(self.hparams['random_walk_len']) +  '_' + str(self.hparams['sample_walk_len']) +  '_' + self.hparams['structure_patch_type'] + '_' + str(self.hparams['max_sim_epochs']) + '.npy')

            if bor_struc_patch_random_walks_path.exists() and not self.hparams['compute_similarities']:
                self.bor_structure_anchor_random_walks = torch.tensor(np.load(bor_struc_patch_random_walks_path, allow_pickle=True))#.to(self.device)
            else:
                self.bor_structure_anchor_random_walks = perform_random_walks(self.hparams, self.networkx_graph, self.structure_anchors, inside=False)
                np.save(bor_struc_patch_random_walks_path, self.bor_structure_anchor_random_walks.cpu().numpy())

            #internal
            int_struc_patch_random_walks_path = sim_path / ('int_struc_patch_random_walks_' + str(self.hparams['n_triangular_walks']) +  '_' + str(self.hparams['random_walk_len']) +  '_' + str(self.hparams['sample_walk_len']) +  '_' + self.hparams['structure_patch_type'] + '_' + str(self.hparams['max_sim_epochs']) + '.npy') 

            if int_struc_patch_random_walks_path.exists() and not self.hparams['compute_similarities']:
                self.int_structure_anchor_random_walks = torch.tensor(np.load(int_struc_patch_random_walks_path, allow_pickle=True))#.to(self.device)
            else:
                self.int_structure_anchor_random_walks = perform_random_walks(self.hparams, self.networkx_graph, self.structure_anchors, inside=True)
                np.save(int_struc_patch_random_walks_path, self.int_structure_anchor_random_walks.cpu().numpy())


            # (3) calculate similarities between anchor patches and connected components

            # filenames where outputs will be stored
            struc_sim_type_fname = '_' + self.hparams['structure_similarity_fn']  if self.hparams['structure_similarity_fn'] != 'dtw' else '' #we only add info about the structure similarity function to the filename if it's not the default dtw
            train_int_struc_path = sim_path /  ('int_struc_' + str(self.hparams['sample_walk_len']) + '_' + self.hparams['structure_patch_type'] + '_' + str(self.hparams['max_sim_epochs'])  + '_'  + str(config.PAD_VALUE) +  struc_sim_type_fname + '_train_similarities.npy') 
            val_int_struc_path = sim_path / ('int_struc_' + str(self.hparams['sample_walk_len']) + '_'  + self.hparams['structure_patch_type'] + '_' + str(self.hparams['max_sim_epochs'])  + '_' + str(config.PAD_VALUE) + struc_sim_type_fname + '_val_similarities.npy')
            test_int_struc_path = sim_path / ('int_struc_' + str(self.hparams['sample_walk_len']) + '_'  + self.hparams['structure_patch_type'] + '_' + str(self.hparams['max_sim_epochs'])  + '_' + str(config.PAD_VALUE) + struc_sim_type_fname + '_test_similarities.npy')
            train_bor_struc_path = sim_path /  ('bor_struc_' + str(self.hparams['sample_walk_len']) + '_'  + self.hparams['structure_patch_type'] + '_' + str(self.hparams['max_sim_epochs'])  + '_' + str(config.PAD_VALUE) + struc_sim_type_fname + '_train_similarities.npy')
            val_bor_struc_path = sim_path / ('bor_struc_' + str(self.hparams['sample_walk_len']) + '_'  + self.hparams['structure_patch_type'] + '_' + str(self.hparams['max_sim_epochs']) + '_' + str(config.PAD_VALUE) + struc_sim_type_fname + '_val_similarities.npy')
            test_bor_struc_path = sim_path / ('bor_struc_' + str(self.hparams['sample_walk_len']) + '_'  + self.hparams['structure_patch_type'] + '_' + str(self.hparams['max_sim_epochs'])  + '_' + str(config.PAD_VALUE) + struc_sim_type_fname + '_test_similarities.npy')
            


            if split == 'test':
                expected = int(self.test_cc_ids.shape[0])
                if test_int_struc_path.exists() and not self.hparams['compute_similarities']:
                    print('--- Loading Test Structure Similarities from File ---', flush=True)
                    loaded = torch.tensor(np.load(test_int_struc_path, allow_pickle=True))
                    if loaded.shape[0] != expected:
                        print(f"--- Int-structure cache mismatch for test (got {loaded.shape[0]}, expected {expected}); recomputing ---", flush=True)
                        self.test_int_struc_similarities = self.compute_structure_patch_similarities(degree_dict, test_int_struc_path, True, self.test_cc_ids, sim_path, 'test', self.test_N_border)
                    else:
                        self.test_int_struc_similarities = loaded
                else:
                    self.test_int_struc_similarities = self.compute_structure_patch_similarities(degree_dict, test_int_struc_path, True, self.test_cc_ids, sim_path, 'test', self.test_N_border)

            elif split == 'train_val':
                expected_train = int(self.train_cc_ids.shape[0])
                expected_val = int(self.val_cc_ids.shape[0])
                if train_int_struc_path.exists() and not self.hparams['compute_similarities']:
                    print('--- Loading Train Structure Similarities from File ---', flush=True)
                    loaded = torch.tensor(np.load(train_int_struc_path, allow_pickle=True))
                    if loaded.shape[0] != expected_train:
                        print(f"--- Int-structure cache mismatch for train (got {loaded.shape[0]}, expected {expected_train}); recomputing ---", flush=True)
                        self.train_int_struc_similarities = self.compute_structure_patch_similarities(degree_dict, train_int_struc_path, True, self.train_cc_ids, sim_path, 'train', self.train_N_border)
                    else:
                        self.train_int_struc_similarities = loaded
                else:
                    self.train_int_struc_similarities = self.compute_structure_patch_similarities(degree_dict, train_int_struc_path, True, self.train_cc_ids, sim_path, 'train', self.train_N_border)

                if val_int_struc_path.exists() and not self.hparams['compute_similarities']:
                    print('--- Loading Val Structure Similarities from File ---', flush=True)
                    loaded = torch.tensor(np.load(val_int_struc_path, allow_pickle=True))
                    if loaded.shape[0] != expected_val:
                        print(f"--- Int-structure cache mismatch for val (got {loaded.shape[0]}, expected {expected_val}); recomputing ---", flush=True)
                        self.val_int_struc_similarities = self.compute_structure_patch_similarities(degree_dict, val_int_struc_path, True, self.val_cc_ids, sim_path, 'val', self.val_N_border)
                    else:
                        self.val_int_struc_similarities = loaded
                else:
                    self.val_int_struc_similarities = self.compute_structure_patch_similarities(degree_dict, val_int_struc_path, True, self.val_cc_ids, sim_path, 'val', self.val_N_border)

            print('Done computing internal structure similarities', flush=True)

            # read in structure similarities
            print('computing border structure sims')

            if split == 'test':
                expected = int(self.test_cc_ids.shape[0])
                if test_bor_struc_path.exists() and not self.hparams['compute_similarities']:
                    print('--- Loading Test Structure Similarities from File ---')
                    loaded = torch.tensor(np.load(test_bor_struc_path, allow_pickle=True))
                    if loaded.shape[0] != expected:
                        print(f"--- Bor-structure cache mismatch for test (got {loaded.shape[0]}, expected {expected}); recomputing ---", flush=True)
                        self.test_bor_struc_similarities = self.compute_structure_patch_similarities(degree_dict, test_bor_struc_path, False, self.test_cc_ids, sim_path, 'test', self.test_N_border)
                    else:
                        self.test_bor_struc_similarities = loaded
                else:
                    self.test_bor_struc_similarities = self.compute_structure_patch_similarities(degree_dict, test_bor_struc_path, False, self.test_cc_ids, sim_path, 'test', self.test_N_border)

            if split == 'train_val':
                expected_train = int(self.train_cc_ids.shape[0])
                expected_val = int(self.val_cc_ids.shape[0])
                if train_bor_struc_path.exists() and not self.hparams['compute_similarities']:
                    print('--- Loading Train Structure Similarities from File ---')
                    loaded = torch.tensor(np.load(train_bor_struc_path, allow_pickle=True))
                    if loaded.shape[0] != expected_train:
                        print(f"--- Bor-structure cache mismatch for train (got {loaded.shape[0]}, expected {expected_train}); recomputing ---", flush=True)
                        self.train_bor_struc_similarities = self.compute_structure_patch_similarities(degree_dict, train_bor_struc_path, False, self.train_cc_ids, sim_path,'train', self.train_N_border)
                    else:
                        self.train_bor_struc_similarities = loaded
                else:
                    self.train_bor_struc_similarities = self.compute_structure_patch_similarities(degree_dict, train_bor_struc_path, False, self.train_cc_ids, sim_path,'train', self.train_N_border)

                if val_bor_struc_path.exists() and not self.hparams['compute_similarities']:
                    print('--- Loading Val Structure Similarities from File ---')
                    loaded = torch.tensor(np.load(val_bor_struc_path, allow_pickle=True))
                    if loaded.shape[0] != expected_val:
                        print(f"--- Bor-structure cache mismatch for val (got {loaded.shape[0]}, expected {expected_val}); recomputing ---", flush=True)
                        self.val_bor_struc_similarities = self.compute_structure_patch_similarities(degree_dict, val_bor_struc_path, False, self.val_cc_ids, sim_path, 'val', self.val_N_border)
                    else:
                        self.val_bor_struc_similarities = loaded
                else:
                    self.val_bor_struc_similarities = self.compute_structure_patch_similarities(degree_dict, val_bor_struc_path, False, self.val_cc_ids, sim_path, 'val', self.val_N_border)
            print('Done computing border structure similarities')

            
        else: # if we're not using the structure channel, we can just set these to None
            self.structure_anchors = None
            self.train_int_struc_similarities = None
            self.val_int_struc_similarities = None
            self.test_int_struc_similarities = None
            self.train_bor_struc_similarities = None
            self.val_bor_struc_similarities = None
            self.test_bor_struc_similarities = None

##################################################
# Prepare data

    def prepare_test_data(self):
        '''
        Same as prepare_data, but for test dataset
        '''
        
        print('--- Started Preparing Test Data ---')
        self.test_cc_ids = self.initialize_cc_ids(self.test_sub_G)

        print('--- Initialize embeddings ---')
        self.init_all_embeddings(split = 'test', trainable = self.hparams['trainable_cc'])

        print('--- Getting Border Sets ---')
        self.get_border_sets(split='test')

        print('--- Getting Similarities ---')
        self.get_similarities(split='test')

        print('--- Initializing Anchor Patches ---')
        # note that we don't need to initialize border position & structure anchor patches because those are shared 
        if self.hparams['use_neighborhood']: 
            self.anchors_neigh_int, self.anchors_neigh_border = init_anchors_neighborhood('test', \
                 self.hparams, self.networkx_graph, self.device, None, None, \
                     self.test_cc_ids, None, None, self.test_N_border)
        else: self.anchors_neigh_int, self.anchors_neigh_border = None, None
        if self.hparams['use_position']: 
            self.anchors_pos_int = init_anchors_pos_int('test', self.hparams, self.networkx_graph, self.device, self.train_sub_G, self.val_sub_G, self.test_sub_G) 
        else: self.anchors_pos_int = None

        print('--- Finished Preparing Test Data ---')
    
    def prepare_data(self):
        '''
        Initialize connected components, precomputed similarity calculations, and anchor patches
        '''
        print('--- Started Preparing Data ---', flush=True)

        # Intialize connected component matrix (n_subgraphs, max_n_cc, max_len_cc)
        self.train_cc_ids = self.initialize_cc_ids(self.train_sub_G)
        self.val_cc_ids = self.initialize_cc_ids(self.val_sub_G)

        # initialize embeddings for each cc
        # 'trainable_cc' flag determines whether the cc embeddings are trainable
        print('--- Initializing CC Embeddings ---', flush=True)
        self.init_all_embeddings(split = 'train_val', trainable = self.hparams['trainable_cc'])

        # Initialize border sets for each cc
        print('--- Initializing CC Border Sets ---', flush=True)
        self.get_border_sets(split='train_val')

        # calculate similarities 
        print('--- Getting Similarities ---', flush=True)
        self.get_similarities(split='train_val')

        # Initialize neighborhood, position, and structure anchor patches
        print('--- Initializing Anchor Patches ---', flush=True)
        if self.hparams['use_neighborhood']: 
            self.anchors_neigh_int, self.anchors_neigh_border = init_anchors_neighborhood('train_val', \
                 self.hparams, self.networkx_graph, self.device, self.train_cc_ids, self.val_cc_ids, \
                     None, self.train_N_border, self.val_N_border, None) # we pass in None for the test_N_border
        else: self.anchors_neigh_int, self.anchors_neigh_border = None, None
        if self.hparams['use_position']:
            self.anchors_pos_int = init_anchors_pos_int('train_val', self.hparams, self.networkx_graph, self.device, self.train_sub_G, self.val_sub_G, None)
            self.anchors_pos_ext = init_anchors_pos_ext(self.hparams, self.networkx_graph, self.device)
        else: self.anchors_pos_int, self.anchors_pos_ext = None, None
        if self.hparams['use_structure']:
            # pass in precomputed sampled structure anchor patches and random walks from which to further subsample
            self.anchors_structure = init_anchors_structure(self.hparams,  self.structure_anchors, self.int_structure_anchor_random_walks, self.bor_structure_anchor_random_walks)
        else: self.anchors_structure = None

        print('--- Finished Preparing Data ---', flush=True)

##################################################
# Data loaders

    def _pad_collate(self, batch):
        '''
        Stacks all examples in the batch to be in shape (batch_sz, ..., ...) 
        & trims padding from border sets & connected component tensors, which were originally 
        padded to the max length across the whole dataset, not the batch
        '''
        subgraph_ids, con_component_ids, N_border, NP_sim, I_S_sim, B_S_sim, idx, labels = zip(*batch)
        # subgraph_ids: (batch_sz, n_nodes_in_subgraph)
        # con_component_ids: (batch_sz, n_con_components, n_nodes_in_cc)
        # con_component_embeds: (batch_sz, n_con_components, hidden_dim)

        # pad subgraph ids in batch to be of shape (batch_sz, max_subgraph_len)
        subgraph_ids_pad = pad_sequence(subgraph_ids, batch_first=True, padding_value=config.PAD_VALUE)



        # stack similarity matrics
        if None in NP_sim: NP_sim = None
        else: NP_sim = torch.stack(NP_sim)
        if None in I_S_sim: I_S_sim = None
        else: I_S_sim = torch.stack(I_S_sim)
        if None in B_S_sim: B_S_sim = None
        else: B_S_sim = torch.stack(B_S_sim)

        # stack and trim the matrix of nodes in each component's border
        if None in N_border: N_border_trimmed = None
        else: 
            N_border = torch.stack(N_border)
            # Trim neighbor border to only be as big as needed for the batch. 
            # This is necessary because the matrix was padded to the max length across all components, not just for the batch
            batch_sz, max_n_cc, _ = N_border.shape
            N_border_reshaped = N_border.view(batch_sz*max_n_cc, -1)
            ind = (torch.sum(torch.abs(N_border_reshaped), dim=0) != 0)
            N_border_trimmed = N_border_reshaped[:,ind].view(batch_sz, max_n_cc, -1)

        
        labels = torch.stack(labels).squeeze() # (batch_sz, 1)
        idx = torch.stack(idx)
        cc_ids = torch.stack(con_component_ids)
        
        # Trim connected component ids to only be as big as needed for the batch 
        batch_sz, max_n_cc, _ = cc_ids.shape
        cc_ids_reshaped = cc_ids.view(batch_sz*max_n_cc, -1)
        ind = (torch.sum(torch.abs(cc_ids_reshaped), dim=0) != 0)
        cc_ids_trimmed = cc_ids_reshaped[:,ind].view(batch_sz, max_n_cc, -1)
        
        return {'subgraph_ids': subgraph_ids_pad, 'cc_ids': cc_ids_trimmed, 'N_border': N_border_trimmed, \
            'NP_sim': NP_sim, 'I_S_sim':I_S_sim, 'B_S_sim':B_S_sim, \
            'subgraph_idx': idx, 'label':labels}

    def train_dataloader(self):
        '''
        Prepare dataloader for training data
        '''

        dataset = SubgraphDataset(self.train_sub_G, self.train_sub_G_label, self.train_cc_ids, \
            self.train_N_border, self.train_neigh_pos_similarities, self.train_int_struc_similarities, \
                self.train_bor_struc_similarities, self.multilabel, self.multilabel_binarizer)

        # drop last examples in batch if batch size is <= number of subgraphs in the training set (this will usually evaluate to true)
        drop_last =  self.hparams['batch_size'] <= len(self.train_sub_G)
        use_weighted_sampler = bool(self.hparams.get('use_weighted_sampler', False))
        if (use_weighted_sampler
            and (not self.multilabel)
            and isinstance(self.train_sub_G_label, torch.Tensor)
            and self.num_classes == 2):
            # Sample inversely proportional to class frequency so minority class appears more often.
            labels = self.train_sub_G_label.view(-1).cpu()
            counts = torch.bincount(labels, minlength=2).float()
            class_w = 1.0 / counts.clamp(min=1.0)
            sample_w = class_w[labels].double().tolist()
            sampler = WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)
            loader = DataLoader(dataset, batch_size=self.hparams['batch_size'], sampler=sampler,
                                shuffle=False, collate_fn=self._pad_collate, drop_last=drop_last)
        else:
            loader = DataLoader(dataset, batch_size = self.hparams['batch_size'], shuffle=True, collate_fn=self._pad_collate, drop_last=drop_last)  #ADDED DROP LAST
        return loader

    def val_dataloader(self):
        '''
        Prepare dataloader for validation data
        '''

        dataset = SubgraphDataset(self.val_sub_G, self.val_sub_G_label, self.val_cc_ids, \
            self.val_N_border, self.val_neigh_pos_similarities, self.val_int_struc_similarities, \
                self.val_bor_struc_similarities, self.multilabel, self.multilabel_binarizer)
        loader = DataLoader(dataset, batch_size = self.hparams['batch_size'], shuffle=False, collate_fn=self._pad_collate)
        return loader
    
    def test_dataloader(self):
        '''
        Prepare dataloader for test data
        '''

        self.prepare_test_data()
        dataset = SubgraphDataset(self.test_sub_G, self.test_sub_G_label, self.test_cc_ids, \
            self.test_N_border, self.test_neigh_pos_similarities, self.test_int_struc_similarities, \
                self.test_bor_struc_similarities, self.multilabel, self.multilabel_binarizer)
        loader = DataLoader(dataset, batch_size = self.hparams['batch_size'], shuffle=False, collate_fn=self._pad_collate)
        return loader

##################################################
# Optimization

    def configure_optimizers(self):
        '''
        Set up Adam optimizer with specified learning rate
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])
        return optimizer

    def backward(self, trainer, loss, optimizer, optimizer_idx): 
        loss.backward(retain_graph=True)
