# General
import numpy as np
import sys
from multiprocessing import Pool
import time

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pytorch Geometric
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import MessagePassing

# Our methods
sys.path.insert(0, '..') # add config to path
import config


class SG_MPN(MessagePassing):
    '''
    A single subgraph-level message passing layer

    Messages are passed from anchor patch to connected component and weighted by the channel-specific similarity between the two.
    The resulting messages for a single component are aggregated and used to update the embedding for the component.
    '''

    def __init__(self, hparams):
        super(SG_MPN, self).__init__(aggr='add')  # "Add" aggregation.
        self.hparams = hparams
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.linear =  nn.Linear(hparams['node_embed_size'] * 2, hparams['node_embed_size']).to(self.device)
        self.linear_position = nn.Linear(hparams['node_embed_size'],1).to(self.device) 

    def create_patch_embedding_matrix(self,cc_embeds, cc_embed_mask, anchor_embeds, anchor_mask):
        '''
        Concatenate the connected component and anchor patch embeddings into a single matrix.
        This will be used an input for the pytorch geometric message passing framework.
        '''
        batch_sz, max_n_cc, cc_hidden_dim = cc_embeds.shape
        anchor_hidden_dim = anchor_embeds.shape[-1]

        # reshape connected component & anchor patch embedding matrices
        reshaped_cc_embeds = cc_embeds.view(-1, cc_hidden_dim) #(batch_sz * max_n_cc , hidden_dim)
        reshaped_anchor_embeds =  anchor_embeds.view(-1, anchor_hidden_dim) #(batch_sz * max_n_cc * n_sampled_patches, hidden_dim)

        # concatenate the anchor patch and connected component embeddings into single matrix
        patch_embedding_matrix = torch.cat([reshaped_anchor_embeds, reshaped_cc_embeds])
        return patch_embedding_matrix

    def create_edge_index(self, reshaped_cc_ids, reshaped_anchor_patch_ids, anchor_mask, n_anchor_patches):
        '''
        Create edge matrix of shape (2, # edges) where edges exist between connected components and their associated anchor patches

        Note that edges don't exist between components or between anchor patches
        '''
        # get indices into patch matrix corresponding to anchor patches
        anchor_inds = torch.tensor(range(reshaped_anchor_patch_ids.shape[0]))
        
        # get indices into patch matrix corresponding to connected components
        cc_inds = torch.tensor(range(reshaped_cc_ids.shape[0])) + reshaped_anchor_patch_ids.shape[0] 
        
        # repeat CC indices n_anchor_patches times
        cc_inds_matched = cc_inds.repeat_interleave(n_anchor_patches)
        
        # stack together two indices to create (2,E) edge matrix
        edge_index = torch.stack((anchor_inds, cc_inds_matched)).to(device=self.device)
        mask_inds = anchor_mask.view(-1, anchor_mask.shape[-1])[:,0]

        return edge_index[:,mask_inds], mask_inds

    def get_similarities(self, networkx_graph, edge_index, sims, cc_ids, anchor_ids, anchors_sim_index):
        '''
        Reshape similarities tensor of shape (n edges, 1) that contains similarity value for each edge in the edge index

        sims: (batch_size, max_n_cc, n possible anchor patches)
        edge_index: (2, number of edges between components and anchor patches)
        anchors_sim_index: indices into sims matrix for the structure channel that specify which anchor patches we're using
        '''
        n_cc = cc_ids.shape[0] 
        n_anchor_patches = anchor_ids.shape[0]
        
        batch_sz, max_n_cc, n_patch_options = sims.shape
        sims = sims.view(batch_sz * max_n_cc, n_patch_options)


        if anchors_sim_index != None: anchors_sim_index = anchors_sim_index * torch.unique(edge_index[1,:]).shape[0] # n unique CC
        
        # NOTE: edge_index contains stacked anchor, cc embeddings
        if anchors_sim_index == None: # neighborhood, position channels
            anchor_indices = anchor_ids[edge_index[0,:],:] - 1 # get the indices into the similarity matrix of which anchors were sampled
            cc_indices = edge_index[1,:] - n_anchor_patches  # get indices of the conneced components into the similarity matrix
            similarities = sims[cc_indices, anchor_indices.squeeze()]
        else: #structure channel

            # get indices of the conneced components into the similarity matrix
            cc_indices = edge_index[1,:] - n_anchor_patches #indexing into edge index is different than indexing into sims because patch matrix from which edge index was derived stacks anchor paches before the cc embeddings
            similarities = sims[cc_indices, torch.tensor(anchors_sim_index)] # anchors_sim_index provides indexing into the big similarity matrix - it tells you which anchors we actually sampled

        if len(similarities.shape) == 1: similarities = similarities.unsqueeze(-1)

        return similarities

    def generate_pos_struc_embeddings(self, raw_msgs, cc_ids, anchor_ids, edge_index, edge_index_mask):
        '''
        Generates the property aware position/structural embeddings for each connected component
        '''
        # Generate position/structure embeddings
        n_cc = cc_ids.shape[0]
        n_anchor_patches = anchor_ids.shape[0]
        embed_sz = raw_msgs.shape[1]
        n_anchors_per_cc = int(n_anchor_patches/n_cc)

        # 1) add masked CC back in & reshape
        # raw_msgs doesn't include padding so we need to add padding back in
        # NOTE: while these are named as position embeddings, these apply to structure channel as well
        pos_embeds = torch.zeros((n_cc * n_anchors_per_cc, embed_sz)).to(device=self.device) + config.PAD_VALUE
        pos_embeds[edge_index_mask] = raw_msgs # raw_msgs doesn't include padding so we need to add padding back in
        pos_embeds_reshaped = pos_embeds.view(-1, n_anchors_per_cc, embed_sz)

        # 2) linear layer + normalization
        position_out = self.linear_position(pos_embeds_reshaped).squeeze(-1)

        # optionally normalize the output of the linear layer (this is what P-GNN paper did) 
        if 'norm_pos_struc_embed' in self.hparams and self.hparams['norm_pos_struc_embed']:
            position_out = F.normalize(position_out, p=2, dim=-1) 
        else: # otherwise, just push through a relu
            position_out = F.relu(position_out) 

        return position_out #(n subgraphs * n_cc, n_anchors_per_cc )

    def forward(self, networkx_graph, sims, cc_ids, cc_embeds, cc_embed_mask, \
            anchor_patches, anchor_embeds, anchor_mask, anchors_sim_index): 
        '''
        Performs a single message passing layer

        Returns:
            - cc_embed_matrix_reshaped: order-invariant hidden representation (batch_sz, max_n_cc, node embed dim)
            - position_struc_out_reshaped: property aware cc representation (batch_sz, max_n_cc, n_anchor_patches)
        '''
        
        # reshape anchor patches & CC embeddings & stack together
        # NOTE: anchor patches then CC stacked in matrix
        patch_matrix = self.create_patch_embedding_matrix(cc_embeds, cc_embed_mask, anchor_embeds, anchor_mask)

        # reshape cc & anchor patch id matrices
        batch_sz, max_n_cc, max_size_cc = cc_ids.shape
        cc_ids = cc_ids.view(-1, max_size_cc) # (batch_sz * max_n_cc, max_size_cc)

        anchor_ids = anchor_patches.contiguous().view(-1, anchor_patches.shape[-1]) # (batch_sz * max_n_cc * n_sampled_patches, anchor patch size)
        n_anchor_patches_sampled = anchor_ids.shape[0]

        # create edge index
        edge_index, edge_index_mask = self.create_edge_index(cc_ids, anchor_ids, anchor_mask, anchor_patches.shape[2])

        # get similarity values for each edge index
        similarities = self.get_similarities( networkx_graph, edge_index, sims, cc_ids, anchor_ids, anchors_sim_index)

        # Perform Message Passing
        # propagated_msgs: (length of concatenated anchor patches & cc, node dim size)
        propagated_msgs, raw_msgs =  self.propagate(edge_index, x=patch_matrix, similarity=similarities) 
        
        # Generate Position/Structure Embeddings
        position_struc_out = self.generate_pos_struc_embeddings(raw_msgs, cc_ids, anchor_ids, edge_index, edge_index_mask)

        # index resulting propagated messagaes to get updated CC embeddings & reshape
        cc_embed_matrix = propagated_msgs[n_anchor_patches_sampled:,:]
        cc_embed_matrix_reshaped = cc_embed_matrix.view(batch_sz , max_n_cc ,-1)

        # reshape property aware position/structure embeddings
        position_struc_out_reshaped = position_struc_out.view(batch_sz, max_n_cc, -1)
        
        return cc_embed_matrix_reshaped, position_struc_out_reshaped

    def propagate(self, edge_index, size=None, **kwargs):
        # Custom lightweight propagate that does not rely on torch_geometric internals.
        x = kwargs.get('x')
        similarity = kwargs.get('similarity')

        if x is None or similarity is None:
            raise ValueError("propagate requires 'x' and 'similarity' tensors")

        src = edge_index[0]
        dst = edge_index[1]
        if similarity.dim() == 1:
            similarity = similarity.unsqueeze(-1)

        msg_out = similarity * x[src]

        out = torch.zeros((x.size(0), x.size(1)), device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg_out)

        out = self.update(out, x)

        return out, msg_out


    def message(self, x_j, similarity): #default is source to target
        '''
        The message is the anchor patch representation weighted by the similarity between the patch and the component
        '''
        return similarity * x_j 

    def update(self, aggr_out, x):
        '''
        Update the connected component embedding from the result of the aggregation. The default is to 'use_mpn_projection',
        i.e. concatenate the aggregated messages with the previous cc embedding and push through a relu
        '''
        if self.hparams['use_mpn_projection']:
            return F.relu(self.linear(torch.cat([x, aggr_out], dim=1)))
        else:
            return aggr_out
