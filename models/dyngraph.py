#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

author: wxu
"""

import torch
from torch import nn, torch
import numpy as np
import math
from collections import defaultdict
from models.BiLSTM import EncoderLSTM
from models.decoder import Decoder
from models.level_transform import doc2sent,sent2doc,doc2graph,split_graph_type,graph2doc,doc_entity_mask
from models.graph import GraphMultiHeadAttention,GraphGateLayer,MultiHeadSelfAttention
from config import dataloader


IGNORE_INDEX = -100
ENTITY_IND = dataloader.ENTITY_IND
MENTION_IND = dataloader.MENTION_IND
SENTENCE_IND = dataloader.SENTENCE_IND


class DynGraphLayer(nn.Module):
    def __init__(self, config):
        super(DynGraphLayer, self).__init__()

        self.embed_layer = EmbedLayer(config)

        self.graph_layer = GraphLayer(config)

        if config.freeze_model:
            for param in self.graph_layer.parameters():              
                    param.requires_grad = False

    def forward(self,data,global_step=None):

        context_output = self.embed_layer(data,global_step)

        return self.graph_layer(data,context_output,global_step)
        
        

class EmbedLayer(nn.Module):
    def __init__(self, config,entity_mask=False):
        super(EmbedLayer, self).__init__()

        self.use_entity_type = config.use_entity_type
        self.use_pos_embed = config.use_pos_embed
        self.entity_mask = entity_mask

        hidden_size = config.word_embed_hidden
        word_vec_size = config.data_word_vec.shape[0]
        self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])
        if self.entity_mask:
            self.type_mask_emb = nn.Embedding(7, config.data_word_vec.shape[1])
        self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))

        self.word_emb.weight.requires_grad = False
        input_size = config.data_word_vec.shape[1]

        if self.use_entity_type:
            input_size += config.entity_type_size
            self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)
        if self.use_pos_embed:
            input_size += config.pos_embed_size
            self.entity_embed = nn.Embedding(config.max_length, config.pos_embed_size, padding_idx=0)
        self.rnn = EncoderLSTM(input_size, hidden_size, 1, True, config.lstm_drop)
        self.linear_context = nn.Linear(2*hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.linear_context.weight.data)#, gain=1.414)
    @staticmethod
    def sort(lengths):
        sorted_idx = np.argsort(-lengths)  # indices that result in sorted sequence
        original_idx = np.argsort(sorted_idx)

        return sorted_idx, original_idx
    def forward(self,data,global_step=None):

        context_idxs = data['context_idxs']
        context_pos = data['context_pos']
        sent_info = data["sent_info"]
        entity_info = data["entity_info"]
        input_lengths =  data['input_lengths']
        context_masks = data['context_masks']
        context_ner = data['context_ner']

        docs = self.word_emb(context_idxs)
        if self.entity_mask:
            docs = doc_entity_mask(docs,entity_info,self.type_mask_emb)
        if self.use_pos_embed:
            docs = torch.cat([docs, self.entity_embed(context_pos)], dim=-1)
        if self.use_entity_type:
            docs = torch.cat([docs, self.ner_emb(context_ner)], dim=-1)
        context_output,_,_ = self.rnn(docs, input_lengths)
        context_output = torch.relu(self.linear_context(context_output))

        return context_output
class GraphLayer(nn.Module):
    def __init__(self, config):
        super(GraphLayer, self).__init__()

        input_size = config.data_word_vec.shape[1]
        self.use_graph = config.use_graph
        self.use_distance = config.use_distance
        self.relation_num = config.relation_num
        self.entity_limit = config.entity_limit
        self.sent_limit = config.sent_limit
        self.node_limit = config.node_limit
        self.mention_limit = config.mention_limit
        self.dis2idx = config.dis2idx

        hidden_size = config.word_embed_hidden

        if self.use_graph:
            self.graph_reason = GraphReasonLayer(config.edges,hidden_size,config.graph_out_hidden,
                                            config.graph_iter,config.graph_type,config.graph_drop)
            hidden_size = config.graph_out_hidden

        self.linear_head = nn.Linear(hidden_size, hidden_size)
        self.linear_tail = nn.Linear(hidden_size, hidden_size)

        re_hidden_size = hidden_size
        if self.use_distance:
            self.dis_embed = nn.Embedding(20, config.dis_size)
            re_hidden_size = re_hidden_size + config.abs_dis_size

        re_hidden_size = 2*re_hidden_size
        self.rel_linear1 = nn.Linear(re_hidden_size, re_hidden_size)
        self.rel_linear2 = nn.Linear(re_hidden_size, config.relation_num)

        self.decoder = Decoder(re_hidden_size,config.graph_out_hidden,config.lstm_drop)


        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,data,context_output,global_step=None):

        context_idxs = data['context_idxs']
        context_pos = data['context_pos']
        input_lengths =  data['input_lengths']
        sent_info = data["sent_info"]
        entity_info = data["entity_info"]
        node_info = data["node_info"]
        node_adj = data["node_adj"]
        node_size = data["node_size"]
        max_size = torch.max(node_size,dim=0)[0].item()
        node_adj = node_adj[:,:max_size,:max_size]
        node_info = node_info[:,:max_size,:]
        context_masks = data['context_masks']
        sample_mask = data["sample_mask"]
        h_t_pair_path = data["h_t_pair_path"]
        h_t_pair_path_len = data["h_t_pair_path_len"]
        h_t_pair_path_edge = data["h_t_pair_path_edge"]
        h_t_pair_label = data["h_t_pair_label"]

        nodes_embed = doc2graph(context_output,entity_info,sent_info)

        if self.use_graph:
            nodes_embed = self.graph_reason(nodes_embed,node_adj,node_info,context_output,sent_info,entity_info,input_lengths,global_step)

        pred_ind = data["sample_mask"]
        b_ind,h_ind,t_ind = torch.where(pred_ind)

        head_embed = torch.relu(self.linear_head(nodes_embed[b_ind,h_ind]))
        tail_embed = torch.relu(self.linear_tail(nodes_embed[b_ind,t_ind]))

        if self.use_distance:
            delta_dis = node_info[b_ind,h_ind,2] - node_info[b_ind,t_ind,2]
            dis = torch.from_numpy(self.dis2idx[torch.abs(delta_dis).cpu()]).cuda()
            dis = torch.where(delta_dis>0,dis,-dis)
            head_dis_embed = self.dis_embed(dis+10)
            tail_dis_embed = self.dis_embed(-dis+10)
            head_embed = torch.cat((head_embed,head_dis_embed),dim=-1)
            tail_embed = torch.cat((tail_embed,tail_dis_embed),dim=-1)

        predict_re = torch.zeros(pred_ind.shape[0],self.entity_limit,self.entity_limit,self.relation_num).cuda()
        predict_re.fill_(IGNORE_INDEX)

        rel_embed = torch.cat((head_embed,tail_embed),dim=-1)
        predict_re[b_ind,h_ind,t_ind] = self.rel_linear2(torch.relu(self.rel_linear1(rel_embed)))

        out_pre_seq = torch.zeros(pred_ind.shape[0],self.entity_limit,self.entity_limit,11,self.node_limit+1).cuda()
        out_tru_seq = torch.zeros(pred_ind.shape[0],self.entity_limit,self.entity_limit,11,dtype=torch.long).cuda()
        out_mask_seq = torch.zeros(pred_ind.shape[0],self.entity_limit,self.entity_limit,11,dtype=torch.bool).cuda()
        seq_pred,seq_truth,seq_mask = self.decoder(rel_embed,nodes_embed,h_t_pair_label,h_t_pair_path,h_t_pair_path_len,\
                                        b_ind,h_ind,t_ind,global_step)
        out_pre_seq[b_ind,h_ind,t_ind,...,:max_size+1] = seq_pred
        out_tru_seq[b_ind,h_ind,t_ind] = seq_truth
        out_mask_seq[b_ind,h_ind,t_ind] = seq_mask
        
        return {"predict_re":predict_re,"seq_pred":out_pre_seq,"seq_truth":out_tru_seq,"seq_mask":out_mask_seq}

class GraphReasonLayer(nn.Module):
    def __init__(self,edges,input_size,out_size,iters,graph_type,graph_drop,graph_head=4):
        super(GraphReasonLayer, self).__init__()
        self.iters = iters
        self.edges = edges
        self.sigmoid = torch.nn.Sigmoid()
        self.graph_type = graph_type
        if graph_type == "atten":
            assert input_size%graph_head == 0
            hidden = int(input_size/graph_head)
            self.block = nn.ModuleList([GraphMultiHeadAttention(edges,(i+1)*input_size,hidden,graph_head) for i in range(iters)])
            hidden = (iters+1)*input_size
            o_size = out_size
        elif graph_type == "gate":
            self.block = nn.ModuleList([GraphGateLayer(edges,input_size) for i in range(iters)])
            hidden = input_size
            o_size = input_size
        else:
            raise("graph choose error")
        self.drop = torch.nn.Dropout(p=graph_drop, inplace=False)
        self.out = nn.Sequential(
            nn.Linear(hidden, o_size),
            nn.ReLU()
        )
        
    def forward(self, nodes_embed,node_adj,node_info,context_output,sent_info,entity_info,input_lengths,global_step):

        for cnt in range(0, self.iters):
            hi = self.block[cnt](nodes_embed,node_adj,node_info,global_step)
            nodes_embed = torch.cat((nodes_embed,hi),dim=-1)
            nodes_embed = self.drop(nodes_embed)

        nodes_embed = self.out(nodes_embed)
    
        return nodes_embed