#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

author: wxu
"""

import torch
from torch import nn, torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F
import math
from torch.nn import init
from collections import defaultdict


class GraphGateLayer(nn.Module):
    def __init__(self,edges,input_size):
        super(GraphGateLayer, self).__init__()
        # self.W = nn.Parameter(nn.init.normal_(torch.empty(input_size, input_size)), requires_grad=True)
        # self.W_node = nn.ModuleList([nn.Linear(input_size,input_size) for i in range(iters)])
        # self.W_sum = nn.ModuleList([nn.Linear(input_size,input_size) for i in range(iters)])
        self.sigmoid = nn.Sigmoid()
        self.edges = edges
        self.sigmoid = torch.nn.Sigmoid()
        self.W_edge = nn.ModuleList([nn.Linear(input_size,input_size) for i in self.edges])
        self.Wz = nn.Linear(input_size,input_size)#,bias=False)
        nn.init.xavier_uniform_(self.Wz.weight.data)#, gain=1.414)
        self.Wr = nn.Linear(input_size,input_size)#,bias=False)
        nn.init.xavier_uniform_(self.Wr.weight.data)#, gain=1.414)
        self.Uz = nn.Linear(input_size,input_size)#,bias=False)
        nn.init.xavier_uniform_(self.Uz.weight.data)#, gain=1.414)
        self.Ur = nn.Linear(input_size,input_size)#,bias=False)
        nn.init.xavier_uniform_(self.Ur.weight.data)#, gain=1.414)
        self.W = nn.Linear(input_size,input_size)#,bias=False)
        nn.init.xavier_uniform_(self.W.weight.data)#, gain=1.414)
        self.U = nn.Linear(input_size,input_size)#,bias=False)
        nn.init.xavier_uniform_(self.U.weight.data)#, gain=1.414)
        # self.reduce = nn.ModuleDict()
        # for k in self.edges:
        #     self.reduce.update({k: nn.Linear(input_size, input_size, bias=False)})

    def forward(self, nodes_embed,node_adj,node_info,global_step):
        sum_nei = torch.zeros_like(nodes_embed)
        for edge_type in range(len(self.edges)):
            mask = (node_adj==(edge_type+1)).float()
            sum_nei += torch.matmul(mask,self.W_edge[edge_type](nodes_embed))
        mean_num = torch.sum((node_adj>0).float(),dim=-1)
        mean_num[mean_num==0] = 1
        sum_nei = torch.div(sum_nei,mean_num.unsqueeze(-1))
        zvt = self.sigmoid(self.Wz(sum_nei)+self.Uz(nodes_embed))
        rvt = self.sigmoid(self.Wr(sum_nei)+self.Ur(nodes_embed))
        h_hat_t = torch.tanh(self.W(sum_nei)+self.U(nodes_embed*rvt))
        nodes_embed = (1-zvt)*nodes_embed + zvt*h_hat_t
    
        return nodes_embed

class GraphMultiHeadAttention(nn.Module):
    def __init__(self,edges,in_fea,hidden,nhead):
        super(GraphMultiHeadAttention, self).__init__()
        self.head_graph = nn.ModuleList([GraphAttentionLayer(edges,in_fea,hidden) for _ in range(nhead)])
        self.nhead = nhead
        self.layer_norm = nn.LayerNorm(in_fea, eps=1e-6)

    def forward(self, nodes_embed,node_adj,node_info,global_step):

        q = self.layer_norm(nodes_embed)
        x = []
        for cnt in range(0, self.nhead):
            x.append(self.head_graph[cnt](q,node_adj,node_info,global_step))
    
        return torch.cat(x,dim=-1)

class GraphAttentionLayer(nn.Module):
    def __init__(self,edges,input_size,hidden_size):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Parameter(torch.zeros(size=(input_size, hidden_size)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.sigmoid = nn.Sigmoid()
        self.edges = edges
        self.sigmoid = torch.nn.Sigmoid()
        self.W_edge = nn.ModuleList([nn.Linear(2*hidden_size,1) for i in (self.edges)])
        for m in self.W_edge:
            nn.init.xavier_uniform_(m.weight.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, nodes_embed,node_adj,node_info,global_step):

        N_bt = nodes_embed.shape[0]
        N = nodes_embed.shape[1]
        h = torch.matmul(nodes_embed,self.W.unsqueeze(0))
        a_input = torch.cat([h.repeat(1,1,N).view(N_bt,N * N, -1), h.repeat(1,N,1)], dim=-1)
        weight = torch.zeros(N_bt,N*N).cuda()
        for edge_type in range(len(self.edges)):
            mask = (node_adj==(edge_type+1)).float().view(N_bt,-1)
            weight += mask * self.W_edge[edge_type](a_input).squeeze(dim=-1)
        weight = self.leakyrelu(weight).view(N_bt,N,N)
        weight = weight.masked_fill(node_adj==0, -1e9)
        attention = F.softmax(weight, dim=-1)
        out = torch.matmul(attention, h)
        out = torch.relu(out)
        return out

class DualAttention(nn.Module):
    def __init__(self,hidden_size,entity_limit,mention_limit,sent_limit):
        super(DualAttention, self).__init__()
        self.linear_head = nn.Linear(hidden_size, hidden_size,bias=False)
        self.linear_tail = nn.Linear(hidden_size, hidden_size,bias=False)
        self.bi_att_c = nn.Linear(hidden_size,1,bias=False)
        self.bi_att_q = nn.Linear(hidden_size,1,bias=False)
        self.bi_att_cq = nn.Linear(hidden_size,1,bias=False)
        self.entity_limit = entity_limit
        self.mention_limit = mention_limit
        self.sent_limit = sent_limit

        # self.head_out = nn.Linear(2*hidden_size, hidden_size,bias=False)
        # self.tail_out = nn.Linear(2*hidden_size, hidden_size,bias=False)

    def forward(self,entity_embed,mention_embed,sent_embed,entity_info,mention_num,b_ind,h_ind,t_ind):

        batchind = torch.zeros_like(entity_info[...,6])
        for i in range(entity_info.shape[0]):
            batchind[i] = i

        # ment2sent = sent_embed[batchind,entity_info[...,6]]

        # mention_embed = torch.cat((mention_embed,ment2sent),dim=-1)

        hfo = mention_embed[b_ind,h_ind]
        tfo = mention_embed[b_ind,t_ind]

        hf = torch.relu(self.linear_head(hfo))
        tf = torch.relu(self.linear_tail(tfo))


        mention_limit = mention_embed.shape[-2]
        
        cq = []
        for i in range(mention_limit):
            hfi = hf.select(2,i).unsqueeze(2)
            cqi = self.bi_att_cq(hfi*tf).squeeze()
            cq.append(cqi)
        cq = torch.stack(cq,dim=-1)

        s = self.bi_att_c(hf).expand(-1,-1,mention_limit) +\
            self.bi_att_q(tf).permute(0,2,1).expand(-1,mention_limit,-1)+ \
            cq
        
        mask = torch.arange(mention_limit).unsqueeze(0).unsqueeze(-1).expand(
                    s.shape[0],-1,s.shape[1]).cuda()
        head_mention_num = mention_num[b_ind,h_ind].unsqueeze(-1).unsqueeze(-1).expand(
                    -1,s.shape[-2],s.shape[-1])
        s = s.masked_fill(mask>=head_mention_num,-1e9)

        mask = torch.arange(mention_limit).unsqueeze(0).unsqueeze(0).expand(
                    s.shape[0],s.shape[1],-1).cuda()
        tail_mention_num = mention_num[b_ind,t_ind].unsqueeze(-1).unsqueeze(-1).expand(
                    -1,s.shape[-2],s.shape[-1])
        s = s.masked_fill(mask>=tail_mention_num,-1e9)


        h_weight = torch.softmax(torch.max(s,dim=-1,keepdim=True)[0],dim=-2).permute(0,2,1)
        start_re_output = torch.matmul(h_weight,hf).squeeze(dim=-2)
        
        t_weight = torch.softmax(torch.max(s,dim=-2,keepdim=True)[0],dim=-1)
        end_re_output = torch.matmul(t_weight,tf).squeeze(dim=-2)

        head_embed = torch.cat((entity_embed[b_ind,h_ind],start_re_output),dim=-1)
        tail_embed = torch.cat((entity_embed[b_ind,t_ind],end_re_output),dim=-1)

        # head_embed = torch.tanh(self.head_out(head_embed))
        # tail_embed = torch.tanh(self.tail_out(tail_embed))

        return head_embed,tail_embed


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(~mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
class MultiHeadSelfAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self,in_fea, nhead=4,dropout=0.1):
        super().__init__()
        self.n_head = nhead

        self.w_qs = nn.Linear(in_fea, in_fea, bias=False)
        self.w_ks = nn.Linear(in_fea, in_fea, bias=False)
        self.w_vs = nn.Linear(in_fea, in_fea, bias=False)

        self.layer_norm = nn.LayerNorm(in_fea, eps=1e-6)
        self.attention = ScaledDotProductAttention(temperature=in_fea ** 0.5)

        self.fc = nn.Linear(in_fea, in_fea, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self,q,mask=None):

        sz_b = q.shape[0]
        n_head = self.n_head
        len_q = q.shape[1]
        sub_fea = int(q.shape[-1]/n_head)

        residual = q
        q = self.layer_norm(q)

        k = q
        v = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, sub_fea)
        k = self.w_ks(k).view(sz_b, len_q, n_head, sub_fea)
        v = self.w_vs(v).view(sz_b, len_q, n_head, sub_fea)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        return q