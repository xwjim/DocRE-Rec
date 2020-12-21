#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

author: wxu
"""

from torch import nn, torch
import numpy as np
from models.BiLSTM import EncoderLSTM

class Decoder(nn.Module):
    def __init__(self,input_size,hidden_size,lstm_drop):
        super(Decoder, self).__init__()

        self.trans_hidden = nn.Linear(input_size,hidden_size)
        self.trans_ceil = nn.Linear(input_size,hidden_size)

        self.rnn = EncoderLSTM(hidden_size, hidden_size, 1, False, lstm_drop)

        self.trans_pred = nn.Linear(hidden_size,hidden_size)

        self.eos_embed = nn.Parameter(torch.zeros(size=(1, hidden_size)))
        self.nop_embed = nn.Parameter(torch.zeros(size=(1, hidden_size)))

    def forward(self, rel_embed,nodes_embed,h_t_pair_label,h_t_pair_path,h_t_pair_path_len,b_ind,h_ind,t_ind,global_step):

        max_path_num = h_t_pair_path.shape[-2]
        max_step_num = h_t_pair_path.shape[-1]
        N_bt = nodes_embed.shape[0]

        # no_rel_mask = (torch.sum(h_t_pair_label[b_ind,h_ind,t_ind,1:],dim=-1)==0)

        select_path_len = h_t_pair_path_len[b_ind,h_ind,t_ind]
        select = (torch.cumsum((select_path_len>0).long(),dim=-1) == 1) & (select_path_len>0)
        path_select_id = torch.nonzero(select.long())[:,1]

        select_path_id = h_t_pair_path[b_ind,h_ind,t_ind][torch.arange(path_select_id.shape[0]).cuda(),path_select_id]


        select_path_len = h_t_pair_path_len[b_ind,h_ind,t_ind][torch.arange(path_select_id.shape[0]).cuda(),path_select_id]

        path_bt_id = b_ind[...,None].repeat(1,max_step_num)
        path_embed = nodes_embed[path_bt_id,select_path_id]
        path_embed = torch.cat((self.eos_embed.repeat(path_embed.shape[0],1,1),path_embed),dim=1)

        init_h = torch.relu(self.trans_hidden(rel_embed)).unsqueeze(dim=0)
        init_c = torch.relu(self.trans_ceil(rel_embed)).unsqueeze(dim=0)

        seq_hidden,_,_ = self.rnn(path_embed,select_path_len,init_h,init_c)

        nodes_ext = torch.cat((self.nop_embed.repeat(N_bt,1,1),nodes_embed),dim=1)
        vocb = torch.relu(self.trans_pred(nodes_ext[b_ind]))

        seq_pred = torch.einsum("abd,acd->abc",seq_hidden,vocb)

        select_path_id = select_path_id + 1
        select_path_id = torch.cat((select_path_id,torch.zeros(select_path_id.shape[0],1,dtype=torch.long).cuda()),dim=-1)
        # select_path_len -= 1

        seqlen, w_ids = torch.broadcast_tensors(select_path_len.unsqueeze(-1),
                                            torch.arange(0, max_step_num+1).cuda()[None,...])
        seq_mask = w_ids<seqlen
        select_path_id[~seq_mask] = 0
    
        return seq_pred,select_path_id,seq_mask
    