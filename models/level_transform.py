import torch
from torch import nn, torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch
import pdb
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
import numpy as np
import math
from torch.nn import init
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from config import dataloader


IGNORE_INDEX = -100
ENTITY_IND = dataloader.ENTITY_IND
MENTION_IND = dataloader.MENTION_IND
SENTENCE_IND = dataloader.SENTENCE_IND


def mergy_token(context_output,info):
    
    word_size =  context_output.shape[1]
    start, end, w_ids = torch.broadcast_tensors(info[...,2].unsqueeze(-1),
                                                info[...,3].unsqueeze(-1),
                                                torch.arange(0, word_size).cuda()[None,None])
    e_mapping = (torch.ge(w_ids, start) & torch.lt(w_ids, end)).float()
    if len(e_mapping.shape) == 4:
        embed = torch.einsum("abcd,ade->abce",e_mapping,context_output)
    elif len(e_mapping.shape) == 3:
        embed = torch.einsum("abd,ade->abe",e_mapping,context_output)
    else:
        raise("dim mismatch")

    spancnt = torch.sum(e_mapping,dim=-1).unsqueeze(-1)
    spancnt[spancnt==0] = 1

    embed = torch.div(embed,spancnt)
    return embed

def mergy_token_max(context_output,info):
    
    word_size =  context_output.shape[1]
    start, end, w_ids = torch.broadcast_tensors(info[...,2].unsqueeze(-1),
                                                info[...,3].unsqueeze(-1),
                                                torch.arange(0, word_size).cuda()[None,None])
    e_mapping = (torch.ge(w_ids, start) & torch.lt(w_ids, end)).float()
    batchind = torch.zeros_like(w_ids)
    for i in range(batchind.shape[0]):
        batchind[i] = i
    if len(e_mapping.shape) == 4:
        embed = torch.zeros(info.shape[0],info.shape[1],info.shape[2],context_output.shape[-1]).cuda()
    elif len(e_mapping.shape) == 3:
        embed = torch.zeros(info.shape[0],info.shape[1],context_output.shape[-1]).cuda()
    else:
        raise("dim mismatch")
    mask = info[...,0] > 0
    mention_span = context_output[batchind[mask],w_ids[mask]]
    mention_span = mention_span.masked_fill(e_mapping[mask].unsqueeze(-1)<1,-1e9)
    embed[mask] = torch.max(mention_span,dim=-2)[0]

    return embed

def split_graph_type(nodes_embed,node_info,entity_limit,mention_limit,sent_limit):

    batchind = torch.zeros_like(node_info[...,0])
    for i in range(node_info.shape[0]):
        batchind[i] = i

    select = (node_info[...,-1] == SENTENCE_IND)
    sent_embed = torch.zeros(node_info.shape[0],sent_limit,nodes_embed.shape[-1]).cuda()
    sent_embed[batchind[select],node_info[...,6][select]] = nodes_embed[select]


    select = (node_info[...,-1] == MENTION_IND)
    mention_embed = torch.zeros(node_info.shape[0],entity_limit,mention_limit,
                                            nodes_embed.shape[-1]).cuda()
    mention_embed[batchind[select],node_info[...,4][select],node_info[...,5][select]] = nodes_embed[select]

    select = (node_info[...,-1] == ENTITY_IND)
    entity_embed = torch.zeros(node_info.shape[0],entity_limit,nodes_embed.shape[-1]).cuda()
    entity_embed[batchind[select],node_info[...,4][select]] = nodes_embed[select]


    return entity_embed,mention_embed,sent_embed
def doc2graph(context_output,entity_info,sent_info):

    mention_embed = mergy_token(context_output,entity_info)
    entity_embed = torch.sum(mention_embed,dim=-2)
    mention_num = torch.sum(entity_info[...,0],dim=-1)
    mention_num[mention_num==0] = -1
    entity_embed = torch.div(entity_embed,mention_num.float().unsqueeze(-1))
    sent_embed = mergy_token(context_output,sent_info)

    nodes = []
    batchnum = context_output.shape[0]
    for i in range(batchnum):
        ins_mention_embed = mention_embed[i]
        ins_entity_embed = entity_embed[i]
        ins_sent_embed = sent_embed[i]
        ins_entity_info = entity_info[i]
        ins_sent_info = sent_info[i]
        ins_node = ins_entity_embed[mention_num[i]>0]
        node_type = 0*torch.ones(ins_node.shape[0]).cuda()
        select = ins_entity_info[:,:,0]>0
        ins_node = torch.cat((ins_node,ins_mention_embed[select]),dim=0)
        node_type = torch.cat((node_type,1*torch.ones(ins_node.shape[0]-node_type.shape[0]).cuda()),dim=0)
        select = ins_sent_info[:,0]>0
        ins_node = torch.cat((ins_node,ins_sent_embed[select]),dim=0)
        node_type = torch.cat((node_type,2*torch.ones(ins_node.shape[0]-node_type.shape[0]).cuda()),dim=0)
        nodes.append(ins_node)

    nodes_embed = pad_sequence(nodes, batch_first=True, padding_value=0)

    return nodes_embed

def doc2sent(docs,sent_info):
    sents = []
    doc_sents_num = np.zeros(sent_info.shape[0],dtype=np.int)
    sents_len = []
    for idx in range(sent_info.shape[0]):
        sents_num = torch.sum(sent_info[idx,:,0])
        doc_word_len = sent_info[idx,sents_num-1,3]
        doc_sents_num[idx] = sents_num
        sents.extend(torch.split(docs[idx][:doc_word_len],(sent_info[idx,:,3]-sent_info[idx,:,2]).tolist(),dim=0)[:sents_num])
        sents_len.extend((sent_info[idx,:,3]-sent_info[idx,:,2]).tolist()[:sents_num])
    sents = pad_sequence(sents, batch_first=True, padding_value=-1)
    sents_len = np.array(sents_len)

    return sents,sents_len,doc_sents_num

def sent2doc(sents,sents_len,doc_sents_num):
    sents = torch.split(sents,1,dim=0)
    sents = list(map(lambda x: torch.squeeze(x[0][0,:x[1]],dim=0),zip(sents,sents_len)))
    doc_sents_num = np.cumsum(doc_sents_num)
    doc = []
    for idx in range(doc_sents_num.shape[0]):
        left = doc_sents_num[idx-1] if idx != 0 else 0 
        right = doc_sents_num[idx]
        doc.append(torch.cat(sents[left:right],dim=-2))
    docs = pad_sequence(doc, batch_first=True, padding_value=-1)

    return docs

def graph2doc(nodes_embed,node_info,context_output):

    w_ids, start, end, n_type= torch.broadcast_tensors(torch.arange(0, context_output.shape[-2]).cuda()[None,...,None],
                                                            node_info[...,2].unsqueeze(1),
                                                            node_info[...,3].unsqueeze(1), 
                                                            node_info[...,-1].unsqueeze(1))
    M_mention = (torch.ge(w_ids, start) & torch.lt(w_ids, end) & (n_type == MENTION_IND)).float()

    bid,wid,nid = torch.where(M_mention>0)
    M = torch.zeros_like(M_mention)
    M[bid,wid,node_info[bid,nid,4]] = 1

    entity = torch.matmul(M,nodes_embed)
    mask = torch.sum(M,dim=-1).unsqueeze(dim=-1)
    docs = torch.where(mask>0,entity,context_output)

    return docs

def doc_entity_mask(docs,entity_info,type_embed):

    context_output = docs.clone()
    word_size =  context_output.shape[1]
    start, end, w_t, w_ids = torch.broadcast_tensors(entity_info[...,2].unsqueeze(-1),
                                                entity_info[...,3].unsqueeze(-1),
                                                entity_info[...,1].unsqueeze(-1),
                                                torch.arange(0, word_size).cuda()[None,None])
    e_mapping = (torch.ge(w_ids, start) & torch.lt(w_ids, end))
    w_type = w_t.clone()
    w_type[~e_mapping] = 0
    entity_mapping = torch.sum(torch.sum(e_mapping.float(),dim=1),dim=1)>0
    type_ind = torch.sum(torch.sum(w_type,dim=1),dim=1)
    type_ind[type_ind>6] = 0
    context_output[entity_mapping] = type_embed(type_ind[entity_mapping])

    return context_output

class span_embed_layer(nn.Module):
    def __init__(self,config,hidden_size):
        super(span_embed_layer, self).__init__()
        self.span_len_embed = nn.Embedding(20, config.dis_size)
        self.scale = nn.Linear(hidden_size,1)
        self.span_out = nn.Linear(3*hidden_size + config.dis_size,hidden_size)

    def forward(self,context_output,info):
        
        word_size =  context_output.shape[1]
        scores = self.scale(context_output).squeeze(dim=-1)
        start, end, w_ids = torch.broadcast_tensors(info[...,2].unsqueeze(-1),
                                                    info[...,3].unsqueeze(-1),
                                                    torch.arange(0, word_size).cuda()[None,None])
        e_mapping = (torch.ge(w_ids, start) & torch.lt(w_ids, end)).float()
        start_mapping = torch.eq(w_ids, start).float()
        end_mapping = torch.eq(w_ids, end).float()
        if len(e_mapping.shape) == 4:
            k = torch.einsum("abcd,ad->abcd",e_mapping,scores)
            k = k.masked_fill(~e_mapping.bool(),-1e9)
            k = F.softmax(k,dim=-1)
            embed = torch.einsum("abcd,ade->abce",k,context_output)
            start_embed = torch.einsum("abcd,ade->abce",start_mapping,context_output)
            end_embed = torch.einsum("abcd,ade->abce",end_mapping,context_output)
        elif len(e_mapping.shape) == 3:
            k = torch.einsum("abd,ad->abd",e_mapping,scores)
            k = k.masked_fill(~e_mapping.bool(),-1e9)
            k = F.softmax(k,dim=-1)
            embed = torch.einsum("abd,ade->abe",k,context_output)
            start_embed = torch.einsum("abd,ade->abe",start_mapping,context_output)
            end_embed = torch.einsum("abd,ade->abe",end_mapping,context_output)
        else:
            raise("dim mismatch")

        len_embed = self.span_len_embed(torch.from_numpy(self.dis2idx[(info[...,3] - info[...,2]).cpu()]).cuda())
        out = torch.cat([len_embed,embed,start_embed,end_embed],dim=-1)
        out = torch.relu(self.span_out(out))

        # embed = torch.cat()

        return out
