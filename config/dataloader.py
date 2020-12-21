# coding: utf-8
from torch.utils.data import Dataset
import time
import random
import numpy as np
import os
from multiprocessing import Pool
from collections import defaultdict
import json
import torch
import copy

IGNORE_INDEX = -100
ENTITY_IND = 1
MENTION_IND = 2
SENTENCE_IND = 3

def cirle_reader(batch):
    out = []
    for (index,ins,config) in batch:
        out.append(reader(index,ins,config))
    return out
def is_label_select(label,vertex2sent,load_data_type):
    if load_data_type == "inter":
        if len(set(vertex2sent[label["h"]]) & set(vertex2sent[label["t"]]))>0:
            return False
        else:
            return True
    if load_data_type == "intra":
        if len(set(vertex2sent[label["h"]]) & set(vertex2sent[label["t"]]))>0:
            return True
        else:
            return False
    return True
def reader(index,ins,config):
    max_length,load_data_type,relation_num,word_size,dis2idx,sent_limit,\
        entity_limit,mention_limit,char_limit,edges,entity_info_len,metapath,type_path_num = config 
    h_t_pair_label = np.zeros((entity_limit,entity_limit,relation_num),dtype=np.int)
    h_t_pair_evi = np.zeros((entity_limit,entity_limit,sent_limit),dtype=np.int)
    # mask type startid endid sentenceid entityid nodetype
    entity_info = np.zeros((entity_limit,mention_limit,entity_info_len),dtype=np.int)
    # mask type startid endid sentenceid entityid nodetype
    sent_info = np.zeros((sent_limit,entity_info_len),dtype=np.int)                  

    vertex2sent = defaultdict(list)
    L = len(ins['vertexSet'])

    for idx in range(min(L,entity_limit)):
        hlist = ins['vertexSet'][idx]
        for cnt in range(min(mention_limit,len(hlist))):
            h = hlist[cnt]
            vertex2sent[idx].append(h["sent_id"])
            entity_info[idx,cnt] = [1,ins["vertexSet"][idx][0]["type"],h['pos'][0],h['pos'][1],idx,cnt,h["sent_id"],MENTION_IND]

    labels = ins['labels']

    for label in labels:
        if not is_label_select(label,vertex2sent,load_data_type):
            continue
        h_t_pair_label[label['h'],label['t'],label['r']] = 1
        for idx in label["evidence"]:
            if idx < sent_limit:
                h_t_pair_evi[label['h'],label['t'],idx] = 1
    for (h_idx,t_idx) in ins["na_triple"]:
        h_t_pair_label[h_idx,t_idx,0] = 1

    for cnt in range(min(len(ins["Ls"])-1,sent_limit)):
        sent_info[cnt] = [1,-100,ins["Ls"][cnt],ins["Ls"][cnt+1],0,0,cnt,SENTENCE_IND]

    # label entitytype pos1 pos2 entity mentioncnt sentid nodetype

    return  {
        "h_t_pair_label":h_t_pair_label,
        "index":index,
        "sent_info":sent_info,
        "entity_info":entity_info}

class SelfData(Dataset):
    def __init__(self,config,datatype="train"):
        self.datatype = datatype
        self.sample_data = config.sample_data
        self.char_limit = config.char_limit
        self.sent_limit = config.sent_limit
        self.entity_limit = config.entity_limit
        self.mention_limit = config.mention_limit
        self.node_limit = config.node_limit
        self.entity_info_len = config.entity_info_len
        self.train_h_t_limit = config.train_h_t_limit
        self.test_h_t_limit = config.test_h_t_limit
        self.max_length = config.max_length
        self.relation_num = config.relation_num
        self.word_size = config.word_size
        self.dis2idx = config.dis2idx
        self.edges = config.edges
        self.load_data_type = config.load_data_type

        self.data_contain = {}
        start_time = time.time()
        if datatype == "train":
            print("Reading training data...")
            prefix = config.train_prefix
            print ("train", prefix)
        elif datatype == "test":
            print("Reading testing data...")
            prefix = config.test_prefix
            print (prefix)
        else:
            raise("Error")
        
        self.data_word = np.load(os.path.join(config.data_path, prefix+'_word.npy'))
        self.data_pos = np.load(os.path.join(config.data_path, prefix+'_pos.npy'))
        self.data_ner = np.load(os.path.join(config.data_path, prefix+'_ner.npy'))
        self.data_char = np.load(os.path.join(config.data_path, prefix+'_char.npy'))
        self.data_file = json.load(open(os.path.join(config.data_path, prefix+'.json')))

        self.data_node_info = np.load(os.path.join(config.data_path, prefix+'_node_info.npy'))
        self.data_node_adj = np.load(os.path.join(config.data_path, prefix+'_node_adj.npy'))
        self.data_h_t_pair_path = np.load(os.path.join(config.data_path, prefix+"_"+'_h_t_pair_path.npy'))
        self.data_h_t_pair_path_len = np.load(os.path.join(config.data_path, prefix+"_"+'_h_t_pair_path_len.npy'))
        self.data_h_t_pair_path_edge = np.load(os.path.join(config.data_path, prefix+"_"+'_h_t_pair_path_edge.npy'))

        if self.datatype == "train":
            self.data_len = ins_num = self.data_word.shape[0] if not config.debug else 100
        else:
            self.data_len = ins_num = self.data_word.shape[0] if not config.debug else 30

        pbar = list(range(self.data_len))
        batchsize = 12
        parameter = (self.max_length,self.load_data_type,self.relation_num,self.word_size,
                    self.dis2idx,self.sent_limit,self.entity_limit,self.mention_limit,
                    self.char_limit,self.edges,self.entity_info_len,config.metapath,config.type_path_num)
        pbatch = list(map(lambda x: (x,self.data_file[x],parameter),pbar))
        pbatch = [pbatch[i:i+batchsize] for i in range(0,len(pbar),batchsize)]
        with Pool(processes=config.worker_num ) as pthread:
            data = pthread.map(cirle_reader,pbatch)
        for insdata in data:
            for item in insdata:
                self.data_contain[item["index"]] = item
        print("Finish reading.datalen:{} consume time:{}".format(self.data_len,time.time()-start_time))

    def __getitem__(self,index):
        ins = self.data_file[index] 

        L = len(ins['vertexSet'])
        title = ins["title"]

        return {"word":self.data_word[index],"deal_data":self.data_contain[index],
        "pos":self.data_pos[index],"ner":self.data_ner[index],"char":self.data_char[index],
        "L":L,"title":title,"index":index,
        "node_info":self.data_node_info[index],"node_adj":self.data_node_adj[index],
        "pair_path":self.data_h_t_pair_path[index],"pair_path_len":self.data_h_t_pair_path_len[index],
        "pair_path_edge":self.data_h_t_pair_path_edge[index]}

    def __len__(self):
        return self.data_len
    def sample_instance_example(self,label,L_vertex):
        batch_size = len(L_vertex)
        mask = torch.Tensor(batch_size,self.entity_limit,self.entity_limit).cuda()
        mask.fill_(0)
        for btind in range(len(L_vertex)):
            evi_ind = torch.sum(label[btind]>0,dim=-1)
            hind,tind = torch.where(evi_ind>0)
            bind = torch.ones_like(hind)*btind
            mask[bind,hind,tind] = 1

            evi_num = torch.sum(evi_ind).item()
            if evi_num ==0:
                evi_num = 2
            seq_ind = torch.arange(self.entity_limit).cuda()
            evi_ind[seq_ind,seq_ind] = 1
            entity_len = L_vertex[btind]
            evi_ind[entity_len:,:] = 1
            evi_ind[:,entity_len:] = 1
            
            hind,tind = torch.where(evi_ind==0)
            bind = torch.ones_like(hind)*btind
            sel_size = min(evi_num*3,bind.shape[0])
            sel_ind = random.sample(list(range(bind.shape[0])),sel_size)
            mask[bind[sel_ind],hind[sel_ind],tind[sel_ind]] = 1
        return mask
    def sample_head_example(self,label,L_vertex):
        batch_size = len(L_vertex)
        mask = torch.Tensor(batch_size,self.entity_limit,self.entity_limit).cuda()
        mask.fill_(0)
        for btind in range(len(L_vertex)):

            entity_len = L_vertex[btind]

            head_ind = torch.sum(torch.sum(label[btind]>0,dim=-1),dim=-1)
            hind, = torch.where(head_ind>0)
            bind = torch.ones_like(hind)*btind
            head_sel_num = int(hind.shape[0]/2)
            sel_ind = random.sample(list(range(hind.shape[0])),head_sel_num)
            mask[bind[sel_ind],hind[sel_ind],:entity_len] = 1

            if head_sel_num ==0:
                head_sel_num = 2
            head_ind[entity_len:] = 1
            hind, = torch.where(head_ind==0)
            bind = torch.ones_like(hind)*btind
            sel_size = min(head_sel_num,hind.shape[0])
            sel_ind = random.sample(list(range(hind.shape[0])),sel_size)
            mask[bind[sel_ind],hind[sel_ind],:entity_len] = 1

            seq_ind = torch.arange(self.entity_limit).cuda()
            bind = torch.ones_like(seq_ind)*btind
            mask[bind,seq_ind,seq_ind] = 0
        return mask
    def sample_test_example(self,L_vertex):
        batch_size = len(L_vertex)
        mask = torch.Tensor(batch_size,self.entity_limit,self.entity_limit).cuda()
        mask.fill_(0)
        for btind in range(len(L_vertex)):
            entity_len = L_vertex[btind]
            mask[btind,:entity_len,:entity_len] = 1

            seq_ind = torch.arange(self.entity_limit).cuda()
            mask[btind,seq_ind,seq_ind] = 0
        return mask

    def get_batch(self,list_data):

        batch_size = len(list_data)

        context_idxs = torch.LongTensor(batch_size, self.max_length).cuda()
        context_masks = torch.LongTensor(batch_size, self.max_length).cuda()
        context_pos = torch.LongTensor(batch_size, self.max_length).cuda()
        context_ner = torch.LongTensor(batch_size, self.max_length).cuda()
        # input
        entity_info = torch.LongTensor(batch_size, self.entity_limit, self.mention_limit,
                                                                    self.entity_info_len).cuda()
        sent_info = torch.LongTensor(batch_size,self.sent_limit,self.entity_info_len).cuda()
        node_size = torch.LongTensor(batch_size).cuda()
        h_t_pair_label = torch.Tensor(batch_size,self.entity_limit,self.entity_limit,self.relation_num).cuda()
        node_info = torch.LongTensor(batch_size,self.node_limit,self.entity_info_len).cuda()
        node_adj = torch.LongTensor(batch_size,self.node_limit,self.node_limit).cuda()
        h_t_pair_path = torch.LongTensor(batch_size,self.entity_limit,self.entity_limit,9,10).cuda()
        h_t_pair_path_len = torch.LongTensor(batch_size,self.entity_limit,self.entity_limit,9).cuda()
        h_t_pair_path_edge = torch.LongTensor(batch_size,self.entity_limit,self.entity_limit,9,10).cuda()

        for mapping in [entity_info,sent_info,h_t_pair_label]:
            mapping.zero_()

        for mapping in [node_adj,node_info,node_size,h_t_pair_path_len,h_t_pair_path,h_t_pair_path_edge]:
            mapping.zero_()

        list_data.sort(key=lambda x: np.sum(x["word"]>0) , reverse = True)

        L_vertex = []
        titles = []
        indexes = []

        for i,data in enumerate(list_data):

            L_vertex.append(data["L"])
            indexes.append(data["index"])
            titles.append(data["title"])

            context_idxs[i].copy_(torch.from_numpy(data["word"]))
            context_pos[i].copy_(torch.from_numpy(data["pos"]))
            context_ner[i].copy_(torch.from_numpy(data["ner"]))


            node_info[i].copy_(torch.from_numpy(data["node_info"]))
            node_adj[i].copy_(torch.from_numpy(data["node_adj"]))
            h_t_pair_path[i].copy_(torch.from_numpy(data["pair_path"]))
            h_t_pair_path_len[i].copy_(torch.from_numpy(data["pair_path_len"]))
            h_t_pair_path_edge[i].copy_(torch.from_numpy(data["pair_path_edge"]))

            deal_data = data["deal_data"]
            

            #input
            entity_info[i].copy_(torch.from_numpy(deal_data["entity_info"]))
            h_t_pair_label[i].copy_(torch.from_numpy(deal_data["h_t_pair_label"]))
            sent_info[i].copy_(torch.from_numpy(deal_data["sent_info"]))
            node_size[i] = torch.sum(node_info[i,...,0],dim=-1)



        input_lengths = (context_idxs > 0).long().sum(dim=1)
        max_c_len = int(input_lengths.max())
        if self.datatype == "train" and self.sample_data:
            sample_mask = self.sample_instance_example(h_t_pair_label[...,1:],L_vertex)
        else:
            sample_mask = self.sample_test_example(L_vertex)
        entity_mask = (torch.sum(sample_mask,dim=-1)>0).float()


        labels = {
                'h_t_pair_label': h_t_pair_label,
                "sample_mask": sample_mask,
                'L_vertex': L_vertex,
                'titles': titles,
                'indexes': indexes,
                "entity_mask": entity_mask,
            }

        return  {'context_idxs': context_idxs[:, :max_c_len].contiguous(),
                'context_pos': context_pos[:, :max_c_len].contiguous(),
                'context_ner': context_ner[:, :max_c_len].contiguous(),
                'context_masks': context_masks[:, :max_c_len].contiguous(),
                'input_lengths' : input_lengths,
                "sample_mask": sample_mask,
                "sent_info":sent_info,
                "entity_info": entity_info,
                "node_info":node_info,
                "node_adj":node_adj,
                "node_size":node_size,
                "h_t_pair_path":h_t_pair_path,
                "h_t_pair_path_len":h_t_pair_path_len,
                "h_t_pair_path_edge":h_t_pair_path_edge,
                'h_t_pair_label': h_t_pair_label,
                },labels
class PathObj():
    def __init__(self,nodeid1):
        self.cur_node = nodeid1
        self.path = [nodeid1]
        self.edge_path = [0]
        self.loss = 0
    def forward(self,nodeid,edge,loss):
        new_path = PathObj(nodeid)
        new_path.path = copy.deepcopy(self.path)
        new_path.path.append(nodeid)
        new_path.edge_path = copy.deepcopy(self.edge_path)
        new_path.edge_path.append(edge)
        new_path.loss = self.loss + loss
        return new_path

class MetaPathfinder():
    def __init__(self,edges,metapath_ori,type_path_num):
        self.edges = edges
        self.edge_diss = [1]*len(edges)
        self.maxloss = 10
        self.metapath = []
        for path in metapath_ori:
            tmp_id = []
            for p_id in path:
                ind = edges.index(p_id)
                ind += 1
                tmp_id.append(ind)
            self.metapath.append(tmp_id)
        self.type_path_num = type_path_num
    def find_path(self,adj,nodeinfo,path_id,startid,endid):
        pathset = [PathObj(startid)]
        path_node_type = [nodeinfo[startid,-1]]
        for edge_type in self.metapath[path_id]:
            new_pathset = []
            for node in pathset:
                cur_node_id = node.cur_node
                for i in range(adj.shape[0]):
                    if adj[cur_node_id,i] != edge_type or i in node.path or i == cur_node_id:
                        continue
                    new_path = node.forward(i,adj[cur_node_id,i],0)
                    new_pathset.append(new_path)
            pathset = copy.deepcopy(new_pathset)
            if pathset!=[]:
                path_node_type.append(nodeinfo[pathset[0].cur_node,-1])
        result_set = []
        result_edge = []
        result_type = []
        for path in pathset:
            if path.path[-1] == endid:
                result_set.append(path.path)
                result_edge.append([0]+self.metapath[path_id])
                result_type.append(path_node_type)
        return result_set,result_type,result_edge
    def form_path(self,adjacency,nodeinfo,L,entity_limit):

        path_len = len(self.metapath)*self.type_path_num
        h_t_pair_path = np.zeros((entity_limit,entity_limit,path_len,10),dtype=np.int)
        h_t_pair_path_edge = np.zeros((entity_limit,entity_limit,path_len,10),dtype=np.int)
        h_t_pair_path_len = np.zeros((entity_limit,entity_limit,path_len),dtype=np.int)

        for s_id in range(L-1):
            for e_id in range(s_id+1,L):
                for path_id in range(len(self.metapath)):
                    # pdb.set_trace()
                    pathset,path_type,path_edge = self.find_path(adjacency,nodeinfo,path_id,s_id,e_id)
                    # assert not (path_id==2 and len(pathset)==0)
                    for i in range(min(len(pathset),self.type_path_num)):
                        pathlen = min(10,len(pathset[i]))
                        h_t_pair_path_len[s_id,e_id,i+path_id*self.type_path_num] = pathlen
                        h_t_pair_path_len[e_id,s_id,i+path_id*self.type_path_num] = pathlen
                        h_t_pair_path[s_id,e_id,i+path_id*self.type_path_num,:pathlen] = \
                                                                    np.array(pathset[i][:pathlen])
                        h_t_pair_path[e_id,s_id,i+path_id*self.type_path_num,:pathlen] = \
                                                                np.array(pathset[i][:pathlen][::-1])
                        h_t_pair_path_edge[s_id,e_id,i+path_id*self.type_path_num,:pathlen] = \
                                                                    np.array(path_edge[i][:pathlen])
                        h_t_pair_path_edge[e_id,s_id,i+path_id*self.type_path_num,:pathlen] = \
                                                                np.array(path_edge[i][:pathlen][::-1])
        return h_t_pair_path,h_t_pair_path_len,h_t_pair_path_edge


def GraphForm(entity_info,sent_info,edges):

    mention_num = np.sum(entity_info[:,:,0],axis=-1)
    nodesinfo = entity_info[mention_num>0,0]
    nodesinfo[:,-1] = ENTITY_IND

    select = entity_info[:,:,0]>0
    ins_info = entity_info[select]
    nodesinfo = np.concatenate((nodesinfo,ins_info),axis=0)

    select = sent_info[:,0]>0
    ins_info = sent_info[select]
    nodesinfo = np.concatenate((nodesinfo,ins_info),axis=0)

    xv, yv = np.meshgrid(np.arange(nodesinfo.shape[0]), np.arange(nodesinfo.shape[0]), indexing='ij')

    r_id, c_id = nodesinfo[xv, -1], nodesinfo[yv, -1]
    r_Eid, c_Eid = nodesinfo[xv, 4], nodesinfo[yv, 4]
    r_Sid, c_Sid = nodesinfo[xv, 6], nodesinfo[yv, 6]

    adjacency = np.full((r_id.shape[0], r_id.shape[0]), 0, 'i')

    # if 'FULL' in self.edges:
    #     adjacency = np.full(adjacency.shape, 1, 'i')

    # mention-mention
    if "MM" in edges:
        ind = edges.index("MM")
        ind += 1
        adjacency = np.where((r_id == MENTION_IND) & (c_id == MENTION_IND) & (r_Sid == c_Sid), ind, adjacency)  # in same sentence

    # mention-mention
    if "CO" in edges:
        ind = edges.index("CO")
        ind += 1
        adjacency = np.where((r_id == MENTION_IND) & (c_id == MENTION_IND) & (r_Eid == c_Eid), ind, adjacency)  # in same sentence

    # entity-mention
    if "EM" in edges or "ME" in edges:
        ind = edges.index("EM") if "EM" in edges else edges.index("ME")
        ind += 1
        adjacency = np.where((r_id == ENTITY_IND) & (c_id == MENTION_IND) & (r_Eid == c_Eid), ind, adjacency)  # belongs to entity
        adjacency = np.where((r_id == MENTION_IND) & (c_id == ENTITY_IND) & (r_Eid == c_Eid), ind, adjacency)

    # sentence-sentence (in order)
    if "SS" in edges:
        ind = edges.index("SS")
        ind += 1
        adjacency = np.where((r_id == SENTENCE_IND) & (c_id == SENTENCE_IND) & (r_Sid == c_Sid - 1), ind, adjacency)
        adjacency = np.where((r_id == SENTENCE_IND) & (c_id == SENTENCE_IND) & (c_Sid == r_Sid - 1), ind, adjacency)

        # sentence-sentence (direct + indirect)
        adjacency = np.where((r_id == SENTENCE_IND) & (c_id == SENTENCE_IND), ind, adjacency)

    # mention-sentence
    if "MS" in edges or "SM" in edges:
        ind = edges.index("SM") if "SM" in edges else edges.index("MS")
        ind += 1
        adjacency = np.where((r_id == MENTION_IND) & (c_id == SENTENCE_IND) & (r_Sid == c_Sid), ind, adjacency)  # belongs to sentence
        adjacency = np.where((r_id == SENTENCE_IND) & (c_id == MENTION_IND) & (r_Sid == c_Sid), ind, adjacency)
    # entity-entity
    if "EE" in edges:
        ind = edges.index("EE")
        ind += 1
        adjacency = np.where((r_id == ENTITY_IND) & (c_id == ENTITY_IND), ind, adjacency)
    if ('ES' in edges) or ('SE' in edges):
        ind = edges.index("SE") if "SE" in edges else edges.index("ES")
        ind += 1
        # entity-sentence
        for x, y in zip(xv.ravel(), yv.ravel()):
            if nodesinfo[x, -1] == ENTITY_IND and nodesinfo[y, -1] == SENTENCE_IND:
                z = np.where((r_Eid == nodesinfo[x, 4]) & (r_id == MENTION_IND) & (c_id == SENTENCE_IND) & (c_Sid == nodesinfo[y, 6]))

                # at least one M in S
                temp_ = np.zeros(adjacency.shape)
                temp_ = np.where((r_id == MENTION_IND) & (c_id == SENTENCE_IND) & (r_Sid == c_Sid), 1, temp_)
                temp_ = np.where((r_id == SENTENCE_IND) & (c_id == MENTION_IND) & (r_Sid == c_Sid), 1, temp_)
                adjacency[x, y] = ind if (temp_[z] > 0).any() else 0
                adjacency[y, x] = ind if (temp_[z] > 0).any() else 0

    # self-loops = 0 [always]
    adjacency[np.arange(r_id.shape[0]), np.arange(r_id.shape[0])] = 0

    return nodesinfo,adjacency





