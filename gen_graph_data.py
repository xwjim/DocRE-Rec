import numpy as np
import os
import json
import argparse
from config.dataloader import GraphForm,MetaPathfinder
from config import dataloader
from multiprocessing import Pool
import time

char_limit = 16
sent_limit = 25
entity_limit = 45
mention_limit = 10
node_limit = 130
ENTITY_IND = dataloader.ENTITY_IND
MENTION_IND = dataloader.MENTION_IND
SENTENCE_IND = dataloader.SENTENCE_IND

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type = str, default = "prepro_data")
parser.add_argument('--type_path_num', type = int, default = 3)
parser.add_argument('--use_glove', type = bool, default = False)
parser.add_argument('--worker_num', type = int, default = 8)
parser.add_argument('--edges', type = list, default = ['MM', 'SS', 'ME', 'MS', 'ES','CO'])
parser.add_argument('--metapath', type = list, default = 
                            [["ME","MM","ME"],["ME","MM","CO","MM","ME"],["ES","SS","ES"]])

args = parser.parse_args()
in_path = args.in_path
out_path = args.in_path
type_path_num = args.type_path_num
case_sensitive = False
entity_info_len = 8
worker_num = args.worker_num
edges = args.edges
metapath = args.metapath

char_limit = 16
sent_limit = 25
word_size = 100

def cirle_reader(batch):
    out = []
    for (index,ins) in batch:
        out.append(GraphPredeal(ins,index))
    return out
def GraphPredeal(ins,index):

    # (sent_limit,entity_limit,mention_limit,char_limit,
    #     edges,entity_info_len,metapath,type_path_num)=parameter

    # mask type startid endid sentenceid entityid nodetype
    entity_info = np.zeros((entity_limit,mention_limit,entity_info_len),dtype=np.int)
    # mask type startid endid sentenceid entityid nodetype
    sent_info = np.zeros((sent_limit,entity_info_len),dtype=np.int)

    path_finder = MetaPathfinder(edges,metapath,type_path_num) 


    L = len(ins['vertexSet'])

    for cnt in range(min(len(ins["Ls"])-1,sent_limit)):
        sent_info[cnt] = [1,-100,ins["Ls"][cnt],ins["Ls"][cnt+1],0,0,cnt,SENTENCE_IND]

    for idx in range(min(L,entity_limit)):
        hlist = ins['vertexSet'][idx]
        for cnt in range(min(mention_limit,len(hlist))):
            h = hlist[cnt]
            entity_info[idx,cnt] = [1,ins["vertexSet"][idx][0]["type"],h['pos'][0],h['pos'][1],idx,cnt,h["sent_id"],MENTION_IND]
    # label entitytype pos1 pos2 entity mentioncnt sentid nodetype

    ins_nodesinfo,ins_adjacency = GraphForm(entity_info,sent_info,edges)
    ins_h_t_pair_path,ins_h_t_pair_path_len,ins_h_t_pair_path_edge = path_finder.form_path(
                                                    ins_adjacency,ins_nodesinfo,L,entity_limit)

    return {
        "index":index,
        "node_info":ins_nodesinfo,
        "node_adj":ins_adjacency,
        "h_t_pair_path":ins_h_t_pair_path,
        "h_t_pair_path_len":ins_h_t_pair_path_len,
        "h_t_pair_path_edge":ins_h_t_pair_path_edge
    }


def init(max_length = 512, is_training = True, suffix=''):

    name_prefix = "dev"
    
    ori_data = json.load(open(os.path.join(out_path, name_prefix + suffix + '.json'), "r"))
    start_time = time.time()

    sen_tot = len(ori_data)

    node_info = np.zeros((sen_tot,node_limit,entity_info_len),dtype = np.int64)
    node_adj = np.zeros((sen_tot,node_limit,node_limit),dtype = np.int64)
    h_t_pair_path = np.zeros((sen_tot,entity_limit,entity_limit,9,10),dtype = np.int64)
    h_t_pair_path_len = np.zeros((sen_tot,entity_limit,entity_limit,9),dtype = np.int64)
    h_t_pair_path_edge = np.zeros((sen_tot,entity_limit,entity_limit,9,10),dtype = np.int64)

    # parameter = (sent_limit,entity_limit,mention_limit,
    #             char_limit,edges,entity_info_len,metapath,type_path_num)
    pbar = list(range(len(ori_data)))
    pbatch = list(map(lambda x: (x,ori_data[x]),pbar))
    batchsize = 12
    pbatch = [pbatch[i:i+batchsize] for i in range(0,len(pbar),batchsize)]
    with Pool(processes=worker_num ) as pthread:
        graph_data = pthread.map(cirle_reader,pbatch)
    # for pdata in pbatch:
    #     graph_data = cirle_reader(pdata)
    
    for insdata in graph_data:
        for item in insdata:
            nodesize = min(item["node_info"].shape[0],node_limit)
            node_info[item["index"],:nodesize,:] = item["node_info"][:nodesize,:]
            node_adj[item["index"],:nodesize,:nodesize] = item["node_adj"][:nodesize,:nodesize]
            h_t_pair_path[item["index"]] = item["h_t_pair_path"]
            h_t_pair_path_len[item["index"]] = item["h_t_pair_path_len"]
            h_t_pair_path_edge[item["index"]] = item["h_t_pair_path_edge"]

    print("Finishing processing")
    np.save(os.path.join(out_path, name_prefix + suffix +'_node_info.npy'), node_info)
    np.save(os.path.join(out_path, name_prefix + suffix +'_node_adj.npy'), node_adj)
    np.save(os.path.join(out_path, name_prefix + suffix +"_"+'_h_t_pair_path.npy'), h_t_pair_path)
    np.save(os.path.join(out_path, name_prefix + suffix +"_"+'_h_t_pair_path_len.npy'), h_t_pair_path_len)
    np.save(os.path.join(out_path, name_prefix + suffix +"_"+'_h_t_pair_path_edge.npy'), h_t_pair_path_edge)
    print("Finish saving")
    print("consume time:{}".format(time.time()-start_time))



init(max_length = 512, is_training = False, suffix='_train')
init(max_length = 512, is_training = False, suffix='_dev')
init(max_length = 512, is_training = False, suffix='_test')


