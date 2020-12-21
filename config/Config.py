# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import wandb
import datetime
import json
import sys
from collections import defaultdict
import torch.utils.data as Data
from .dataloader import SelfData


IGNORE_INDEX = -100


class Config(object):
    def __init__(self, args):
        self.data_path = args.data_path
        # self.data_path = './glove_data'
        self.max_length = 512
        self.pos_num = 2 * self.max_length
        self.entity_num = self.max_length
        self.relation_num = 97
        self.multi_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.bce = nn.BCELoss(reduction="none")
        self.single_loss = nn.CrossEntropyLoss(reduction="none")
        self.worker_num = 4

        self.lr = args.lr
        self.opt_method = 'Adam'
        self.optimizer = None

        self.pos_embed_size = 20
        self.entity_type_size = 20
        self.abs_dis_size = 20
        self.node_type_size = 20
        self.max_epoch = args.max_epoch
        self.word_size = 100

        self.checkpoint_dir = './checkpoint'
        self.fig_result_dir = './fig_result'
        self.test_epoch = args.test_epoch

        self.debug = args.debug
        self.freeze_model = args.freeze_model
        self.sample_data = args.sample_data
        self.use_graph = args.use_graph
        self.graph_iter = args.graph_iter
        self.graph_type = args.graph_type#"atten"#"gate"#"atten","aggcn"
        self.graph_head = args.graph_head

        self.use_entity_type = args.use_entity_type
        self.use_distance = args.use_distance
        self.use_pos_embed = args.use_pos_embed

        self.batch_size = args.batch_size if not self.debug else 2

        self.lstm_drop = args.lstm_drop  # for lstm
        self.graph_drop = args.graph_drop
        self.rel_theta = args.rel_theta
        self.seq_theta = args.seq_theta
        self.word_embed_hidden = args.word_embed_hidden
        self.graph_out_hidden = args.graph_out_hidden
        self.edges = args.edges
        self.metapath = args.metapath
        self.type_path_num = args.type_path_num
        self.load_model = args.load_model
        self.load_data_type = args.load_data_type
        self.max_grad_norm = args.max_grad_norm # gradient clipping

        self.test_batch_size = args.batch_size
        self.char_limit = 16
        self.sent_limit = 25
        self.entity_limit = 45
        self.mention_limit = 10
        self.node_limit = 130
        self.entity_info_len = 8
        self.train_h_t_limit = 300
        self.test_h_t_limit = 1800
        #self.combined_sent_limit = 200
        self.dis2idx = np.zeros((512), dtype='int64')
        self.dis2idx[1] = 1
        self.dis2idx[2:] = 2
        self.dis2idx[4:] = 3
        self.dis2idx[8:] = 4
        self.dis2idx[16:] = 5
        self.dis2idx[32:] = 6
        self.dis2idx[64:] = 7
        self.dis2idx[128:] = 8
        self.dis2idx[256:] = 9
        self.dis_size = 20

        self.train_prefix = args.train_prefix
        self.test_prefix = args.test_prefix

        self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
        self.id2rel = {v: k for k,v in self.rel2id.items()}
        self.path_atten_data = []
        self.setparameter = args.setparameter
        self.seq_loss_weight = args.seq_loss_weight
        self.seq_eval_weight = args.seq_eval_weight

        self.log_rate = int(1500.0/self.batch_size)
        if self.debug:
            self.log_rate = 10
        wandb_name = args.wandb_name + "_" if args.wandb_name != "" else ""
        if self.debug:
            wandb_name +="Debug_" + args.model_name
        elif args.eval_model:
            wandb_name +="Eval_" + args.model_name
        else:
            wandb_name += args.model_name
        if self.use_graph:
            wandb_name += "_graph_"+ self.graph_type + "_" + str(self.graph_iter)

        wandb.init(name=wandb_name,project="meta",config={"batch_size": self.batch_size,
            "epoch":self.max_epoch,"lr":self.lr,"model_name": args.model_name,
            "use_graph":self.use_graph,"freeze_model":self.freeze_model,
            "graph_type":self.graph_type,"graph_iter":self.graph_iter,
            "rel_theta":self.rel_theta,"edges":self.edges,
            "metapath":self.metapath,"type_path_num":self.type_path_num,
            "load_data_type":self.load_data_type,"max_grad_norm":self.max_grad_norm,
            "graph_out_hidden":self.graph_out_hidden,
            "word_embed_hidden":self.word_embed_hidden,"entity_type_size":self.entity_type_size,
            "sample_data":self.sample_data,
            "use_entity_type":self.use_entity_type,"use_pos_embed":self.use_pos_embed,
            "seq_eval_weight":self.seq_eval_weight,
            "use_distance":self.use_distance,"seq_loss_weight":self.seq_loss_weight,
            "train_data":self.train_prefix,"test_data":self.test_prefix})

        if not os.path.exists("log"):
            os.mkdir("log")
        
        self.loss_type_str = ["Re"]
        self.loss_type_str.append("Seq")

        self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))

    def load_train_data(self):
        shuffle = not self.debug
        self.train_data = SelfData(self,"train")
        self.train_load = Data.DataLoader(self.train_data, batch_size=self.batch_size,
            shuffle=shuffle, sampler=None,batch_sampler=None, num_workers=0, collate_fn=self.train_data.get_batch,
            pin_memory=False, drop_last=False, timeout=0,worker_init_fn=None)
    def load_test_data(self):
        shuffle = not self.debug
        self.test_data = SelfData(self,"test")
        self.test_load = Data.DataLoader(self.test_data, batch_size=self.batch_size,
            shuffle=shuffle, sampler=None,batch_sampler=None, num_workers=0, collate_fn=self.test_data.get_batch,
            pin_memory=False, drop_last=False, timeout=0,worker_init_fn=None)
        
    def criterion(self,inputs,output,labels):
        r_predict = output["predict_re"]
        r_label = labels["h_t_pair_label"]
        pair_mask = labels["sample_mask"]

        loss_re = torch.sum(self.multi_loss(r_predict[...,1:], r_label[...,1:])*pair_mask.unsqueeze(-1))
        loss_re = loss_re/(torch.sum(pair_mask)*self.relation_num)
            
        seq_mask = output["seq_mask"]
        adj_mask = self.get_adj_mask(inputs,output)
        rel_mask = (torch.sum(labels["h_t_pair_label"][...,1:],dim=-1)>0)

        mask_pred = output["seq_pred"].masked_fill(~adj_mask,-1e9)
        mask_pred = torch.softmax(mask_pred,dim=-1)
        seq_pred = mask_pred.view(-1,mask_pred.shape[-1])[torch.arange(output["seq_truth"].view(-1).shape[0]),output["seq_truth"].view(-1)].view(output["seq_truth"].shape)
        seq_truth = rel_mask.float().unsqueeze(dim=-1).repeat(1,1,1,seq_mask.shape[-1])
        mask = torch.sum(seq_mask,dim=-1)
        mask = mask.masked_fill(mask == 0,1)
        seq_loss = torch.sum(self.bce(seq_pred,seq_truth)*seq_mask.float(),dim=-1)
        seq_loss = torch.sum(seq_loss)/(torch.sum(seq_mask))

        loss_seq = seq_loss * self.seq_loss_weight/self.relation_num
        loss = loss_re + loss_seq

        return loss,{"loss_re":loss_re,"loss_seq":loss_seq}
    def interface(self,inputs,output,labels,loss_dict,type="train"):
        loss_re = loss_dict["loss_re"]
        # head label
        predict_re = output["predict_re"]
        
        loss_seq = loss_dict["loss_seq"]
        seq_mask = output["seq_mask"]
        seq_truth = output["seq_truth"]
        adj_mask = self.get_adj_mask(inputs,output)
        seq_pred = output["seq_pred"]
        seq_pred = seq_pred.masked_fill(~adj_mask,-1e9)
        seq_pred = torch.softmax(seq_pred,dim=-1)
        seq_pred = seq_pred.view(-1,seq_pred.shape[-1])[torch.arange(output["seq_truth"].view(-1).shape[0]),output["seq_truth"].view(-1)].view(output["seq_truth"].shape)

        seq_pred = seq_pred.masked_fill(~seq_mask,1)
        seq_pred = torch.div(torch.sum(torch.log(seq_pred)*seq_mask,dim=-1),torch.sum(seq_mask,dim=-1))

        seq_label = (seq_pred>self.seq_theta).long()
        rel_mask = (torch.sum(labels["h_t_pair_label"][...,1:],dim=-1)>0)
        seq_truth = rel_mask.float()
        if type == "train":
            self.train_metric["Seq"].record(loss_seq,seq_label,seq_truth,\
                        labels["sample_mask"])
        elif type == "test":
            self.test_metric["Seq"].roc_record(loss_seq,seq_pred,seq_truth,\
                        labels["sample_mask"])
        else:
            raise("error")
        
        ## Relation
        label = labels["h_t_pair_label"]

        scores = torch.log(torch.sigmoid(predict_re)) + self.seq_eval_weight*seq_pred.unsqueeze(dim=-1)
        pre_re_label = (scores>self.rel_theta).long()

        mask = labels["sample_mask"].unsqueeze(-1)
        if type == "train":
            self.train_metric["Re"].record(loss_re,pre_re_label[:,:,:,1:],label[:,:,:,1:],mask)
        elif type == "test":
            self.test_metric["Re"].roc_record(loss_re,scores[:,:,:,1:],label[:,:,:,1:],labels["sample_mask"])
        else:
            raise("error")

        return pre_re_label

    def get_adj_mask(self,inputs,output):
        adj = inputs["node_adj"]
        batchind = torch.zeros_like(output["seq_truth"],dtype=torch.long)
        for i in range(batchind.shape[0]):
            batchind[i] = i
        adj_ext = torch.ones(adj.shape[0],adj.shape[1]+1,adj.shape[2]+1).cuda()
        adj_ext[:,1:,1:] = adj
        # startnode = torch.arange(self.entity_limit)[None,...,None].repeat(adj.shape[0],1,self.entity_limit).unsqueeze(dim=-1).cuda()+1
        startnode = torch.zeros_like(output["seq_truth"][...,0]).cuda().unsqueeze(-1)
        seq_ext = torch.cat((startnode,output["seq_truth"][...,:-1]),dim=-1).long()
        adj_mask = adj_ext[batchind,seq_ext]>0
        seq_mask = output["seq_mask"]
        adj_mask[~seq_mask] = False
        return adj_mask


            
    def train(self, model_pattern, model_name):
        ori_model = model_pattern(config = self)
        wandb.watch(ori_model)
        ori_model.cuda()
        if self.load_model:
            ori_model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, model_name)))
        model = nn.DataParallel(ori_model)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=self.lr)#,weight_decay=self.weight_decay)

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        best_auc = 0.0
        best_f1 = 0.0
        best_theta = 0.0
        best_recall = 0.0
        best_epoch = 0
        global_step = 0

        model.train()

        total_loss = 0
        start_time = time.time()

        self.logger = Logger(model_name)

        self.logger.logging(self.setparameter)
        self.train_metric = {}
        for name in self.loss_type_str:
            self.train_metric[name] = Metrics(name + " Train",self.logger)

        for epoch in range(self.max_epoch):

            for loss_ins in self.train_metric.values():
                loss_ins.reset()

            for inputs,labels in self.train_load:

                output = model(inputs,global_step)
                predict_re = output["predict_re"]

                ## Loss
                loss,loss_dict = self.criterion(inputs,output,labels)
                self.interface(inputs,output,labels,loss_dict,"train")

                if torch.sum(torch.isnan(loss)).item() != 0:
                    pdb.set_trace()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()

                global_step += 1

                if global_step%self.log_rate == 0:
                    for loss_ins in self.train_metric.values():
                        loss_ins.cal_metric(global_step,log=False)

            self.logger.logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - start_time))
            for loss_ins in self.train_metric.values():
                loss_ins.cal_metric(global_step)
            if (epoch+1) % self.test_epoch == 0:
                model.eval()
                test_acc_re,test_recall_re,test_f1_re,theta = self.test(model, model_name,
                                            epoch=epoch,global_step=global_step)
                model.train()

                if test_f1_re > best_f1:
                    best_f1 = test_f1_re
                    best_auc = test_acc_re
                    best_recall = test_recall_re
                    best_theta = theta
                    best_epoch = epoch
                    path = os.path.join(self.checkpoint_dir, model_name)
                    torch.save(ori_model.state_dict(), path)
                self.logger.logging("best f1:{:8.3f}auc:{:8.3f}recall:{:8.3f}  epoch:{:d}".format(best_f1,
                                                    best_auc,best_recall,best_epoch))

            start_time = time.time()

        self.logger.logging("Finish training")
        self.logger.logging("Best epoch = %d | theta = %f  auc = %f recall = %f f1 = %f" % (best_epoch, best_theta,
                                                                        best_auc,best_recall ,best_f1))
        self.logger.logging("Storing best result...")
        self.logger.logging("Finish storing")


    def test(self, model, model_name,epoch=0,global_step=0,gen_output=False,send_data=True):
        data_idx = 0
        eval_start_time = time.time()
        # test_result_ignore = []
        total_recall_ignore = 0

        test_result = []
        total_recall = 0
        top1_acc = have_label = 0

        predicted_as_zero = 0
        total_ins_num = 0

        test_result = []
        exist_res = []
        rel_res = []
        evi_res = []

        self.test_metric = {}
        for name in self.loss_type_str:
            self.test_metric[name] = Metrics(name + " Test",self.logger)
            self.test_metric[name].reset()

        for inputs,labels in self.test_load:
            with torch.no_grad():

                titles = labels['titles']
                indexes = labels['indexes']
                # if 647 in indexes:
                #     pdb.set_trace()
                L_vertex = labels["L_vertex"]

                output = model(inputs)

                ## Exist
                loss,loss_dict = self.criterion(inputs,output,labels)
                pre_re_label=self.interface(inputs,output,labels,loss_dict,"test")
                
                if gen_output:
                    pre_re_label = pre_re_label.data.cpu().numpy()
                    for i in range(len(titles)):
                        for h_idx in range(L_vertex[i]):
                            for t_idx in range(L_vertex[i]):
                                if h_idx == t_idx:
                                    continue
                                for label_idx in range(1,self.relation_num):
                                    if pre_re_label[i,h_idx,t_idx,label_idx]>0:
                                        test_result.append({"title":titles[i],"h_idx":h_idx,"t_idx":t_idx,"r":self.id2rel[label_idx]})

        if gen_output:
            json.dump(test_result, open(self.test_prefix + "_index.json", "w"))

        self.logger.logging('-' * 89)
        self.logger.logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
        for loss_type,loss_ins in self.test_metric.items():
            if loss_type == "Re":
                test_acc_re,test_recall_re,test_f1_re,theta = loss_ins.cal_roc_metric(global_step,send_data=send_data)
            else:
                loss_ins.cal_roc_metric(global_step,send_data=send_data)
        self.logger.logging('-' * 89)


        return test_acc_re,test_recall_re,test_f1_re,theta


    def testall(self, model_pattern, model_name):#, ignore_input_theta):

        self.logger = Logger(model_name)

        self.logger.logging(self.setparameter)

        model = model_pattern(config = self)

        model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, model_name)))
        model.cuda()
        model.eval()
        #self.test_anylyse(model, model_name, True, input_theta)
        test_acc_re,test_recall_re,test_f1_re,theta = self.test(model, model_name,gen_output=True,
                                                        send_data=False)
        self.logger.logging("Test Re Acc:{:8.3f}Recall:{:8.3f}F1:{:8.3f}".format(test_acc_re,test_recall_re,test_f1_re))

class Metrics():
    def __init__(self,prefix,logout):
        self.prefix = prefix
        self.total_acc = 0
        self.total_predict = 0
        self.total_truth = 0
        self.total_loss = 0
        self.batch_cnt = 0
        self.logger = logout
        self.res = []
        self.res1 = []
    def record(self,loss,predict,labels,mask,binmode=True):
        acc,predict_truth,total_truth = self.cal_bin_data(predict,labels,mask)
        self.total_predict += predict_truth
        self.total_acc += acc
        self.total_truth += total_truth
        self.total_loss += loss.item()
        self.batch_cnt += 1
    def cal_bin_data(self,predict,label,mask):
        acc = torch.sum((torch.gt(label,0) & torch.eq(predict,label.long())).float()*mask) #torch.gt(output_exist,0)
        predict_truth = torch.sum(torch.gt(predict,0).float()*mask).item()
        total_truth = torch.sum(torch.gt(label,0).float()*mask).item()
        return acc.item(),predict_truth,total_truth
    def reset(self):
        self.total_acc = 0
        self.total_predict = 0
        self.total_truth = 0
        self.total_loss = 0
        self.batch_cnt = 0
        self.res = []
        self.res1 = []
    def cal_metric(self,global_step,log=True,send_data=True):
        if self.batch_cnt == 0:
            return
        acc = 1e-9 if int(self.total_predict)==0 else 1.0*self.total_acc/self.total_predict+1e-9
        recall = 1e-9 if int(self.total_truth)==0 else 1.0*self.total_acc/self.total_truth+1e-9
        f1 = 2*acc*recall/(acc+recall)
        loss = self.total_loss/self.batch_cnt
        if send_data:
            wandb.log({ self.prefix+'accuracy': acc, 
                    self.prefix+"recall":recall,
                    self.prefix+'f1': f1,
                    self.prefix+"loss":loss},step=global_step)
        if log:
            self.logger.logging("{:15} Loss{:8.3f} Predict:{:8d}Total:{:8d}True:{:8d} Acc:{:8.3f}%Recall:{:8.3f}%F1:{:8.3f}%".format(self.prefix,loss,int(self.total_predict),int(self.total_truth),int(self.total_acc),100*acc,100*recall,100*f1))
        return acc,recall,f1
    def roc_record(self,loss,scores,label,mask,exist_label=None):
        self.total_loss += loss.item()
        if exist_label is not None:
            b_ind,h_ind,t_ind = torch.where(mask*exist_label)
            self.res.append(np.hstack((scores[b_ind,h_ind,t_ind].cpu().numpy().reshape(-1,1),
                        label[b_ind,h_ind,t_ind].cpu().numpy().reshape(-1,1))))
            b_ind,h_ind,t_ind = torch.where(mask*(1-exist_label))
            self.res1.append(np.hstack((scores[b_ind,h_ind,t_ind].cpu().numpy().reshape(-1,1),
                        label[b_ind,h_ind,t_ind].cpu().numpy().reshape(-1,1))))
        else:
            b_ind,h_ind,t_ind = torch.where(mask)
            self.res.append(np.hstack((scores[b_ind,h_ind,t_ind].cpu().numpy().reshape(-1,1),
                        label[b_ind,h_ind,t_ind].cpu().numpy().reshape(-1,1))))
        self.batch_cnt += 1
    def cal_roc_metric(self,global_step,log=True,send_data=True):
        if self.res == []:
            self.cal_metric(global_step,log,send_data)
            return
        if self.res1 == []:
            rel_res = np.vstack(self.res)
            theta,acc,recall,f1 = roc_cal(rel_res,prefix=self.prefix)
        else:
            rel_res = np.vstack(self.res)
            correct,predict,recall = roc_cal(rel_res,prefix=self.prefix,out_type="data")
            rel_res = np.vstack(self.res1)
            theta,acc,recall,f1 = roc_cal(rel_res,prefix=self.prefix,his_acc=correct,his_pre=predict,his_recall=recall)
        loss = self.total_loss/self.batch_cnt
        if send_data:
            wandb.log({ self.prefix+'accuracy': acc, 
                    self.prefix+"recall":recall,
                    self.prefix+'f1': f1,
                    self.prefix+"loss":loss},step=global_step)
        if log:
            self.logger.logging("{:15} Loss{:8.3f} Theta:{:8.3f}{:29} Acc:{:8.3f}%Recall:{:8.3f}%F1:{:8.3f}%".format(self.prefix,loss,theta," ",100*acc,100*recall,100*f1))
        return acc,recall,f1,theta
def roc_cal(res,prefix = "test",his_acc = 0,his_pre = 0,his_recall = 0,out_type="metric"):

    res = res[np.argsort(-res[:,0])]
    pr_x = []
    pr_y = []
    correct = np.cumsum(res[:,1]) + his_acc

    total_recall = np.sum(res[:,1]) + his_recall
    if total_recall == 0:
        total_recall = 1  # for test

    total_predict = np.arange(1,len(res)+1) + his_pre
    
    pr_y = correct/total_predict
    pr_x = correct/total_recall
    # for i, item in enumerate(res):
    #     correct += item[1]
    #     pr_y.append(float(correct) / (i + 1))
    #     pr_x.append(float(correct) / total_recall)

    f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
    f1 = f1_arr.max()
    f1_pos = f1_arr.argmax()
    theta = res[f1_pos][0]
    acc = pr_x[f1_pos]
    recall = pr_y[f1_pos]

    if out_type == "metric":
        return theta,acc,recall,f1
    else:
        return correct[f1_pos],total_predict[f1_pos],total_recall

class Logger():
    def __init__(self,model_name):
        self.model_name = model_name
    def logging(self,s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", self.model_name)), 'a+') as f_log:
                    f_log.write(s + '\n')





