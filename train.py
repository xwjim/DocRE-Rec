import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import argparse
import random
import torch

def str2bool(argstr):
    flag = argstr.lower()
    if flag != "true" and flag!="false":
        raise("Input Error")
    if flag == 'true':
        return True
    else:
        return False
    
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'MPR', help = 'name of the model')
parser.add_argument('--save_name', type = str)

parser.add_argument('--train_prefix', type = str, default = 'dev_train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')
parser.add_argument('--lr', type = float, default = 1e-4)
parser.add_argument('--batch_size', type = int, default = 20)
parser.add_argument('--max_epoch', type = int, default = 300)
parser.add_argument('--debug', type = str2bool, default = "False")

parser.add_argument('--lr_decay', type=float, default=0.98)
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

parser.add_argument('--sample_data', type = str2bool, default = "False")

parser.add_argument('--freeze_model', type = str2bool, default = "False")
parser.add_argument('--use_graph', type = str2bool, default = "True")
parser.add_argument('--graph_iter', type = int, default = 2)
parser.add_argument('--graph_type', type = str, default = "atten")
parser.add_argument('--graph_head', type = int, default = 4)
parser.add_argument('--use_entity_type', type = str2bool, default = "True")
parser.add_argument('--use_distance', type = str2bool, default = "True")
parser.add_argument('--use_pos_embed', type = str2bool, default = "False")
parser.add_argument('--load_model', type = str2bool, default = "False")
parser.add_argument('--test_epoch', type = int, default = 1)

parser.add_argument('--lstm_drop', type = int, default = 0.2)
parser.add_argument('--graph_drop', type = int, default = 0.2)
parser.add_argument('--rel_theta', type = float, default = 0.5)
parser.add_argument('--seq_theta', type = float, default = 0.5)
parser.add_argument('--word_embed_hidden', type = int, default = 128)
parser.add_argument('--graph_out_hidden', type = int, default = 256)
parser.add_argument('--type_path_num', type = int, default = 3)

# need to rum gen_graph_data.py
parser.add_argument('--edges', type = list, default = ['MM', 'SS', 'ME', 'MS', 'ES','CO'])
parser.add_argument('--metapath', type = list, default = [["ME","MM","ME"],["ME","MM","CO","MM","ME"],["ES","SS","ES"]])

parser.add_argument('--use_wandb', type = str2bool, default = "False")
parser.add_argument('--data_path', type = str, default = './prepro_data')
parser.add_argument('--load_data_type', type = str, default = "all")
parser.add_argument('--eval_model', type = str2bool, default = "False")
parser.add_argument('--wandb_name', type = str, default = "")
parser.add_argument('--seq_loss_weight', type = float, default = 0.4)
parser.add_argument('--seq_eval_weight', type = float, default = 0.4)

args = parser.parse_args()
args.setparameter = str(sys.argv)
if not args.use_wandb:
    os.environ['WANDB_MODE'] = 'dryrun'
model = {
    "DynGraph": models.DynGraphLayer,
}

if args.eval_model:
    con = config.Config(args)
    con.load_test_data()
    con.testall(model[args.model_name], args.save_name)#, args.ignore_input_theta)

else:
    con = config.Config(args)
    con.load_train_data()
    con.load_test_data()
    con.train(model[args.model_name], args.save_name)


