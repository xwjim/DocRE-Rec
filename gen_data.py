import numpy as np
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type = str, default =  "../data")
parser.add_argument('--out_path', type = str, default = "prepro_data")

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path
case_sensitive = False

char_limit = 16
sent_limit = 25
word_size = 100
train_distant_file_name = os.path.join(in_path, 'train_distant.json')
train_annotated_file_name = os.path.join(in_path, 'train_annotated.json')
dev_file_name = os.path.join(in_path, 'dev.json')
test_file_name = os.path.join(in_path, 'test.json')

rel2id = json.load(open(os.path.join(out_path, 'rel2id.json'), "r"))
id2rel = {v:u for u,v in rel2id.items()}
json.dump(id2rel, open(os.path.join(out_path, 'id2rel.json'), "w"))
fact_in_train = set([])
fact_in_dev_train = set([])

# if not args.use_glove:
word2id_name = "word2id.json"
# else:
#     word2id_name = "glove_word2id.json"

def sents_2_idx(sents, word2id):
    #sents_idx = np.zeros([sent_limit, word_size]) + word2id['BLANK']
    sents_idx = []
    start_idx = 0
    for i, sent in enumerate(sents[:sent_limit]):
        sents_idx.append(list(range(start_idx, start_idx+len(sent))))
        start_idx += len(sent)
    return sents_idx

def init(data_file_name, rel2id, max_length = 512, is_training = True, suffix=''):

    ori_data = json.load(open(data_file_name))


    Ma = 0
    Ma_e = 0
    data = []
    intrain = notintrain = notindevtrain = indevtrain = 0
    word2id = json.load(open(os.path.join(out_path, word2id_name)))
    ner2id = json.load(open(os.path.join(out_path, "ner2id.json")))
    for i in range(len(ori_data)):
        Ls = [0]
        L = 0
        for x in ori_data[i]['sents']:
            L += len(x)
            Ls.append(L)

        vertexSet =  ori_data[i]['vertexSet']
        # point position added with sent start position
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet[j])):
                vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])

                sent_id = vertexSet[j][k]['sent_id']
                dl = Ls[sent_id]
                pos1 = vertexSet[j][k]['pos'][0]
                pos2 = vertexSet[j][k]['pos'][1]
                vertexSet[j][k]['pos'] = (pos1+dl, pos2+dl)
                vertexSet[j][k]["type"] = ner2id[vertexSet[j][k]["type"]]

        ori_data[i]['vertexSet'] = vertexSet

        item = {}
        item['vertexSet'] = vertexSet
        labels = ori_data[i].get('labels', [])

        train_triple = set([])
        new_labels = []
        for label in labels:
            rel = label['r']
            assert(rel in rel2id)
            label['r'] = rel2id[label['r']]

            train_triple.add((label['h'], label['t']))


            if suffix=='_train':
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_dev_train.add((n1['name'], n2['name'], rel))


            if is_training:
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_train.add((n1['name'], n2['name'], rel))

            else:
                # fix a bug here
                label['intrain'] = False
                label['indev_train'] = False

                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        if (n1['name'], n2['name'], rel) in fact_in_train:
                            label['intrain'] = True

                        if suffix == '_dev' or suffix == '_test':
                            if (n1['name'], n2['name'], rel) in fact_in_dev_train:
                                label['indev_train'] = True


            new_labels.append(label)

        item['labels'] = new_labels
        item['title'] = ori_data[i]['title']

        na_triple = []
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet)):
                if (j != k):
                    if (j, k) not in train_triple:
                        na_triple.append((j, k))

        item['na_triple'] = na_triple
        item['Ls'] = Ls
        item['sents'] = ori_data[i]['sents']
        item['sents_idx'] = sents_2_idx(ori_data[i]['sents'], word2id)
        data.append(item)

        Ma = max(Ma, len(vertexSet))
        Ma_e = max(Ma_e, len(item['labels']))


    print ('data_len:', len(ori_data))
    # print ('Ma_V', Ma)
    # print ('Ma_e', Ma_e)
    # print (suffix)
    # print ('fact_in_train', len(fact_in_train))
    # print (intrain, notintrain)
    # print ('fact_in_devtrain', len(fact_in_dev_train))
    # print (indevtrain, notindevtrain)


    # saving
    print("Saving files")
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"

    json.dump(data , open(os.path.join(out_path, name_prefix + suffix + '.json'), "w"))

    char2id = json.load(open(os.path.join(out_path, "char2id.json")))
    # id2char= {v:k for k,v in char2id.items()}
    # json.dump(id2char, open("data/id2char.json", "w"))

    word2id = json.load(open(os.path.join(out_path, word2id_name)))

    sen_tot = len(ori_data)
    sen_word = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_pos = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_ner = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_char = np.zeros((sen_tot, max_length, char_limit), dtype = np.int64)

    for i in range(len(ori_data)):
        item = ori_data[i]
        words = []
        for sent in item['sents']:
            words += sent

        for j, word in enumerate(words):
            word = word.lower()

            if j < max_length:
                if word in word2id:
                    sen_word[i][j] = word2id[word]
                else:
                    sen_word[i][j] = word2id['UNK']

            for c_idx, k in enumerate(list(word)):
                if c_idx>=char_limit:
                    break
                sen_char[i,j,c_idx] = char2id.get(k, char2id['UNK'])

        for j in range(j + 1, max_length):
            sen_word[i][j] = word2id['BLANK']

        vertexSet = item['vertexSet']

        for idx, vertex in enumerate(vertexSet, 1):
            for v in vertex:
                sen_pos[i][v['pos'][0]:v['pos'][1]] = idx
                sen_ner[i][v['pos'][0]:v['pos'][1]] = v['type']

    print("Finishing processing")
    np.save(os.path.join(out_path, name_prefix + suffix + '_word.npy'), sen_word)
    np.save(os.path.join(out_path, name_prefix + suffix + '_pos.npy'), sen_pos)
    np.save(os.path.join(out_path, name_prefix + suffix + '_ner.npy'), sen_ner)
    np.save(os.path.join(out_path, name_prefix + suffix + '_char.npy'), sen_char)
    print("Finish saving")



#init(train_distant_file_name, rel2id, max_length = 512, is_training = True, suffix='')
if os.path.exists(train_annotated_file_name):
    init(train_annotated_file_name, rel2id, max_length = 512, is_training = False, suffix='_train')
else:
    print("train annotated file not exist")
if os.path.exists(dev_file_name):
    init(dev_file_name, rel2id, max_length = 512, is_training = False, suffix='_dev')
else:
    print("dev file not exist")
if os.path.exists(test_file_name):
    init(test_file_name, rel2id, max_length = 512, is_training = False, suffix='_test')
else:
    print("test file not exist")