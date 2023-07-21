"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
import codecs

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, target_client,clients,batch_size, opt, evaluation):
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation

        # ************* target data *****************
        target_train_data = "../data/" + filename + "/train/" + target_client + ".txt"
        target_test_data = "../data/" + filename + "/test/" + target_client + ".txt"
        self.target_ma_set, self.target_ma_list, self.target_train_data, self.target_test_data, self.target_user, self.target_item = self.read_data(
            target_train_data, target_test_data)
        opt["target_user_num"] = len(self.target_user)
        opt["target_item_num"] = len(self.target_item)

        # ************* source data *****************
        if self.eval ==-1 or self.eval==1:
            clients=clients.split(",")
            self.source_num=0
            self.source_data={}
            self.data={}
            for i in clients:
                if i != target_client:
                    one_source = {}
                    self.source_num+=1
                    source_train_data = "../data/" + filename + "/train/"+i+".txt"
                    source_test_data = "../data/" + filename + "/test/"+i+".txt"
                    source_ma_set, source_ma_list, source_train_data, source_test_data, source_user, source_item = self.read_data(source_train_data, source_test_data)
                    one_source['source_ma_set']=source_ma_set
                    one_source['source_ma_list'] = source_ma_list
                    one_source['source_train_data'] = source_train_data
                    one_source['source_test_data'] = source_test_data
                    one_source['source_user'] = source_user
                    one_source['source_item']=source_item
                    self.source_data[self.source_num]=one_source
                    opt["source"+str(self.source_num)+"_user_num"] = len(source_user)
                    opt["source"+str(self.source_num)+"_item_num"] = len(source_item)
                    assert opt["source"+str(self.source_num)+"_user_num"] == opt["target_user_num"]

                    opt["rate"] = self.rate()

                    if evaluation == -1:
                        data = self.preprocess()
                    else :
                        data = self.preprocess_for_predict()
                    # shuffle for training
                    if evaluation == -1:
                        indices = list(range(len(data)))
                        random.shuffle(indices)
                        data = [data[i] for i in indices]
                        if batch_size > len(data):
                            batch_size = len(data)
                            self.batch_size = batch_size
                        if len(data)%batch_size != 0:
                            data += data[:batch_size]
                        data = data[: (len(data)//batch_size) * batch_size]
                    self.num_examples = len(data)

                    # chunk into batches
                    data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
                    self.data[self.source_num]=data
        elif self.eval==2:
            data = self.preprocess_for_predict()
            self.num_examples = len(data)
            # chunk into batches
            data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
            self.data = data

    def read_data(self, train_file, test_file):
        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
            user = {}
            item = {}
            ma = {}
            ma_list = {}
            for line in infile:
                line=line.strip().split("\t")
                line[0] = int(line[0])
                line[1] = int(line[1])
                if user.get(line[0], "zxczxc") is "zxczxc":
                    user[line[0]] = len(user)
                if item.get(line[1], "zxczxc") is "zxczxc":
                    item[line[1]] = len(item)
                line[0] = user[line[0]]
                line[1] = item[line[1]]
                train_data.append([line[0],line[1]])
                if line[0] not in ma:
                    ma[line[0]] = set()
                    ma_list[line[0]] = []
                ma[line[0]].add(line[1])
                ma_list[line[0]].append(line[1])
        with codecs.open(test_file,"r",encoding="utf-8") as infile:
            test_data=[]
            for line in infile:
                line=line.strip().split("\t")
                line[0] = int(line[0])
                line[1] = int(line[1])
                if user.get(line[0], "zxczxc") is "zxczxc":
                    continue
                if item.get(line[1], "zxczxc") is "zxczxc":
                    continue
                line[0] = user[line[0]]
                line[1] = item[line[1]]

                ret = [line[1]]
                for i in range(999):
                    while True:
                        rand = random.randint(0, len(item)-1)
                        if rand in ma[line[0]]:
                            continue
                        ret.append(rand)
                        break
                test_data.append([line[0],ret])

        return ma, ma_list, train_data, test_data, user, item

    def rate(self):
        ret = []
        source_ma_set=self.source_data[self.source_num]['source_ma_set']
        for i in range(len(source_ma_set)):
            ret = len(source_ma_set[i]) / (len(source_ma_set[i]) + len(self.target_ma_set[i]))
        return ret

    def preprocess_for_predict(self):
        processed=[]
        if self.eval == 1:
            for d in self.source_data[self.source_num]['source_test_data']:
                processed.append([d[0],d[1]]) # user, item_list(pos in the first node)
        else :
            for d in self.target_test_data:
                processed.append([d[0],d[1]]) # user, item_list(pos in the first node)
        return processed
    def preprocess(self):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in self.source_data[self.source_num]['source_train_data']:
            d = [d[1], d[0]]
            processed.append(d + [-1])
        for d in self.target_train_data:
            processed.append([-1] + d)
        return processed

    def find_pos(self,ma_list, user):
        rand = random.randint(0, 1000000)
        rand %= len(ma_list[user])
        return ma_list[user][rand]

    def find_neg(self, ma_set, user, type):
        n = 5
        while n:
            n -= 1
            rand = random.randint(0, self.opt[type] - 1)
            if rand not in ma_set[user]:
                return rand
        return rand

    def __len__(self,num):
        return len(self.data[num])

    def all_len(self):
        len_list={}
        for i in range(self.source_num):
            len_list[i+1]=self.__len__(i+1)
        return len_list

    def t_len(self):
        return len(self.data)

    def __getitem__(self, key,num):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data[num]):
            raise IndexError
        batch = self.data[num][key]
        batch_size = len(batch)
        if self.eval!=-1 :
            batch = list(zip(*batch))
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]))

        else :
            source_neg_tmp = []
            target_neg_tmp = []
            source_pos_tmp = []
            target_pos_tmp = []
            user = []
            for b in batch:
                if b[0] == -1:
                    source_ma_list=self.source_data[num]['source_ma_list']
                    source_pos_tmp.append(self.find_pos(source_ma_list, b[1]))
                    target_pos_tmp.append(b[2])
                else:
                    source_pos_tmp.append(b[0])
                    target_pos_tmp.append(self.find_pos(self.target_ma_list, b[1]))
                source_ma_set = self.source_data[num]['source_ma_set']
                source_neg_tmp.append(self.find_neg(source_ma_set, b[1], "source"+str(num)+"_item_num"))
                target_neg_tmp.append(self.find_neg(self.target_ma_set, b[1], "target_item_num"))
                user.append(b[1])
            return (torch.LongTensor(user), torch.LongTensor(source_pos_tmp), torch.LongTensor(source_neg_tmp), torch.LongTensor(target_pos_tmp), torch.LongTensor(target_neg_tmp))

    def t__getitem__(self,key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        # if self.eval != -1:
        batch = list(zip(*batch))
        return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]))

    def __iter__(self):
        if self.eval==2:
            for t in range(self.t_len()):
                yield self.t__getitem__(t)
        else:
            for j in range(self.source_num):
                num=j+1
                for i in range(self.__len__(num)):
                    yield self.__getitem__(i,num)