from torch.utils import data
import os
import random
import numpy as np
from gensim.models import KeyedVectors


class MR_Dataset(data.Dataset):
    def __init__(self,state="train",k=0,embedding_type="word2vec"):

        self.path = os.path.abspath('..')
        if "data" not in self.path:
            self.path+="/data"
        # 导入数据及
        pos_samples = open(self.path+"/rt-polaritydata/rt-polarity.pos",errors="ignore").readlines()
        neg_samples = open(self.path+"/rt-polaritydata/rt-polarity.neg",errors="ignore").readlines()
        datas = pos_samples+neg_samples
        #datas = [nltk.word_tokenize(data) for data in datas]
        datas = [data.split() for data in datas]
        max_sample_length = max([len(sample) for sample in datas]) # 求句子最大长度，将所有句子pad成一样的长度
        labels = [1]*len(pos_samples)+[0]*len(neg_samples)
        word2id = {"<pad>":0} # 生成word2id
        for i,data in enumerate(datas):
            for j,word in enumerate(data):
                if word2id.get(word)==None:
                    word2id[word] = len(word2id)
                datas[i][j] = word2id[word]
            datas[i] = datas[i]+[0]*(max_sample_length-len(datas[i]))
        self.n_vocab = len(word2id)
        self.word2id = word2id
        self.get_word2vec()
        c = list(zip(datas,labels)) # 打乱训练集
        random.seed(1)
        random.shuffle(c)
        datas[:],labels[:] = zip(*c)
        if state=="train": # 生成训练集
            self.datas = datas[:int(k * len(datas) / 10)] + datas[int((k + 1) * len(datas) / 10):]
            self.labels = labels[:int(k * len(datas) / 10)] + labels[int((k + 1) * len(labels) / 10):]
            self.datas = np.array(self.datas[0:int(0.9*len(self.datas))])
            self.labels = np.array(self.labels[0:int(0.9*len(self.labels))])
        elif state == "valid": # 生成验证集
            self.datas = datas[:int(k * len(datas) / 10)] + datas[int((k + 1) * len(datas) / 10):]
            self.labels = labels[:int(k * len(datas) / 10)] + labels[int((k + 1) * len(labels) / 10):]
            self.datas = np.array(self.datas[int(0.9 * len(self.datas)):])
            self.labels = np.array(self.labels[int(0.9 * len(self.labels)):])
        elif state == "test": # 生成测试集
            self.datas = np.array(datas[int(k * len(datas) / 10):int((k + 1) * len(datas) / 10)])
            self.labels = np.array(labels[int(k * len(datas) / 10):int((k + 1) * len(datas) / 10)])

    def __getitem__(self, index):
        return self.datas[index], self.labels[index]

    def __len__(self):
        return len(self.datas)


    def get_word2vec(self):

    # 生成word2vec词向量
    # :return: 根据词表生成的词向量
        if  not os.path.exists(self.path+"/word2vec_embedding_mr.npy"): # 如果已经保存了词向量，就直接读取
            print ("Reading word2vec Embedding...")
            wvmodel = KeyedVectors.load_word2vec_format(os.path.abspath('../model/GoogleNews-vectors-negative300.bin.gz'),binary=True)
            tmp = []
            for word, index in self.word2id.items():
                try:
                    tmp.append(wvmodel.get_vector(word))
                except:
                    pass
            mean = np.mean(np.array(tmp))
            std = np.std(np.array(tmp))
            print (mean,std)
            vocab_size = self.n_vocab
            embed_size = 300
            embedding_weights = np.random.normal(mean,std,[vocab_size,embed_size]) # 正太分布初始化方法
            for word, index in self.word2id.items():
                try:
                    embedding_weights[index, :] = wvmodel.get_vector(word)
                except:
                    pass
            np.save(self.path+"/word2vec_embedding_mr.npy", embedding_weights) # 保存生成的词向量
        else:
            embedding_weights = np.load(self.path+"/word2vec_embedding_mr.npy") # 载入生成的词向量
            # print("载入生成的词向量")
        self.weight = embedding_weights

