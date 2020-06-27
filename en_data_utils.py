'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-09-16 17:10:09
@LastEditors: Please set LastEditors
@LastEditTime: 2020-06-27 22:01:28
'''


import os
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import gensim
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from torch.utils.data import Dataset


def build_tokenizer(max_seq_len, tokenizer_dat, df_names, mode):
    if os.path.exists(tokenizer_dat):
        print('loading tokenizer : ', tokenizer_dat)
        tokenizer = pickle.load(open(tokenizer_dat, 'rb'))
    else:
        text = []
        for df_name in df_names:
            df = pd.read_pickle(df_name)
            for i in range(0, len(df)):                                                   
                if mode == 'word':
                    text.extend(df.sen_cut[i])
                elif mode == 'pos':
                    text.extend(df.pos[i])
        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text) # 此处text包含重复单词
        pickle.dump(tokenizer, open(tokenizer_dat, 'wb'))
    return tokenizer


def _load_glove(pre_trained_file, word2idx=None):
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    raw_dic = dict(get_coefs(*o.split(" ")) for o in open(pre_trained_file, encoding='utf-8', errors='ignore'))
    
    word_vec = {}
    for word in raw_dic.keys():
        if word2idx is None or word in word2idx.keys():
            word_vec[word] = raw_dic[word]
    print('the ratio coverage of pre-trained embedding is : %.4f' % (len(word_vec)/float(len(word2idx))))
    return word_vec


def _load_word_vec(pre_trained_file, binary=False, word2idx=None):
    if binary == True:
        raw_dic = gensim.models.KeyedVectors.load_word2vec_format(pre_trained_file, binary=True)
    else:
        raw_dic = gensim.models.KeyedVectors.load_word2vec_format(pre_trained_file)

    word_vec = {}
    for word in raw_dic.wv.vocab.keys():
        if word2idx is None or word in word2idx.keys():
            word_vec[word] = raw_dic[word]
    print('the ratio coverage of pre-trained embedding is : %.4f' % (len(word_vec)/float(len(word2idx))))
    return word_vec


def _load_fasttext(pre_trained_file, word2idx=None):
    raw_dic = {}
    with open(pre_trained_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        n, d = map(int, fin.readline().split())
        for line in fin:
            values = line.rstrip().split(' ')
            raw_dic[values[0]] = np.asarray(values[1:], dtype='float32')

    word_vec = {}
    for word in raw_dic.keys():
        if word2idx is None or word in word2idx.keys():
            word_vec[word] = raw_dic[word]
    print('the ratio coverage of pre-trained embedding is : %.4f' % (len(word_vec)/float(len(word2idx))))
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, format, binary, pre_trained_file, initial_embedding_matrix):
    if os.path.exists(initial_embedding_matrix):
        print('loading initial_matrix : ', initial_embedding_matrix)
        embedding_matrix = pickle.load(open(initial_embedding_matrix, 'rb'))
    else:
        print('loading pre-trained embedding file : ', pre_trained_file)
        embedding_matrix = np.zeros((len(word2idx)+2, embed_dim))
        if format == 'glove':
            word_vec = _load_glove('E:/language_model/' + pre_trained_file, word2idx=word2idx)
        elif format == 'word2vec':
            word_vec = _load_word_vec('E:/language_model/' + pre_trained_file, binary=True, word2idx=word2idx)
        elif format == 'fasttext':
            word_vec = _load_fasttext('E:/language_model/' + pre_trained_file, word2idx=word2idx)
        print('build initial_embedding_matrix : ', initial_embedding_matrix)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # 0行和len(word2idx)+1行是全0向量，没有对应预训练向量的词也用全0向量表示
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(initial_embedding_matrix, 'wb'))
    return embedding_matrix


def pad_and_trunc(sequence, max_seq_len, dtype='int64', padding='post', truncating='post', value=0.):
    x = (np.ones(max_seq_len) * value).astype(dtype)
    if truncating == 'post':
        trunc = sequence[:max_seq_len]
    else:
        trunc = sequence[-max_seq_len:]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len
        self.word_freq_dic = {}
        self.sorted_words = []
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, all_word_list):
        for word in all_word_list:
            if word not in self.word_freq_dic:
                self.word_freq_dic[word] = 1
            else:
                self.word_freq_dic[word] += 1
        # 返回单词列表，由词频从高到低排列
        self.sorted_words = sorted(self.word_freq_dic, key=self.word_freq_dic.__getitem__, reverse=True)
        for word in self.sorted_words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, sen_cut, reverse=False, padding='post', truncating='post'):
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in sen_cut]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        paded_sequence = pad_and_trunc(sequence, self.max_seq_len, padding=padding, truncating=truncating)
        return paded_sequence


class MyDataset(Dataset):
    def __init__(self, df_name, tokenizer):
        df = pd.read_pickle(df_name)

        all_data = []
        for i in range(0, len(df)):
            x_data = tokenizer.text_to_sequence(df.sen_cut[i])
            y_label = df.label[i]
            data = {
                'x_data' : x_data,
                'y_label': y_label
                }
            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
        