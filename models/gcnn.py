'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-11-26 21:10:19
@LastEditors: Zhang Xiaozhu
@LastEditTime: 2019-12-11 10:48:06
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


gcnn_config = {
    'emb_dim': 300,
    'kernel_size': 3,
    'kernel_sizes': [2, 3, 5],
    'kernel_num': 128,
    'mlp_dim': 128,
    'dropout': 0.5,
    'input_drop' : 0.5
}


class GCNN(nn.Module):
    def __init__(self, embedding_matrix_list, opt):
        super(GCNN, self).__init__()
        self.embed0 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[0], dtype=torch.float), freeze=True)
        self.gate = GLU() # 选择一种门控机制
        self.dense = nn.Linear(gcnn_config['kernel_size'] * gcnn_config['kernel_num'], opt.polarities_dim)
        
        self.input_drop = nn.Dropout(gcnn_config['input_drop'])
        self.dropout =nn.Dropout(gcnn_config['dropout'])
        
    def forward(self, inputs):
        '''
        ids to emb
        '''
        sen_indicies = inputs[0]
        sen_emb = self.embed0(sen_indicies)
        sen_feature = self.input_drop(sen_emb)

        '''
        produce feature maps
        '''
        sen_conv_list = self.gate(sen_feature)

        sen_rep = [i.view(i.size(0), -1) for i in sen_conv_list]
        sen_rep = torch.cat(sen_rep, 1)
        
        '''
        classification
        '''
        logits = self.dense(sen_rep)

        return logits


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
        # (emb_dim, kernel_num, kernel_size)
        self.convs = nn.ModuleList([nn.Conv1d(gcnn_config['emb_dim'], gcnn_config['kernel_num'], K) for K in gcnn_config['kernel_sizes']])
        self.dropout =nn.Dropout(gcnn_config['dropout'])

    def forward(self, feature):
        conv_list = []
        for conv in self.convs:
            conv_S = torch.sigmoid(conv(feature.transpose(1, 2)))
            conv_L = conv(feature.transpose(1, 2))
            conv_MUL = conv_S * conv_L
            conv_ADD = torch.add(conv_L, conv_MUL)
            conv_ADD = self.dropout(conv_ADD)
            conv_ADD = F.max_pool1d(conv_ADD, conv_ADD.size(2)).squeeze(2)
            conv_list.append(conv_ADD)
        return conv_list

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        # (emb_dim, kernel_num, kernel_size)
        self.convs = nn.ModuleList([nn.Conv1d(gcnn_config['emb_dim'], gcnn_config['kernel_num'], K) for K in gcnn_config['kernel_sizes']])
        self.dropout =nn.Dropout(gcnn_config['dropout'])

    def forward(self, feature):
        conv_list = []
        for conv in self.convs:
            conv_S = torch.sigmoid(conv(feature.transpose(1, 2)))
            conv_R = F.relu(conv(feature.transpose(1, 2)))
            conv_MUL = conv_S * conv_R
            conv_ADD = torch.add(conv_R, conv_MUL)
            conv_ADD = self.dropout(conv_ADD)
            conv_ADD = F.max_pool1d(conv_ADD, conv_ADD.size(2)).squeeze(2)
            conv_list.append(conv_ADD)
        return conv_list
