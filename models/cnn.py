'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-09-18 10:21:11
@LastEditors  : Zhang Xiaozhu
@LastEditTime : 2020-01-03 10:35:22
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


cnn_config = {
    'emb_dim': 300,
    'kernel_size': 3,
    'kernel_sizes': [2, 3, 5],
    'kernel_num': 64,
    'mlp_dim': 128,
    'dropout': 0.5,
    'input_drop' : 0.5
}


class CNN(nn.Module):
    def __init__(self, embedding_matrix_list, opt):
        super(CNN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[0], dtype=torch.float), freeze=True)
        # self.input_drop = nn.Dropout(cnn_config['input_drop'])
        self.convs = nn.ModuleList([nn.Conv1d(cnn_config['emb_dim'], cnn_config['kernel_num'], K) for K in cnn_config['kernel_sizes']])
        self.dropout =nn.Dropout(cnn_config['dropout'])

        self.dense = nn.Linear(cnn_config['kernel_size'] * cnn_config['kernel_num'], opt.polarities_dim)
                
    def forward(self, inputs):
        '''
        ids to emb
        '''
        sen_indicies = inputs[0]
        sen_feature = self.embed(sen_indicies)
        # sen_feature = self.input_drop(sen_emb)

        '''
        produce feature maps
        '''
        conv_list = []
        for conv in self.convs:
            conv_L = conv(sen_feature.transpose(1, 2))
            conv_L = self.dropout(conv_L)
            conv_L = F.max_pool1d(conv_L, conv_L.size(2)).squeeze(2)
            conv_list.append(conv_L)

        sen_out = [i.view(i.size(0), -1) for i in conv_list]
        sen_out = torch.cat(sen_out, dim=1)

        '''
        classification
        '''
        logits = self.dense(sen_out)
        
        return logits
        