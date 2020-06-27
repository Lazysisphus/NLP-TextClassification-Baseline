'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-09-18 10:21:11
@LastEditors: Please set LastEditors
@LastEditTime: 2020-06-27 21:57:03
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.dynamic_rnn import DynamicLSTM

lstm_config = {
    'embed_dim' : 300,
    'hidden_dim' : 64,
    'dropout' : 0.5,
    'input_drop' : 0.5,
    'num_layers' : 1
}


class LSTM(nn.Module):
    def __init__(self, embedding_matrix_list, opt):
        super(LSTM, self).__init__()
        self.embed0 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[0], dtype=torch.float), freeze=True)
        # self.embed1 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[1], dtype=torch.float), freeze=True)
        # self.embed2 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[2], dtype=torch.float), freeze=True)
        # self.embed3 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[3], dtype=torch.float), freeze=True)
        # self.embed4 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[4], dtype=torch.float), freeze=True)
        # self.embed5 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[5], dtype=torch.float), freeze=True)
        # self.elmo = ElmoLayer(opt.max_seq_len, num_output_representations=1, dropout=0.5, requires_grad=False)
        # self.input_drop = nn.Dropout(lstm_config['input_drop'])

        self.lstm = DynamicLSTM(lstm_config['embed_dim'],
                                lstm_config['hidden_dim'],
                                num_layers=lstm_config['num_layers'],
                                batch_first=True,
                                bidirectional=False
                                )
                                
        self.dense = nn.Linear(lstm_config['hidden_dim'], opt.polarities_dim)

        self.classification_layer = nn.Sequential(
                                    # nn.Dropout(lstm_config['dropout']),
                                    # nn.Linear(lstm_config['hidden_dim'], lstm_config['hidden_dim']),
                                    # nn.Tanh(),
                                    # nn.Dropout(lstm_config['dropout']),
                                    nn.Linear(lstm_config['hidden_dim'], opt.polarities_dim)
                                    )
        

    def forward(self, inputs):
        '''
        ids to emb
        '''
        sen_indicies = inputs[0]
        # sen_elmo_indicies = inputs[1]

        sen_emb = self.embed0(sen_indicies)
        # sen_emb1 = self.embed1(sen_indicies)
        # sen_emb2 = self.embed2(sen_indicies)
        # sen_emb3 = self.embed3(sen_indicies)
        # sen_emb4 = self.embed4(sen_indicies)
        # sen_elmo_emb = self.elmo(sen_elmo_indicies)[0] # (bsz, max_seq_len, 1024)

        # sen_emb = self.input_drop(torch.cat((sen_elmo_emb, sen_emb0), dim=-1))

        '''
        lstm
        '''
        sen_len = torch.sum(sen_indicies != 0, dim=-1)
        _, (sen_ht, _) = self.lstm(sen_emb, sen_len)
        # sen_ht = self.tanh(sen_ht)

        '''
        classification
        '''
        # logits = self.dense(sen_ht[0])
        logits = self.classification_layer(sen_ht[0])
        
        return logits
        