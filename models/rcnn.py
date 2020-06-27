'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-11-26 21:10:19
@LastEditors: Zhang Xiaozhu
@LastEditTime: 2019-12-11 10:52:18
'''


from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F


rcnn_config = {
    'embed_dim': 300,
    'hidden_dim': 128,
    'input_drop' : 0.5,
    'dropout': 0.5,
    'lstm_drop' : 0.0
}

class RCNN(nn.Module):
    def __init__(self, embedding_matrix_list, opt):
        super(RCNN, self).__init__()
        self.embed0 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[0], dtype=torch.float), freeze=True)
        # self.embed1 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[1], dtype=torch.float), freeze=True)
        # self.embed2 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[2], dtype=torch.float), freeze=True)
        # self.embed3 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[3], dtype=torch.float), freeze=True)
        # self.embed4 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[4], dtype=torch.float), freeze=True)
        self.input_drop = nn.Dropout(rcnn_config['input_drop'])

        # 使用torch自己的LSTM
        self.lstm = nn.LSTM(rcnn_config['embed_dim'], 
                            rcnn_config['hidden_dim'], 
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=True,
                            dropout=rcnn_config['lstm_drop'])

        self.dense = nn.Linear((rcnn_config['hidden_dim'] * 2 + rcnn_config['embed_dim']), opt.polarities_dim)

    def forward(self, inputs):
        '''
        ids to emb
        '''
        sen_indicies = inputs[0]
        # sen_elmo_indicies = inputs[1]

        sen_emb0 = self.embed0(sen_indicies)
        # sen_emb1 = self.embed1(sen_indicies)
        # sen_emb2 = self.embed2(sen_indicies)
        # sen_emb3 = self.embed3(sen_indicies)
        # sen_emb4 = self.embed4(sen_indicies)
        # sen_elmo_emb = self.elmo(sen_elmo_indicies)[0] # (bsz, max_seq_len, 1024)

        sen_emb = self.input_drop(sen_emb0)

        '''
        bilstm
        '''
        H, (_, _) = self.lstm(sen_emb) # lstm_out-(batch_size, max_seq_len, hidden_dim*2)

        '''
        concat
        '''
        H = torch.split(H, rcnn_config['hidden_dim'], dim=-1)
        H = torch.cat((H[0], sen_emb, H[1]), dim=-1)
        H = H.transpose(1, 2)

        '''
        max-pool
        '''
        sen_rep = F.max_pool1d(H, H.size(2)).squeeze(2)

        '''
        classification
        '''
        logits = self.dense(sen_rep)
        
        return logits
