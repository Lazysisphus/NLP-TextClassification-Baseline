'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-11-26 21:10:19
@LastEditors: Zhang Xiaozhu
@LastEditTime: 2019-12-11 10:55:55
'''


from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F


rnn_cnn_config = {
    'embed_dim': 300,
    'hidden_dim': 256,
    'kernel_size': 3,
    'kernel_sizes': [2, 3, 5],
    'kernel_num': 300,
    'input_drop' : 0.5,
    'dropout': 0.5,
    'lstm_drop' : 0.0
}

class RNN_CNN(nn.Module):
    def __init__(self, embedding_matrix_list, opt):
        super(RNN_CNN, self).__init__()
        self.embed0 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[0], dtype=torch.float))
        self.input_drop = nn.Dropout(rnn_cnn_config['input_drop'])

        self.lstm = DynamicLSTM(rnn_cnn_config['embed_dim'], 
                                rnn_cnn_config['hidden_dim'], 
                                num_layers=1, 
                                batch_first=True, 
                                bidirectional=True,
                                dropout=rnn_cnn_config['lstm_drop'])
        Ks = [int(k) for k in rnn_cnn_config['kernel_sizes']] # 卷积核尺寸
        self.convs = nn.ModuleList([nn.Conv1d(rnn_cnn_config['hidden_dim'] * 2, rnn_cnn_config['kernel_num'], K) for K in Ks]) # (hidden*2_dim, kernel_num, kernel_size)
        self.dense = nn.Linear(rnn_cnn_config['kernel_num'] * rnn_cnn_config['kernel_size'], opt.polarities_dim)

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
        lstm
        '''
        sen_len = torch.sum(sen_indicies != 0, dim=-1) # 计算一个batch中各个句子的长度，(128)
        sen_H, (_, _) = self.lstm(sen_emb, sen_len) # lstm_out-(batch_size, max_seq_len, hidden_dim*2)

        '''
        cnn
        '''
        sen_cnn = [torch.tanh(conv(sen_H.transpose(1, 2))) for conv in self.convs]
        sen_pool = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in sen_cnn]
        sen_pool = [i.view(i.size(0), -1) for i in sen_pool]
        sen_pool = torch.cat(sen_pool, 1)

        '''
        classification
        '''
        logits = self.dense(sen_pool)

        return logits
