'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-09-18 10:21:11
@LastEditors  : Zhang Xiaozhu
@LastEditTime : 2020-01-02 21:25:46
'''


from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F


bilstm_config = {
    'embed_dim' : 300,
    'hidden_dim' : 64,
    'input_drop' : 0.5,
    'num_layers' : 1
}


class BILSTM(nn.Module):
    def __init__(self, embedding_matrix_list, opt):
        super(BILSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[0], dtype=torch.float), freeze=True)
        # self.input_drop = nn.Dropout(bilstm_config['input_drop'])
        self.bilstm = DynamicLSTM(bilstm_config['embed_dim'],
                                bilstm_config['hidden_dim'],
                                num_layers=bilstm_config['num_layers'],
                                batch_first=True,
                                bidirectional=True
                                )           
        self.dense = nn.Linear(bilstm_config['hidden_dim'] * 2, opt.polarities_dim)

    def forward(self, inputs):
        '''
        ids to emb
        '''
        sen_indicies = inputs[0]
        sen_emb = self.embed(sen_indicies)

        '''
        bilstm
        '''
        sen_len = torch.sum(sen_indicies != 0, dim=-1)
        sen_M, (sen_ht, _) = self.bilstm(sen_emb, sen_len) # sen_M - (bsz, seq_len, hidden_dim * 2)
        sen_rep = torch.cat((sen_ht[0], sen_ht[1]), dim=-1)

        '''
        classification
        '''
        logits = self.dense(sen_rep)
        
        return logits
