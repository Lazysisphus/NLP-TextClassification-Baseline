'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-09-18 10:21:11
@LastEditors  : Zhang Xiaozhu
@LastEditTime : 2020-01-06 09:36:22
'''


from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F


bilstm_attention_config = {
    'embed_dim' : 300,
    'hidden_dim' : 64,
    'input_drop' : 0.5,
    'num_layers' : 1
}


class BILSTM_ATTENTION(nn.Module):
    def __init__(self, embedding_matrix_list, opt):
        super(BILSTM_ATTENTION, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[0], dtype=torch.float), freeze=True)
        # self.input_drop = nn.Dropout(bilstm_attention_config['input_drop'])
        self.bilstm = DynamicLSTM(bilstm_attention_config['embed_dim'],
                                bilstm_attention_config['hidden_dim'],
                                num_layers=bilstm_attention_config['num_layers'],
                                batch_first=True,
                                bidirectional=True
                                )
        self.weight1 = nn.Parameter(torch.Tensor(bilstm_attention_config['hidden_dim'] * 2, bilstm_attention_config['hidden_dim'] * 2))
        self.weight2 = nn.Parameter(torch.Tensor(bilstm_attention_config['hidden_dim'] * 2, 1))

        nn.init.uniform_(self.weight1, -0.1, 0.1)
        nn.init.uniform_(self.weight2, -0.1, 0.1)
                      
        self.dense = nn.Linear(bilstm_attention_config['hidden_dim'] * 2, opt.polarities_dim)

    def forward(self, inputs):
        '''
        ids to emb
        '''
        sen_indicies = inputs[0]
        sen_emb = self.embed(sen_indicies)
        # sen_emb = self.input_drop(sen_emb0)

        '''
        bilstm
        '''
        sen_len = torch.sum(sen_indicies != 0, dim=-1)
        sen_M, (sen_ht, _) = self.bilstm(sen_emb, sen_len) # sen_M - (bsz, seq_len, hidden_dim * 2)

        '''
        attention
        '''
        score = torch.tanh(torch.matmul(sen_M, self.weight1))
        attention_weights = F.softmax(torch.matmul(score, self.weight2), dim=1) # attention_weights - (bsz, seq_len, 1)

        sen_out = sen_M * attention_weights
        sen_rep = torch.sum(sen_out, dim=1)

        '''
        classification
        '''
        logits = self.dense(sen_rep)
        
        return logits
