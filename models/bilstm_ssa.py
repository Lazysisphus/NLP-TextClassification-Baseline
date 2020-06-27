'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-09-18 10:21:11
@LastEditors  : Zhang Xiaozhu
@LastEditTime : 2020-01-03 20:56:02
'''


from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F

bilstm_ssa_config = {
    'embed_dim' : 300,
    'hidden_dim' : 128,
    'input_drop' : 0.5,
    'dropout' : 0.5,
    'lstm_drop' : 0,
    'num_layers' : 1,
    'n_hop' : 1,
}


class BILSTM_SSA(nn.Module):
    def __init__(self, embedding_matrix_list, opt):
        super(BILSTM_SSA, self).__init__()
        self.embed0 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[0], dtype=torch.float), freeze=True)
        self.input_drop = nn.Dropout(bilstm_ssa_config['input_drop'])
        self.lstm = DynamicLSTM(bilstm_ssa_config['embed_dim'],
                                bilstm_ssa_config['hidden_dim'],
                                num_layers=bilstm_ssa_config['num_layers'], 
                                batch_first=True, 
                                bidirectional=True,
                                dropout=bilstm_ssa_config['lstm_drop'])
                                
        self.self_attn = _SSA(input_size=bilstm_ssa_config['hidden_dim'] * 2, hidden_size=bilstm_ssa_config['hidden_dim'], n_hop=bilstm_ssa_config['n_hop'])

        self.drop = nn.Dropout(0.5)
        self.dense = nn.Linear(bilstm_ssa_config['hidden_dim'] * 2, opt.polarities_dim)

        # self.dropout = nn.Dropout(bilstm_ssa_config['dropout'])
        # self.identity = torch.eye(opt.n_hop, dtype=torch.float, requires_grad=False).to(opt.device)

    def forward(self, inputs):
        '''
        ids to emb
        '''
        sen_indicies = inputs[0]

        sen_emb0 = self.embed0(sen_indicies)
        # sen_emb1 = self.embed1(sen_indicies)
        # sen_emb2 = self.embed2(sen_indicies)
        # sen_emb3 = self.embed3(sen_indicies)
        # sen_emb4 = self.embed4(sen_indicies)

        sen_emb = self.input_drop(sen_emb0)
        
        '''
        bilstm
        '''
        sen_len = torch.sum(sen_indicies != 0, dim=-1) # 计算一个batch中各个句子的长度，(128)
        sen_H, (_, _) = self.lstm(sen_emb, sen_len) # lstm_out-(batch_size, max_seq_len, hidden_dim*2)

        '''
        structured self-attention
        '''
        sen_attW = self.self_attn(sen_H) # att_weight-(batch_size, batch_max_len, n_hop)，n_hop == 1
        sen_attW = sen_attW.transpose(1, 2) # attnT-(batch_size, n_hop, max_seq_len)
        sen_rep = torch.bmm(sen_attW, sen_H) # output-(batch_size, n_hop, hidden_dim*2)
        sen_rep = sen_rep.squeeze(1)

        '''
        classification
        '''
        logits = self.dense(sen_rep)

        return logits


class _SSA(nn.Module):
    def __init__(self, input_size, hidden_size, n_hop):
        super(_SSA, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size, bias=False) # input_layer-(hidden_dim*2, hidden_dim)
        self.output_layer = nn.Linear(hidden_size, n_hop, bias=False) # output_layer-(hidden_dim*2, n_hop)

    def forward(self, input_seq):
        """
        :param input_seq:(batch_size, max_seq_len, hidden_dim*2)
        :return:
        """
        # output-(batch_size, max_seq_len, hidden_dim) = input_seq-(batch_size, max_seq_len, hidden_dim*2) * input_layer-(hidden_dim*2, hidden_dim)
        output = torch.tanh(self.input_layer(input_seq))
        # output-(batch_size, max_seq_len, n_hop) = out_put(batch_size, max_seq_len, hidden_dim) * output_layer-(hidden_dim, n_hop)
        output = self.output_layer(output)
        output = F.softmax(output, dim=1)

        return output
