'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-09-18 10:21:11
@LastEditors: Zhang Xiaozhu
@LastEditTime: 2019-12-11 10:47:01
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


dpcnn_config = {
    'embed_dim': 300,
    'kernel_sizes': 3,
    'kernel_num': 128,
    'input_drop': 0.5,
    'num_layers': 2
}


class DPCNN(nn.Module):
    def __init__(self, embedding_matrix_list, opt):
        super(DPCNN, self).__init__()
        self.embed0 = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix_list[0], dtype=torch.float))
        self.input_drop = nn.Dropout(dpcnn_config['input_drop'])
        
        # 序列长度为n，卷积核大小为m，步长为s，等长卷积在序列两端补零个数p=(m-1)/2
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.region_embedding = nn.Conv1d(dpcnn_config['embed_dim'], dpcnn_config['kernel_num'], kernel_size=dpcnn_config['kernel_sizes'])
        self.zero_pad1 = nn.ConstantPad1d(padding=(1, 1), value=0)
        
        self.zero_pad2 = nn.ConstantPad1d(padding=(0, 1), value=0)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv = nn.Conv1d(dpcnn_config['kernel_num'], dpcnn_config['kernel_num'], dpcnn_config['kernel_sizes'])
        self.relu = nn.ReLU()
        
        self.dense = nn.Linear(dpcnn_config['kernel_num'], opt.polarities_dim)

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
        dpcnn
        '''
        # sen1
        sen_emb = sen_emb.permute(0, 2, 1) # (bsz, emb_dim, max_seq_len)
        sen_emb = self.region_embedding(sen_emb) # (bsz, kernel_num, max_seq_len-3+1)

        sen_emb = self.zero_pad1(sen_emb) # (bsz, kernel_num, max_seq_len)
        sen_emb = self.relu(sen_emb)
        sen_conv = self.conv(sen_emb) # (bsz, kernel_num, max_seq_len-3+1)
        sen_conv = self.zero_pad1(sen_conv) # (bsz, kernel_num, max_seq_len)
        sen_conv = self.relu(sen_conv)
        sen_conv = self.conv(sen_conv) # (bsz, kernel_num, max_seq_len-3+1)

        while sen_conv.size()[2] >= 2:
            sen_conv = self._block(sen_conv)
        sen_conv = sen_conv.squeeze(2)

        '''
        classification
        '''
        logits = self.dense(sen_conv)

        return logits

    def _block(self, x):
        x = self.zero_pad2(x)
        px = self.max_pool(x)

        x = self.zero_pad1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.zero_pad1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x
