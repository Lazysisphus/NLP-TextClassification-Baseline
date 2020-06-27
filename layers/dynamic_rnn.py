'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-09-16 17:09:56
@LastEditors: Please set LastEditors
@LastEditTime: 2020-06-08 19:21:37
'''


import torch
import torch.nn as nn
import numpy as np

class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                bidirectional=False, only_use_last_hidden_state=False, rnn_type = 'LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type
        
        if self.rnn_type == 'LSTM': 
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)  
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        

    def forward(self, x, x_len):
        """
        sequence -> sort -> pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors（128, 80, 300） (batch_size, max_seq_len, emb_dim)
        :param x_len：一个batch中各个句子的长度，(128)
        :return:
        """

        """sort"""
        # 先对x_len各个元素取负，再排升序，其实就是取降序...
        # torch返回：1.排序后的tensor；2.排序后tensor中各个元素之前的idx
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long() # 从0开始序列，两次取址，得到原始序列
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx] # 将x中各个句子按照长度的降序排序

        """pack"""
        # https://zhuanlan.zhihu.com/p/34418001讲得很好
        # 避免补齐符号参与RNN的运算，影响最后的表示，所以将每个句子压缩
        # 返回PackedSequence类型
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        
        # process using the selected RNN
        if self.rnn_type == 'LSTM': 
            # out_pack：(seq_len, batch, num_directions * hidden_size)，最后一层LSTM各个时间步的输出
            # 如果inputs是PackedSequence类型，那么out_pack也是
            # ht(1, 128, 300)，ct(1, 128, 300)，(num_layers * num_directions, batch, hidden_size)
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else: 
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        
        """unsort: h"""
        # 交换给定两个维度的位置
        ht = torch.transpose(ht, 0, 1)[x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        # 然后再交换...?
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            # out[0]：tensor，每个序列各个时间步的h
            # out[1]：tensor，每个序列的长度，降序
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0] 
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type =='LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)