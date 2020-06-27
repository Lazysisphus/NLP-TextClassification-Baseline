'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-09-16 17:09:58
@LastEditors: Please set LastEditors
@LastEditTime: 2020-06-08 19:36:09
'''

'''
CNN
'''
from models.cnn import CNN
from models.gcnn import GCNN
from models.dpcnn import DPCNN
from models.rcnn import RCNN
from models.rnn_cnn import RNN_CNN

'''
RNN + ATTENTION
'''
from models.lstm import LSTM
from models.bilstm import BILSTM
from models.bilstm_attention import BILSTM_ATTENTION
from models.bilstm_ssa import BILSTM_SSA