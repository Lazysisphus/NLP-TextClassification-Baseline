'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-09-16 14:13:06
@LastEditors: Please set LastEditors
@LastEditTime: 2020-06-27 22:00:12
'''


import os
import sys
import math
import logging
import random
import argparse

import pandas as pd
import numpy as np
from sklearn import metrics
from time import strftime, localtime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset

from transformers import BertModel
from transformers import AdamW

from en_data_utils import build_embedding_matrix
from en_data_utils import build_tokenizer, MyDataset

from models import CNN, GCNN, DPCNN, RCNN, RNN_CNN
from models import LSTM, BILSTM, BILSTM_ATTENTION, BILSTM_SSA


'''
df_file path
'''
train_df = './data/train.pkl'
dev_df = './data/dev.pkl'
test_df = './data/test.pkl'

'''
pre-trained file path
'''
# glove
PreEmbedding1 = 'glove.840B.300d.txt'
# word2vec
PreEmbedding2 = 'GoogleNews-vectors-negative300.bin'


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        tokenizer = build_tokenizer(
            max_seq_len=opt.max_seq_len,
            tokenizer_dat='tokenizer.dat',
            df_names=[train_df, dev_df, test_df],
            mode='word')
        embedding_matrix_G = build_embedding_matrix(
            word2idx=tokenizer.word2idx,
            embed_dim=300,
            format='glove',
            binary=False,
            pre_trained_file=PreEmbedding1,
            initial_embedding_matrix='{0}_{1}_matrix.dat'.format(str(300), 'Glove840'))
        embedding_matrix_list = [
            embedding_matrix_G
        ]
        self.model = opt.model_class(embedding_matrix_list, opt).to(opt.device)
        self.trainset = MyDataset(train_df, tokenizer)
        self.devset = MyDataset(dev_df, tokenizer)
        self.testset = MyDataset(test_df, tokenizer)
        
        if opt.device.type == 'cuda':
            print('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            for p in child.parameters():
                if p.requires_grad:
                    if len(p.shape) > 1:
                        self.opt.initializer(p)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, train_data_loader, val_data_loader, test_data_loader):
        '''
        multi gpu
        '''
        if self.opt.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        '''
        Loss & Optimizer
        '''
        criterion = nn.CrossEntropyLoss()
        if 'bert' in self.opt.model_name:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 
                'weight_decay': 0.01},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 
                'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.learning_rate, eps=self.opt.adamw_epsilon)
        else:
            optimizer_grouped_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(optimizer_grouped_parameters, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        
        '''
        metrics to select model
        '''
        max_val_acc = 0
        max_val_f1 = 0

        path = None
        self.model.zero_grad()
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: {}'.format(epoch))
            global_step = 0
            n_correct, n_total, loss_total = 0, 0, 0
            
            for i_batch, sample_batched in enumerate(train_data_loader):
                self.model.train()
                global_step += 1
                
                
                inputs = [torch.tensor(sample_batched[col], dtype=torch.long).to(self.opt.device) for col in self.opt.inputs_cols]
                targets = torch.tensor(sample_batched['y_label'], dtype=torch.long).to(self.opt.device)
                outputs = self.model(inputs)
                
                loss = criterion(outputs, targets)
                if self.opt.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()
                self.model.zero_grad()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    print('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
                    
                    '''
                    validate per log_steps
                    '''
                    # val_precision, val_recall, val_f1, val_acc = self._evaluate_acc_f1(val_data_loader)
                    # print('> val_precision: {:.4f}, val_recall: {:.4f}, val_f1: {:.4f}, val_acc: {:.4f}'.format(val_precision, val_recall, val_f1, val_acc))
                    # test_precision, test_recall, test_f1, test_acc = self._evaluate_acc_f1(test_data_loader)
                    # print('> test_precision: {:.4f}, test_recall: {:.4f}, test_f1: {:.4f}, test_acc: {:.4f}'.format(test_precision, test_recall, test_f1, test_acc))
                    # if val_acc > max_val_acc:
                    #     max_val_acc = val_acc
                    #     if not os.path.exists('state_dict'):
                    #         os.mkdir('state_dict')
                    #     path = 'state_dict/{0}_{1}_val_acc{2}'.format(self.opt.model_name, str(self.opt.seed), round(val_acc, 4))
                    #     torch.save(self.model.state_dict(), path)
                    #     print('>> saved: {}'.format(path))
                    # if val_f1 > max_val_f1:
                    #     max_val_f1 = val_f1
            '''
            validate per epoch
            if valset_ratio is 0, valset == testset else valset is the subset of trainset
            '''
            val_precision, val_recall, val_f1, val_acc = self._evaluate_acc_f1(val_data_loader)
            print('> val_precision: {:.4f}, val_recall: {:.4f}, val_f1: {:.4f}, val_acc: {:.4f}'.format(val_precision, val_recall, val_f1, val_acc))
            test_precision, test_recall, test_f1, test_acc = self._evaluate_acc_f1(test_data_loader)
            print('> test_precision: {:.4f}, test_recall: {:.4f}, test_f1: {:.4f}, test_acc: {:.4f}'.format(test_precision, test_recall, test_f1, test_acc))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_acc{2}'.format(self.opt.model_name, str(self.opt.seed), round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                print('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        return path

    def _evaluate_acc_f1(self, data_loader, test_flag=False):
        n_correct, n_total = 0, 0
        targets_all, outputs_all = None, None
        self.model.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(data_loader):
                inputs = [torch.tensor(sample_batched[col], dtype=torch.long).to(self.opt.device) for col in self.opt.inputs_cols]
                targets = torch.tensor(sample_batched['y_label'], dtype=torch.long).to(self.opt.device)
                outputs = self.model(inputs)
                
                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)

                if targets_all is None:
                    targets_all = targets
                    outputs_all = outputs
                else:
                    targets_all = torch.cat((targets_all, targets), dim=0)
                    outputs_all = torch.cat((outputs_all, outputs), dim=0)
        
        acc = n_correct / n_total
        precision = metrics.precision_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu())
        recall = metrics.recall_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu())
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu())

        if test_flag == True:
            prediction_list = torch.argmax(outputs_all, -1).cpu().numpy().tolist()
            df_predictions = pd.DataFrame({'label' : prediction_list})
            df_predictions.to_csv('test_res.csv', encoding='utf-8')
        return precision, recall, f1, acc

    def run(self):
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        dev_data_loader = DataLoader(dataset=self.devset, batch_size=self.opt.batch_size, shuffle=False)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path = self._train(train_data_loader, dev_data_loader, test_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        test_flag = False
        test_precision, test_recall, test_f1, test_acc = self._evaluate_acc_f1(test_data_loader, test_flag)
        print('>> test_precision: {:.4f}, test_recall: {:.4f}, test_f1: {:.4f}, test_acc: {:.4f}'.format(test_precision, test_recall, test_f1, test_acc))

        return test_precision, test_recall, test_f1, test_acc


def set_seed(num):
    os.environ['PYTHONHASHSEED'] = str(num)
    np.random.seed(num)
    random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(increase_seed):
    parser = argparse.ArgumentParser()
    '''
    optimizer_parameters
    '''
    parser.add_argument('--model_name', default='lstm', type=str)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--adamw_epsilon', default=1e-8, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    '''
    data_size_parameters
    '''
    parser.add_argument('--max_seq_len', default=32, type=int)
    parser.add_argument('--num_epoch', default=16, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--log_step', default=64, type=int)
    parser.add_argument('--polarities_dim', default=2, type=int)
    '''
    device_parameters
    '''
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--n_gpu', default=1, type=int)
    opt = parser.parse_args()

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD
    }
    model_classes = {
        'cnn' : CNN,
        'gcnn' : GCNN,
        'dpcnn' : DPCNN,
        'rcnn' : RCNN,
        'rnn_cnn' : RNN_CNN,
        'lstm' : LSTM,
        'bilstm' : BILSTM,
        'bilstm_attention' : BILSTM_ATTENTION,
        'bilstm_ssa' : BILSTM_SSA
    }
    input_cols = {
        'cnn': ['x_data'],
        'gcnn': ['x_data'],
        'dpcnn': ['x_data'],
        'rcnn': ['x_data'],
        'rnn_cnn': ['x_data'],
        'lstm': ['x_data'],
        'bilstm': ['x_data'],
        'bilstm_attention' : ['x_data'],
        'bilstm_ssa': ['x_data']
    }
    
    opt.model_class = model_classes[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.inputs_cols = input_cols[opt.model_name]
    opt.optimizer = optimizers[opt.optimizer]
    opt.seed = increase_seed

    if opt.device is None:
        opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        opt.device = torch.device(opt.device)
    
    set_seed(opt.seed)
    ins = Instructor(opt)
    test_precision, test_recall, test_f1, test_acc = ins.run()

    return test_precision, test_recall, test_f1, test_acc


if __name__ == "__main__":
    # try different seed
    # precision_list = []
    # recall_list = []
    # f1_list = []
    # acc_list = []
    
    # for increase_seed in range(0, 5):
    #     test_precision, test_recall, test_f1, test_acc = main(increase_seed)
    #     precision_list.append(test_precision)
    #     recall_list.append(test_recall)
    #     f1_list.append(test_f1)
    #     acc_list.append(test_acc)

    # avg_precision = np.average(precision_list)
    # avg_recall = np.average(recall_list)
    # avg_f1 = np.average(f1_list)
    # avg_acc = np.average(acc_list)

    # print('>> avg_precision: {:.4f}, avg_recall: {:.4f}, avg_f1: {:.4f}, avg_acc: {:.4f}'.format(avg_precision, avg_recall, avg_f1, avg_acc))

    # try one seed
    seed = 777
    test_precision, test_recall, test_f1, test_acc = main(seed)
    print('>> test_precision: {:.4f}, test_recall: {:.4f}, test_f1: {:.4f}, test_acc: {:.4f}'.format(test_precision, test_recall, test_f1, test_acc))
