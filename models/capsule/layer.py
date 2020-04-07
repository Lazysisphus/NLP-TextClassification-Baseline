import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def squash_v1(x, axis): # x-(256, 16, 32, 446, 1)
    s_squared_norm = (x ** 2).sum(axis, keepdim=True) # s_squared_norm-(256, 1, 32, 446, 1)，相当于||x||^2，即2-范数的平方，模长的平方
    scale = torch.sqrt(s_squared_norm)/ (0.5 + s_squared_norm) # ||x||/(0.5+||x||^2)
    return scale * x


def dynamic_routing(batch_size, b_ij, u_hat, input_capsule_num):
    num_iterations = 3

    for i in range(num_iterations):
        if True:
            leak = torch.zeros_like(b_ij).sum(dim=2, keepdim=True)
            leaky_logits = torch.cat((leak, b_ij),2)
            leaky_routing = F.softmax(leaky_logits, dim=2)
            c_ij = leaky_routing[:,:,1:,:].unsqueeze(4)
        else:
            c_ij = F.softmax(b_ij, dim=2).unsqueeze(4)
        v_j = squash_v1((c_ij * u_hat).sum(dim=1, keepdim=True), axis=3)
        if i < num_iterations - 1:
            b_ij = b_ij + (torch.cat([v_j] * input_capsule_num, dim=1) * u_hat).sum(3)

    poses = v_j.squeeze(1)
    activations = torch.sqrt((poses ** 2).sum(2))
    return poses, activations


# batch_size-256 
# b_ij-(256, 128, 758, 1): 
#   每个batch 256 个句子，低级胶囊和高级胶囊之间有 128*758 个连接，每个连接需要训练一个系数b_ij（原文中是cij）
# u_hat-(256, 128, 758, 16, 1): 
#   每个batch 256 个句子，每个句子有 128 个低级特征，这 128 个低级特征需要归纳为 758 个高级特征
#   每个低级特征是 16 维向量，每个高级特征是 1 维向量
def Adaptive_KDE_routing(batch_size, b_ij, u_hat):
    '''
    迭代求解cij、vj
    '''
    last_loss = 0.0
    while True:
        '''
        归一化bij
        '''
        if False:
            leak = torch.zeros_like(b_ij).sum(dim=2, keepdim=True)
            leaky_logits = torch.cat((leak, b_ij), 2)
            leaky_routing = F.softmax(leaky_logits, dim=2)
            c_ij = leaky_routing[:,:,1:,:].unsqueeze(4)
        else:
            # 归一化每个低级胶囊和所有高级胶囊的之间的连接权重，即对每个c_ij[a][b]，其在 dim=0 上求和值为1
            c_ij = F.softmax(b_ij, dim=2).unsqueeze(4) # c_ij-(256, 128, 758, 1, 1)，所有权重的初始值为 1/758 = 0.0013
        # 以上操作有点多余，感觉直接c_ij = F.softmax(b_ij, dim=1).unsqueeze(4)就可以，初始化的意思就是说，对每个高级胶囊，初始的时候所有低级胶囊对得到该高胶囊类别的重要性是一样的
        c_ij = c_ij/c_ij.sum(dim=1, keepdim=True) # c_ij-(256, 128, 758, 1, 1)，所有权重的值为 1/128 = 0.0078

        '''
        更新vj 
        {c_ij-(256, 128, 758, 1, 1)，u_hat-(256, 128, 758, 16, 1)}
        '''
        v_j = squash_v1((c_ij * u_hat).sum(dim=1, keepdim=True), axis=3) # v_j = squash_v1(tensor(256, 1, 758, 16, 1), axis=3) = (256, 1, 758, 16, 1)

        '''
        更新bij
        '''
        dd = 1 - ((squash_v1(u_hat, axis=3)-v_j)** 2).sum(3) # dd-(256, 128, 758, 1)
        b_ij = b_ij + dd # b_ij-(256, 128, 758, 1)，取原文中α=1

        '''
        保证c_ij、dd有相同shape
        '''
        c_ij = c_ij.view(batch_size, c_ij.size(1), c_ij.size(2)) # c_ij-(256, 128, 758)
        dd = dd.view(batch_size, dd.size(1), dd.size(2)) # dd-(256, 128, 758)

        '''
        计算kde_loss
        '''
        kde_loss = torch.mul(c_ij, dd).sum()/batch_size
        kde_loss = np.log(kde_loss.item())

        if abs(kde_loss - last_loss) < 0.05:
            break
        else:
            last_loss = kde_loss
    poses = v_j.squeeze(1) # poses-(256, 758, 16, 1)
    activations = torch.sqrt((poses ** 2).sum(2)) # activations-(256, 758, 1)
    return poses, activations


def KDE_routing(batch_size, b_ij, u_hat):
    num_iterations = 3
    for i in range(num_iterations):
        if False:
            leak = torch.zeros_like(b_ij).sum(dim=2, keepdim=True)
            leaky_logits = torch.cat((leak, b_ij),2)
            leaky_routing = F.softmax(leaky_logits, dim=2)
            c_ij = leaky_routing[:,:,1:,:].unsqueeze(4)
        else:
            c_ij = F.softmax(b_ij, dim=2).unsqueeze(4)

        c_ij = c_ij/c_ij.sum(dim=1, keepdim=True)
        v_j = squash_v1((c_ij * u_hat).sum(dim=1, keepdim=True), axis=3)

        if i < num_iterations - 1:
            dd = 1 - ((squash_v1(u_hat, axis=3)-v_j)** 2).sum(3)
            b_ij = b_ij + dd
    poses = v_j.squeeze(1)
    activations = torch.sqrt((poses ** 2).sum(2))
    return poses, activations


class FlattenCaps(nn.Module):
    def __init__(self):
        super(FlattenCaps, self).__init__()
    def forward(self, p, a): # p-(256, 16, 32, 446, 1) a_doc-(256, 32, 446, 1)
        poses = p.view(p.size(0), p.size(2) * p.size(3) * p.size(4), -1)
        activations = a.view(a.size(0), a.size(1) * a.size(2) * a.size(3), -1)
        return poses, activations


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()
        # in_channels=32, out_channels*num_capsules=32*16, kernel_size=1, stride=1
        self.capsules = nn.Conv1d(in_channels, out_channels * num_capsules, kernel_size, stride)

        torch.nn.init.xavier_uniform_(self.capsules.weight)

        self.out_channels = out_channels
        self.num_capsules = num_capsules

    def forward(self, x): # x-(256, 32, 446), feature maps
        batch_size = x.size(0)
        u = self.capsules(x).view(batch_size, self.num_capsules, self.out_channels, -1, 1) # u-(256, 16, 32, 446, 1)
        poses = squash_v1(u, axis=1) # poses-(256, 16, 32, 446, 1)
        activations = torch.sqrt((poses ** 2).sum(1)) # activations-(256, 32, 446, 1)
        return poses, activations


class FCCaps(nn.Module):
    def __init__(self, args, input_capsule_num, output_capsule_num, in_channels, out_channels):
        super(FCCaps, self).__init__()

        self.in_channels = in_channels # 16
        self.out_channels = out_channels # 16
        self.input_capsule_num = input_capsule_num # 128
        self.output_capsule_num = output_capsule_num # 3954
        # self.W1-(1, 128, 3954, 16, 16)
        self.W1 = nn.Parameter(torch.FloatTensor(1, input_capsule_num, output_capsule_num, in_channels, out_channels))
        torch.nn.init.xavier_uniform_(self.W1)

        self.is_AKDE = args.is_AKDE
        self.sigmoid = nn.Sigmoid()

    # x-(256, 128, 16) y-(256, 128)
    def forward(self, x, y, labels):
        batch_size = x.size(0)
        variable_output_capsule_num = len(labels) # 这批数据包含的标签种类数
        W1 = self.W1[:,:,labels,:,:] # W1-(1, 128, 758, 16, 16)
        W1 = W1.repeat(batch_size, 1, 1, 1, 1) # W1-(256, 128, 758, 16, 16)
        x = torch.stack([x] * variable_output_capsule_num, dim=2).unsqueeze(4) # x-(256, 128, 758, 16, 1)

        u_hat = torch.matmul(W1, x) # u_hat-(256, 128, 758, 16, 1) = W1-(256, 128, 758, 16, 16) * x-(256, 128, 758, 16, 1)

        b_ij = Variable(torch.zeros(batch_size, self.input_capsule_num, variable_output_capsule_num, 1)) # b_ij-(256, 128, 758, 1)

        if self.is_AKDE == True:
            poses, activations = Adaptive_KDE_routing(batch_size, b_ij, u_hat) # poses-(256, 758, 16, 1) activations-(256, 758, 1)
        else:
            #poses, activations = dynamic_routing(batch_size, b_ij, u_hat, self.input_capsule_num)
            poses, activations = KDE_routing(batch_size, b_ij, u_hat)
        return poses, activations
