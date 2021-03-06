import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import PrimaryCaps, FCCaps, FlattenCaps

def BCE_loss(x, target):
    return nn.BCELoss()(x.squeeze(2), target)

class CapsNet_Text(nn.Module):
    def __init__(self, args, w2v):
        super(CapsNet_Text, self).__init__()
        self.num_classes = args.num_classes
        self.embed = nn.Embedding(args.vocab_size, args.vec_size)
        self.embed.weight = nn.Parameter(torch.from_numpy(w2v))

        self.ngram_size = [2,4,8]

        self.convs_doc = nn.ModuleList([nn.Conv1d(args.sequence_length, 32, K, stride=2) for K in self.ngram_size])
        torch.nn.init.xavier_uniform_(self.convs_doc[0].weight)
        torch.nn.init.xavier_uniform_(self.convs_doc[1].weight)
        torch.nn.init.xavier_uniform_(self.convs_doc[2].weight)

        self.primary_capsules_doc = PrimaryCaps(num_capsules=args.dim_capsule, in_channels=32, out_channels=32, kernel_size=1, stride=1)

        self.flatten_capsules = FlattenCaps()

        self.W_doc = nn.Parameter(torch.FloatTensor(14272, args.num_compressed_capsule))
        torch.nn.init.xavier_uniform_(self.W_doc)

        self.fc_capsules_doc_child = FCCaps(args, 
                                            input_capsule_num=args.num_compressed_capsule, # 压缩后的胶囊数
                                            output_capsule_num=args.num_classes, # 3954，训练集出现的标签种类数
                                            in_channels=args.dim_capsule, # 16
                                            out_channels=args.dim_capsule) # 16

    def compression(self, poses, W):
        poses = torch.matmul(poses.permute(0,2,1), W).permute(0,2,1) # poses-(256, 16, 32*446)*W-(32*446, 128)，把32*446个胶囊压缩到128个
        activations = torch.sqrt((poses ** 2).sum(2)) # 计算每个胶囊的模
        return poses, activations # poses-(256, 128, 16) activations-(256, 128)

    def forward(self, data, labels): # data-(bsz, max_pad_len) labels-list[idx0, idx1, ...]
        data = self.embed(data) # data-(bsz, max_pad_len, embed_dim)-(256, 500, 300)
        nets_doc_l = []
        for i in range(len(self.ngram_size)): # 在词向量的维度上扫描做卷积
            nets = self.convs_doc[i](data) # nets-(256, 32, 150、149、147)
            nets_doc_l.append(nets)
        nets_doc = torch.cat((nets_doc_l[0], nets_doc_l[1], nets_doc_l[2]), 2) # nets_doc-(256, 32, 446)，features maps-(32, 446)，每个element都是一个特征
        poses_doc, activations_doc = self.primary_capsules_doc(nets_doc) # poses_doc-(256, 16, 32, 446, 1) activations_doc-(256, 32, 446, 1)
        poses, activations = self.flatten_capsules(poses_doc, activations_doc) # poses_doc-(256, 32*446, 16) activations_doc-(256, 32*446, 1)
        poses, activations = self.compression(poses, self.W_doc) # poses-(256, 128, 16) activations-(256, 128)
        poses, activations = self.fc_capsules_doc_child(poses, activations, labels) # poses-(256, 758, 16, 1) activations-(256, 758, 1)
        return poses, activations


class CNN_KIM(nn.Module):
    def __init__(self, args, w2v):
        super(CNN_KIM, self).__init__()
        self.embed = nn.Embedding(args.vocab_size, args.vec_size)
        self.embed.weight = nn.Parameter(torch.from_numpy(w2v))
        self.conv13 = nn.Conv2d(1, 128, (3, args.vec_size))
        self.conv14 = nn.Conv2d(1, 128, (4, args.vec_size))
        self.conv15 = nn.Conv2d(1, 128, (5, args.vec_size))

        self.fc1 = nn.Linear(3 * 128, args.num_classes)
        self.m = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def loss(self, x, target):
        return nn.BCELoss()(x, target)

    def forward(self, x):
        x = self.embed(x).unsqueeze(1)
        x1 = self.conv_and_pool(x,self.conv13)
        x2 = self.conv_and_pool(x,self.conv14)
        x3 = self.conv_and_pool(x,self.conv15)
        x = torch.cat((x1, x2, x3), 1)
        activations = self.fc1(x)
        return self.m(activations)


class XML_CNN(nn.Module):
    def __init__(self, args, w2v):
        super(XML_CNN, self).__init__()
        self.embed = nn.Embedding(args.vocab_size, args.vec_size)
        self.embed.weight = nn.Parameter(torch.from_numpy(w2v))
        self.conv13 = nn.Conv1d(500, 32, 2, stride=2)
        self.conv14 = nn.Conv1d(500, 32, 4, stride=2)
        self.conv15 = nn.Conv1d(500, 32, 8, stride=2)

        self.fc1 = nn.Linear(14272, 512)
        self.fc2 = nn.Linear(512, args.num_classes)
        self.m = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        return x

    def loss(self, x, target):
        return nn.BCELoss()(x, target)

    def forward(self, x):
        x = self.embed(x)
        batch_size = x.shape[0]

        x1 = self.conv13(x).reshape(batch_size, -1)
        x2 = self.conv14(x).reshape(batch_size, -1)
        x3 = self.conv15(x).reshape(batch_size, -1)
        x = torch.cat((x1, x2, x3), 1)
        hidden = self.fc1(x)
        activations = self.fc2(hidden)
        return self.m(activations)
