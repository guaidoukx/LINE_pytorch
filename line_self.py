"""
github: https://github.com/guaidoukx/LINE_pytorch.git
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import networkx as nx
import collections
from tqdm import tqdm
import numpy as np
import random


def load_data_construct_graph(edge_file_path, direction, power):
    weightSum = 0
    sumOutDegreePower = 0
    outDegreeDict = collections.defaultdict(int)  # 就像是初始化 like initialization.
    outDegreeProbDict = collections.defaultdict(float)
    edgeWeightProbDict = collections.defaultdict(float)
    if direction:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # nLines = 0
    # with open(edge_file_path, 'r') as f:
    #     for l in f:
    #         nLines += 1
    # print("NLINES", nLines)
    with open(edge_file_path, 'r') as f:
        line = f.readline()
        while line:
            line_splited = line.split(' ')
        # for line in tqdm(f, total=nLines):
        #     line_splited = line.replace('\n', '').split(' ')
            
            # Option1 [has weight or not] ->有权重
            # weight = int(line_splited[2])
            
            # Option1 [has weight or not] ->无权重
            weight = 1
            node1, node2 = int(line_splited[0]), int(line_splited[1])
            # if node1 != node2:
            G.add_edge(node1, node2, weight=weight)
            weightSum += weight
            outDegreeDict[node1] += weight
            if not direction:
                outDegreeDict[node2] += weight
            line = f.readline()
    
    outDegrees = np.array(list(outDegreeDict.values()))
    sumOutDegreePower = np.sum(np.power(outDegrees, power))
    for node, outDegree in outDegreeDict.items():
        outDegreeProbDict[node] = np.power(outDegree, power) / sumOutDegreePower
    for edge in G.edges():
        edgeWeightProbDict[tuple([edge[0], edge[1]])] = G[edge[0]][edge[1]]['weight'] / weightSum
    return edgeWeightProbDict, outDegreeProbDict, outDegreeDict, len(G.nodes())
    

class AliasSample:
    def __init__(self, prob_dict):
        self.prob_dict = prob_dict
        self.alias_setup_with_dict()
        # self.prob_table = collections.defaultdict(float)
        # self.alias_table = {}
        # self.node_list = []
    
    def alias_setup_with_dict(self):
        num = len(self.prob_dict)
        self.table_probs = collections.defaultdict(float)
        self.table_alias = {}  # -1 用来表示，本身自己就足够了，和那些用第一根柱子（下标0）的区分开来
        
        small, large = [], []  # 记录乘以4以后的概率 大于1还是小于1 的下标
        for i, prob in self.prob_dict.items():
            self.table_probs[i] = num * prob  # 概率乘以类型数（4）
            if self.table_probs[i] < 1.0:
                small.append(i)
            else:
                large.append(i)
            # print(i)
        
        while len(small) > 0 and len(large) > 0:
            smaller = small.pop()  # 从大小中各任取一个
            larger = large.pop()
            # prob_table 用来记录本身的概率，而不是加上去的那一部分的概率
            self.table_probs[larger] = self.table_probs[smaller] + self.table_probs[larger] - 1.0
            self.table_alias[smaller] = larger  # b用来记录完成每个柱子=1的操作，补自哪个柱子
            if self.table_probs[larger] < 1.0:  # 记录大于1的那个概率，给了别人以后，剩下部分是不是还是大于1（是不是还能继续给别人）
                small.append(larger)
            else:
                large.append(larger)
        while large:  # 当输进去的概率和不为1的时候，要有之后的过程。
            self.table_probs[large.pop()] = 1.0
        while small:
            self.table_probs[small.pop()] = 1.0
        self.node_list = list(self.table_probs)
    
    def generate_instance(self):
        node_ = random.choice(self.node_list)
        if np.random.rand() <= self.table_probs[node_]:
            return node_
        else:
            return self.table_alias[node_]
        
    def sample_n(self, n):
        for i in range(n):
            yield self.generate_instance()


def get_data_of_an_edge(source_node, target_node, neg_sampler, neg_size):
    return [source_node, target_node] + list(neg_sampler.sample_n(neg_size))


def get_batch_data(batch_size, edge_sampler, neg_sampler, neg_size):
    for edge in edge_sampler.sample_n(batch_size):
        yield get_data_of_an_edge(edge[0], edge[1], neg_sampler, neg_size)


class LINE(nn.Module):
    def __init__(self, embedding_dim, embedding_size):
        super(LINE, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding_size = embedding_size
        self.node_embeddings = nn.Embedding(embedding_size, embedding_dim)
        self.context_embeddings = nn.Embedding(embedding_size, embedding_dim)
        # self.node_embeddings.weight.data = self.node_embeddings.weight.data.uniform_(0.5, 0.5) / embedding_dim
        # self.context_embeddings.weight.data = self.context_embeddings.weight.data.uniform_(0.5, 0.5) / embedding_dim
    
    def forward(self, v_i, v_j, negative_samples, device):
        v_i_emb = self.node_embeddings(v_i).to(device)
        v_j_emb = self.context_embeddings(v_j).to(device)
        # v_j_context_emb = self.context_embeddings(v_j)
        negative_samples_emb = self.context_embeddings(negative_samples).to(device)
        positive_ = torch.sigmoid(torch.sum(torch.mul(v_i_emb, v_j_emb), 1))
        negative_mul = torch.mul(v_j_emb.view(len(negative_samples), 1, self.embedding_dim), -negative_samples_emb)
        negative_ = torch.sum(F.logsigmoid(torch.sum(negative_mul, 2)), 1)
        loss = negative_+positive_
        return -torch.mean(loss)
        

if __name__ == '__main__':
    
    edge_file_path = 'data/wiki/Wiki_edgelist.txt'
    negative_power = 0.75
    learning_rate = 0.03
    batch_size = 150
    embedding_dim = 128
    epochs = 300
    neg_size = 5
    print(
        "\nnegative_power", negative_power,
        "\nlearning_rate", learning_rate,
        "\nbatch_size", batch_size,
        "\nembedding_dim", embedding_dim,
        '\nepochs', epochs
    )
    edgeWeightProbDict, outDegreeProbDict, outDegreeDict, embedding_size \
        = load_data_construct_graph(edge_file_path, True, negative_power)
    edge_sampler = AliasSample(edgeWeightProbDict)
    neg_sampler = AliasSample(outDegreeProbDict)
 
    line_ = LINE(embedding_dim, embedding_size)
    optim = torch.optim.SGD(line_.parameters(), lr=learning_rate,momentum=0.9, nesterov=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lossdata = {"it": [], "loss": []}

    n_batch = int(np.ceil(float(len(edgeWeightProbDict)) / batch_size))
    print("\nTraining on {}...\n".format(device))
    print("start train....")
    for epoch in range(epochs):
        it = 1
        for i in range(n_batch):
            batch_ = list(get_batch_data(batch_size, edge_sampler, neg_sampler, neg_size))
            batch = torch.LongTensor(batch_)
            # print("batch_", batch)

            v_i = batch[:, 0]
            v_j = batch[:, 1]
            negative_samples = batch[:, 2:]
            line_.zero_grad()
            loss = line_(v_i, v_j, negative_samples, device)
            loss.backward()
            optim.step()

            # lossdata['it'].append(it)
            lossdata['loss'].append(loss)
            # it = it +1
        print('Epoch {}, loss {}'.format(epoch, loss))
