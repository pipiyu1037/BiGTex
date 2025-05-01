import numpy as np
import torch
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
import pandas as pd

def load_arxiv():
    """
    加载ogbn-arxiv数据集的DGL版本
    """
    # 加载数据集
    dataset = DglNodePropPredDataset(name='ogbn-arxiv')
    graph, labels = dataset[0]
    
    # 获取训练/验证/测试集划分
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    
    # 获取节点特征和标签
    features = graph.ndata['feat']
    labels = labels.squeeze()
    
    # 加载文本数据
    nodeidx2paperid = pd.read_csv(
        './dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')
    
    raw_text = pd.read_csv('https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz',
                           sep='\t', header=None, names=['paper id', 'title', 'abs'])
    
    nodeidx2paperid['paper id'] = nodeidx2paperid['paper id'].astype(str)
    raw_text['paper id'] = raw_text['paper id'].astype(str)
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')
    
    texts = []
    for ti, ab in zip(df['title'], df['abs']):
        t = '[sep] ' + ti + '[sep]' + ab
        texts.append(t)
    
    return graph, features, labels, train_idx, valid_idx, test_idx, texts