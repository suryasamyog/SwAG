import os
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, Actor, HeterophilousGraphDataset, Amazon, Coauthor
from torch_geometric.data import Data
from torch_geometric.datasets import StochasticBlockModelDataset

from torch_geometric.utils import stochastic_blockmodel_graph
from sklearn.datasets import make_classification

import math
import numpy as np
import torch
from torch_geometric.data import Data

import scipy.io
import gdown
from ogb.nodeproppred import NodePropPredDataset

def load_npz_dataset(name, data_dir):
    file_path = os.path.join(data_dir, f'{name.replace("-", "_")}.npz')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing {file_path}. Please ensure .npz files are in {data_dir}")
        
    data = np.load(file_path)
    x = torch.tensor(data['node_features'], dtype=torch.float)
    y = torch.tensor(data['node_labels'], dtype=torch.long)
    edge_index = torch.tensor(data['edges'], dtype=torch.long)
    
    train_masks = torch.BoolTensor(data['train_masks']).reshape((-1, 10))
    val_masks = torch.BoolTensor(data['val_masks']).reshape((-1, 10))
    test_masks = torch.BoolTensor(data['test_masks']).reshape((-1, 10))

    
    dataset = Data(x=x, edge_index=edge_index, y=y, 
                   train_mask=train_masks, val_mask=val_masks, test_mask=test_masks)
    
    return dataset

def get_dataset(name, root_dir="./data"):

    os.makedirs(root_dir, exist_ok=True)
    name = name.lower()

    os.makedirs(root_dir, exist_ok=True)
    name = name.lower()

    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root_dir, name=name, transform=T.NormalizeFeatures())
    
    elif name == "coauthor-cs":
        dataset = Coauthor(root=root_dir, name="CS", transform=T.NormalizeFeatures())
    elif name == "coauthor-physics":
        dataset = Coauthor(root=root_dir, name="Physics", transform=T.NormalizeFeatures())
    
    # Heterophilic benchmarks
    elif name in ["texas", "cornell", "wisconsin"]:
        dataset = WebKB(root=root_dir, name=name)
    
    elif name == "actor":
        dataset = Actor(root=root_dir)
        
    elif name in ['chameleon-filtered', 'squirrel-filtered']:
        dataset = load_npz_dataset(name, root_dir)
        return dataset
        
    elif name == "amazon-photo":
        dataset = Amazon(root=os.path.join(root_dir, "Amazon"), name="Photo", transform=T.NormalizeFeatures())
    elif name == "amazon-computer":
        dataset = Amazon(root=os.path.join(root_dir, "Amazon"), name="Computers", transform=T.NormalizeFeatures())
    
    else:
        dataset = HeterophilousGraphDataset(root=root_dir, name=name)

    return dataset[0] 
