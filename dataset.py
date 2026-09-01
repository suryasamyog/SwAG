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

dataset_drive_url = {
    'pokec': '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y'
}

splits_drive_url = {
    'pokec': '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_'
}


def generate_hsbm(root_dir="./data", pattern="mixed", num_nodes=5000, num_features=2000, num_classes=3):
    """
    Generates an HSBM based on complex, class-specific connection probabilities.
    (Ref: arXiv:2401.09125)
    """
    print(f"[*] Generating HSBM (Pattern: {pattern}, N={num_nodes}, C={num_classes})...")
    
    # 1. Distribute nodes evenly across classes
    block_sizes = [num_nodes // num_classes] * num_classes
    block_sizes[-1] += num_nodes - sum(block_sizes) # Catch remainders
    
    # 2. Define the Heterophily Matrices (Must be symmetric for undirected graphs)
    if pattern == "bipartite":
        # Class 0 and 1 form a bipartite heterophilic core; Class 2 is isolated homophily
        edge_probs = torch.tensor([
            [0.001, 0.025, 0.001],
            [0.025, 0.001, 0.001],
            [0.001, 0.001, 0.025]
        ])
    elif pattern == "mixed":
        # A chaotic mix: 0 is homophilic, 1 is heterophilic, 2 is weakly connected
        edge_probs = torch.tensor([
            [0.030, 0.005, 0.010],
            [0.005, 0.002, 0.020],
            [0.010, 0.020, 0.005]
        ])
    else:
        # Fallback to pure Homophily (for sanity checking)
        edge_probs = torch.eye(num_classes) * 0.03 + 0.002

    # 3. Generate Topology natively via PyG
    edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs, directed=False)
    
    # 4. Generate Node Features (Distinct Gaussian clusters per class)
    x = torch.zeros((num_nodes, num_features), dtype=torch.float32)
    y = []
    
    # Define a distinct feature center for each class
    centers = torch.randn(num_classes, num_features) * 2.0 
    
    node_idx = 0
    for c, size in enumerate(block_sizes):
        y.extend([c] * size)
        # Features = Class Center + Gaussian Noise
        x[node_idx:node_idx+size] = centers[c] + torch.randn(size, num_features) * 1.0
        node_idx += size
        
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=y_tensor)
    return data.coalesce()

def generate_csbm(root_dir="./data", phi=1.0, num_nodes=5000, num_features=2000, avg_degree=5, epsilon=3.25):
    """
    Generates a cSBM exactly matching the PolyGCL implementation.
    """
    print(f"[*] Generating PolyGCL-style cSBM (phi={phi}, N={num_nodes}, F={num_features})... This might take a moment.")
    
    # 1. Calculate Lambda and mu based on phi (theta)
    gamma = num_nodes / num_features
    
    # PolyGCL clamps theta between -1 and 1
    theta = max(-1.0, min(1.0, float(phi)))
    
    Lambda = np.sqrt(1 + epsilon) * math.sin(theta * math.pi / 2)
    mu = np.sqrt(gamma * (1 + epsilon)) * math.cos(theta * math.pi / 2)

    # 2. Base parameters
    n = num_nodes
    p = num_features
    d = avg_degree

    c_in = d + np.sqrt(d) * Lambda
    c_out = d - np.sqrt(d) * Lambda
    
    # 3. Assign Labels (Half +1, Half -1)
    y_math = np.ones(n)
    y_math[int(n / 2) + 1:] = -1  # Replicating PolyGCL's exact indexing
    y_math = np.asarray(y_math, dtype=int)

    # 4. Generate Edges (O(N^2) loop, matching PolyGCL)
    edge_index = [[], []]
    for i in range(n - 1):
        for j in range(i + 1, n):
            if y_math[i] * y_math[j] > 0:
                Flip = np.random.binomial(1, c_in / n)
            else:
                Flip = np.random.binomial(1, c_out / n)
                
            if Flip > 0.5:
                edge_index[0].extend([i, j])
                edge_index[1].extend([j, i])

    # 5. Generate Node Features
    x = np.zeros([n, p])
    u = np.random.normal(0, 1 / np.sqrt(p), [1, p])
    for i in range(n):
        Z = np.random.normal(0, 1, [1, p])
        x[i] = np.sqrt(mu / n) * y_math[i] * u + Z / np.sqrt(p)
        
    # Map labels from {-1, 1} to {0, 1} for PyTorch Geometric
    y_pyg = (y_math + 1) // 2

    data = Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                y=torch.tensor(y_pyg, dtype=torch.long))

    # Clean up duplicate edges and sort
    data = data.coalesce()
    
    return data
    
def load_npz_dataset(name, data_dir):
    """Loads Chameleon or Squirrel from .npz files with standard splits."""
    file_path = os.path.join(data_dir, f'{name.replace("-", "_")}.npz')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing {file_path}. Please ensure .npz files are in {data_dir}")
        
    data = np.load(file_path)
    x = torch.tensor(data['node_features'], dtype=torch.float)
    y = torch.tensor(data['node_labels'], dtype=torch.long)
    edge_index = torch.tensor(data['edges'], dtype=torch.long)
    
    # Load the 10 fixed splits from Geom-GCN
    train_masks = torch.BoolTensor(data['train_masks']).reshape((-1, 10))
    val_masks = torch.BoolTensor(data['val_masks']).reshape((-1, 10))
    test_masks = torch.BoolTensor(data['test_masks']).reshape((-1, 10))

    
    dataset = Data(x=x, edge_index=edge_index, y=y, 
                   train_mask=train_masks, val_mask=val_masks, test_mask=test_masks)
    
    return dataset

# def load_pokec_mat():
#     # https://github.com/ChenJY-Count/PolyGCL/blob/master/non-homophilous/dataset.py
#     """ requires pokec.mat
#     """
#     if not path.exists(f'{DATAPATH}pokec.mat'):
#         gdown.download(id=dataset_drive_url['pokec'], \
#             output=f'{DATAPATH}pokec.mat', quiet=False)

#     fulldata = scipy.io.loadmat(f'{DATAPATH}pokec.mat')

#     dataset = NCDataset('pokec')
#     edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
#     node_feat = torch.tensor(fulldata['node_feat']).float()
#     num_nodes = int(fulldata['num_nodes'])
#     dataset.graph = {'edge_index': edge_index,
#                      'edge_feat': None,
#                      'node_feat': node_feat,
#                      'num_nodes': num_nodes}

#     label = fulldata['label'].flatten()
#     dataset.label = torch.tensor(label, dtype=torch.long)

#     return dataset


# def load_arxiv_year_dataset(nclass=5):
#     # https://github.com/ChenJY-Count/PolyGCL/blob/master/non-homophilous/dataset.py
#     # from polygcl
#     filename = 'arxiv-year'
#     dataset = NCDataset(filename)
#     ogb_dataset = NodePropPredDataset(name='ogbn-arxiv')
#     dataset.graph = ogb_dataset.graph
#     dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
#     dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

#     label = even_quantile_labels(
#         dataset.graph['node_year'].flatten(), nclass, verbose=False)
#     dataset.label = torch.as_tensor(label).reshape(-1, 1)
#     return dataset

def load_pokec(root_dir="./data"):
    mat_path = os.path.join(root_dir, "pokec.mat")

    if not os.path.exists(mat_path):
        gdown.download(id=dataset_drive_url['pokec'],
                       output=mat_path, quiet=False)

    fulldata = scipy.io.loadmat(mat_path)

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    x = torch.tensor(fulldata['node_feat']).float()
    y = torch.tensor(fulldata['label'].flatten(), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    # ---- FIXED SPLIT ONLY ----
    splits_path = os.path.join(root_dir, "pokec-splits.npy")

    if not os.path.exists(splits_path):
        gdown.download(id=splits_drive_url['pokec'],
                       output=splits_path, quiet=False)

    splits_lst = np.load(splits_path, allow_pickle=True)
    split = splits_lst[0]

    data.train_mask = index_to_mask(torch.tensor(split['train']), data.num_nodes)
    data.val_mask = index_to_mask(torch.tensor(split['valid']), data.num_nodes)
    data.test_mask = index_to_mask(torch.tensor(split['test']), data.num_nodes)

    return data

def load_arxiv_year(root_dir="./data", nclass=5):
    dataset = NodePropPredDataset(name='ogbn-arxiv', root=root_dir)
    graph = dataset.graph

    edge_index = torch.as_tensor(graph['edge_index'])
    x = torch.as_tensor(graph['node_feat'])

    # quantile labels (PolyGCL)
    y = even_quantile_labels(graph['node_year'].flatten(), nclass)
    y = torch.tensor(y, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    # ---- OGB SPLIT ONLY ----
    split_idx = dataset.get_idx_split()

    data.train_mask = index_to_mask(split_idx['train'], data.num_nodes)
    data.val_mask = index_to_mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index_to_mask(split_idx['test'], data.num_nodes)

    return data

def get_dataset(name, root_dir="./data"):
    """
    Unified dataset loader for both homophilic and heterophilic graphs.
    """
    os.makedirs(root_dir, exist_ok=True)
    name = name.lower()

    os.makedirs(root_dir, exist_ok=True)
    name = name.lower()

    # ---> UPDATED CSBM HOOK <---
    if name.startswith("csbm_"):
        # Extract phi (e.g., from "csbm_0.25")
        phi_val = float(name.split("_")[1])
        
        # Pass PolyGCL's exact shell script parameters
        dataset = generate_csbm(
            root_dir=root_dir, 
            phi=phi_val, 
            num_nodes=5000, 
            num_features=2000, 
            avg_degree=5, 
            epsilon=3.25
        )
        return dataset
    
    # ---> UPDATED CSBM HOOK <---
    if name.startswith("csbm_"):
        phi_val = float(name.split("_")[1])
        return generate_csbm(root_dir=root_dir, phi=phi_val)

    # ---> NEW HSBM HOOK <---
    if name.startswith("hsbm_"):
        pattern_val = name.split("_")[1] # e.g., "hsbm_mixed" or "hsbm_bipartite"
        return generate_hsbm(root_dir=root_dir, pattern=pattern_val)

    # Homophilic benchmarks
    if name in ['cora', 'citeseer', 'pubmed']:
        # dataset = Planetoid(root=os.path.join(root_dir, name), name=name, transform=T.NormalizeFeatures())
        dataset = Planetoid(root=root_dir, name=name, transform=T.NormalizeFeatures())
    
    elif name == "coauthor-cs":
        dataset = Coauthor(root=root_dir, name="CS", transform=T.NormalizeFeatures())
    elif name == "coauthor-physics":
        dataset = Coauthor(root=root_dir, name="Physics", transform=T.NormalizeFeatures())
    
    # Heterophilic benchmarks
    elif name in ["texas", "cornell", "wisconsin"]:
        # dataset = WebKB(root=root_dir, name=name, transform=T.NormalizeFeatures())
        dataset = WebKB(root=root_dir, name=name)
    
    elif name == "actor":
        # dataset = Actor(root=root_dir, transform=T.NormalizeFeatures())
        dataset = Actor(root=root_dir)
        
    elif name in ['chameleon-filtered', 'squirrel-filtered']:
        # These rely on the specific .npz files from the Specformer/Geom-GCN repos
        dataset = load_npz_dataset(name, root_dir)
        # dataset = T.NormalizeFeatures()(dataset)
        return dataset # Already a Data object
        
    elif name == "amazon-photo":
        dataset = Amazon(root=os.path.join(root_dir, "Amazon"), name="Photo", transform=T.NormalizeFeatures())
    elif name == "amazon-computer":
        dataset = Amazon(root=os.path.join(root_dir, "Amazon"), name="Computers", transform=T.NormalizeFeatures())
    
    elif name == 'pokec':
        return load_pokec(root_dir)

    elif name == 'arxiv-year':
        return load_arxiv_year(root_dir)


    else:
        # Fallback for newer PyG heterophilic datasets (e.g., Roman-empire, Minesweeper)
        # dataset = HeterophilousGraphDataset(root=root_dir, name=name, transform=T.NormalizeFeatures())
        dataset = HeterophilousGraphDataset(root=root_dir, name=name)

    return dataset[0] # Return the Data object directly
