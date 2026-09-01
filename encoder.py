import sys
import os
import torch
import torch.nn as nn

# 1. Add the NSD folder to the Python path so its internal absolute imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
nsd_path = os.path.join(current_dir, 'neural_sheaf_diffusion')

if nsd_path not in sys.path:
    sys.path.insert(0, nsd_path)

# 2. Now we can import exactly as if we were inside their repository!
from models.disc_models import (
    DiscreteDiagSheafDiffusion, 
    DiscreteBundleSheafDiffusion, 
    DiscreteGeneralSheafDiffusion
)

# from torch_geometric.nn import GCNConv
# import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn.functional as F

class GCNEncoder(nn.Module):
    def __init__(self, edge_index, args):
        super(GCNEncoder, self).__init__()
        self.edge_index = edge_index # <--- Cache it here to fix the crash
        input_dim = args['input_dim']
        hidden_dim = args['hidden_channels']
        layers = args['layers']
        
        self.dropout = args.get('dropout', 0.5)
        self.d = args.get('d', 1)
        self.final_d = self.d + (1 if args.get('add_lp') else 0) + (1 if args.get('add_hp') else 0)
        
        # Fair comparison: make the output dimension match the flattened sheaf
        out_dim = hidden_dim * self.final_d 

        self.convs = nn.ModuleList()
        if layers == 1:
            self.convs.append(GCNConv(input_dim, out_dim))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, out_dim))

    def grouped_parameters(self):
        return [], list(self.parameters())

    def forward(self, x, edge_index=None, **kwargs):
        # Use cached edge_index if none is provided
        e_idx = edge_index if edge_index is not None else self.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, e_idx)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, e_idx)
        return {"z": x, "maps": None} 

class MLPEncoder(nn.Module):
    def __init__(self, edge_index, args):
        super(MLPEncoder, self).__init__()
        self.edge_index = edge_index
        input_dim = args['input_dim']
        hidden_dim = args['hidden_channels']
        layers = args['layers']
        
        self.dropout = args.get('dropout', 0.5)
        self.d = args.get('d', 1)
        self.final_d = self.d + (1 if args.get('add_lp') else 0) + (1 if args.get('add_hp') else 0)
        out_dim = hidden_dim * self.final_d

        self.lins = nn.ModuleList()
        if layers == 1:
            self.lins.append(nn.Linear(input_dim, out_dim))
        else:
            self.lins.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(layers - 2):
                self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            self.lins.append(nn.Linear(hidden_dim, out_dim))

    def grouped_parameters(self):
        return [], list(self.parameters())

    def forward(self, x, edge_index=None, **kwargs):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return {"z": x, "maps": None}

class GraphSAGEEncoder(nn.Module):
    """
    Full-batch GraphSAGE encoder.

    Matches the output convention used by NSD/GCN/MLP encoders:
        return {"z": x, "maps": None}

    Output dimension is hidden_channels * final_d, so it is compatible with
    the current GraphSwAV projection/prototype head.
    """
    def __init__(self, edge_index, args):
        super(GraphSAGEEncoder, self).__init__()

        self.edge_index = edge_index

        input_dim = args["input_dim"]
        hidden_dim = args["hidden_channels"]
        layers = args["layers"]

        self.dropout = args.get("dropout", 0.5)
        self.d = args.get("d", 1)
        self.final_d = (
            self.d
            + (1 if args.get("add_lp") else 0)
            + (1 if args.get("add_hp") else 0)
        )

        out_dim = hidden_dim * self.final_d

        # Optional GraphSAGE-specific args.
        self.aggr = args.get("sage_aggr", "mean")
        self.sage_normalize = args.get("sage_normalize", False)
        self.sage_root_weight = args.get("sage_root_weight", True)
        self.sage_project = args.get("sage_project", False)

        # Optional normalization between layers.
        # Keep default "none" for fair comparison.
        self.norm_type = args.get("sage_norm", "none")

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if layers == 1:
            self.convs.append(
                SAGEConv(
                    input_dim,
                    out_dim,
                    aggr=self.aggr,
                    normalize=self.sage_normalize,
                    root_weight=self.sage_root_weight,
                    project=self.sage_project,
                )
            )
        else:
            self.convs.append(
                SAGEConv(
                    input_dim,
                    hidden_dim,
                    aggr=self.aggr,
                    normalize=self.sage_normalize,
                    root_weight=self.sage_root_weight,
                    project=self.sage_project,
                )
            )

            for _ in range(layers - 2):
                self.convs.append(
                    SAGEConv(
                        hidden_dim,
                        hidden_dim,
                        aggr=self.aggr,
                        normalize=self.sage_normalize,
                        root_weight=self.sage_root_weight,
                        project=self.sage_project,
                    )
                )

            self.convs.append(
                SAGEConv(
                    hidden_dim,
                    out_dim,
                    aggr=self.aggr,
                    normalize=self.sage_normalize,
                    root_weight=self.sage_root_weight,
                    project=self.sage_project,
                )
            )

        # Norms only for hidden layers, not the final output layer.
        for _ in range(max(layers - 1, 0)):
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))
            else:
                self.norms.append(nn.Identity())

    def grouped_parameters(self):
        # Same API as the other encoders.
        return [], list(self.parameters())

    def forward(self, x, edge_index=None, **kwargs):
        e_idx = edge_index if edge_index is not None else self.edge_index

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, e_idx)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, e_idx)

        return {
            "z": x,
            "maps": None,
        }

def build_nsd_encoder(args_dict, edge_index):
    """
    Initializes the Neural Sheaf Diffusion encoder based on the requested geometry.
    """
    print(f"Initializing {args_dict['model']} Encoder...")
    
    if args_dict['model'] == 'DiagSheaf':
        model_cls = DiscreteDiagSheafDiffusion
    elif args_dict['model'] == 'BundleSheaf':
        model_cls = DiscreteBundleSheafDiffusion
    elif args_dict['model'] == 'GeneralSheaf':
        model_cls = DiscreteGeneralSheafDiffusion
    elif args_dict['model'] == 'GCN':
        model_cls = GCNEncoder
    elif args_dict['model'] == 'GraphSAGE':
        model_cls = GraphSAGEEncoder
    elif args_dict['model'] == 'MLP':
        model_cls = MLPEncoder
    else:
        raise ValueError(f"Unknown sheaf model: {args_dict['model']}")

    # Initialize the model using the dictionary
    encoder = model_cls(edge_index, args_dict)
    return encoder