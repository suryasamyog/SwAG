

import os

# use this only if OpenBLAS crashes on your machine.
os.environ.setdefault("OPENBLAS_CORETYPE", "HASWELL")

# do not force everything to 1 thread during normal training.
default_n_threads = int(os.environ.get("NUM_THREADS", "8"))
os.environ["OPENBLAS_NUM_THREADS"] = str(default_n_threads)
os.environ["MKL_NUM_THREADS"] = str(default_n_threads)
os.environ["OMP_NUM_THREADS"] = str(default_n_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(default_n_threads)

import gc
import argparse
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import json
import random
from tqdm import tqdm
from typing import List, Optional, Tuple, Sequence, Union


import torch
import torch_sparse

import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE
import pandas as pd
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, get_laplacian

# imports from our modules
from dataset import get_dataset
from aug import generate_sheaf_views, stalk_masking, drop_node_features
from encoder import build_nsd_encoder
from swag import GraphSwAG
from model import LogReg
from analysis import (
    compute_spectral_metrics_robust, 
    compute_manifold_metrics_robust,
    run_post_training_audits
)




torch.set_num_threads(default_n_threads)
torch.set_num_interop_threads(min(4, default_n_threads))


torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



def seed_everything(seed=42, deterministic=True):
    import os
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

def hard_cleanup():
    import gc

    # Close figures if any analysis/visual code opened them.
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass

    # Finish W&B run cleanly.
    try:
        if wandb.run is not None:
            wandb.finish(quiet=True)
    except Exception as e:
        print(f"[cleanup] wandb.finish failed: {e}")

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def log_run_id(mode, dataset, run_id, project_name):
    """Appends the exact run command, W&B ID, and Project to a local tracker."""
    
    # Reconstruct the exact command
    cmd = " ".join(sys.argv)
    gpu_info = os.environ.get('HIP_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', 'CPU'))
    full_cmd = f"HIP_VISIBLE_DEVICES={gpu_info} python3 {cmd}"
    
    with open("wandb_run_tracker.txt", "a") as f:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{timestamp} | Proj: {project_name[:25]:25s} | {mode:15s} | Data: {dataset:15s} | ID: {run_id}\n")
        f.write(f"    Cmd: {full_cmd}\n")

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def random_splits(label, num_classes, percls_trn, val_lb, seed=42):
    num_nodes = label.shape[0]
    index = [i for i in range(num_nodes)]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    
    for c in range(num_classes):
        class_idx = np.where(label.cpu() == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
            
    rest_index = [i for i in index if i not in train_idx]

    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]

    train_mask = index_to_mask(train_idx, size=num_nodes)
    val_mask = index_to_mask(val_idx, size=num_nodes)
    test_mask = index_to_mask(test_idx, size=num_nodes)
    return train_mask, val_mask, test_mask

class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError(f'Step ({step}) > total number of steps ({self.total_steps}).')
        
class EMACosineScheduler:
    def __init__(self, base_value, total_steps):
        self.base_value = base_value
        self.total_steps = total_steps

    def get(self, step):
        if step >= self.total_steps:
            return 1.0
        
        return 1.0 - (1.0 - self.base_value) * (np.cos(np.pi * step / self.total_steps) + 1.0) / 2.0



sweep_config = {
    'method': 'bayes',
    # 'metric': {'name': 'Eval/Test_Score_Mean', 'goal': 'maximize'},
    'metric': {'name': 'Sweep_Target/Validation_Score', 'goal': 'maximize'}, # <-- THE FIX
    'parameters': {
        'lr': {'values': [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]},
        'sheaf_lr': {'values': [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05]},
        'weight_decay': {'values': [5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1]},
        'sheaf_decay': {'values': [5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1]},
        'eval_lr': {'values': [0.05]},
        'eval_weight_decay': {'values': [1e-4]},
        'eval_patience': {'values': [100]},


        'temp': {'distribution': 'uniform', 'min': 0.05, 'max': 1.5}, 
        'num_prototypes': {'values': [8, 16, 32, 64, 128, 256]},
        'sk_iter': {'values': [3, 4]},
        'proj_hidden_dim': {'values': [512, 1024]},
        'proj_dim': {'values': [32, 64]},
        'eps': {'distribution': 'uniform', 'min': 0.01, 'max': 0.1}, 
 
        
        'layers': {'values': [1, 2, 3, 4, 5, 6]},
        'hidden_channels': {'values': [32, 64, 128, 256]},
        'd': {'values': [2, 4, 8]}, # d=1 makes it equivalent to a standard GCN
        'dropout': {'values': [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]},
        'input_dropout': {'values': [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]},
        'view_strategy': {'value': 'post_sheaf_transport'}, 
        'target_rep_dim': {'values': [512]},
        
        'epochs': {'value': 1000},
        'use_markov_stability': {'values': [True]},
        'markov_weight': {'distribution': 'uniform', 'min': 0.01, 'max': 0.4},
        'markov_times': {'values': [[1,2], [1, 2, 4], [1, 2, 4, 8]]},
        'markov_fast_cumulative': {'values': [True]},
        'markov_adj_norm': {'values': ['row']},
        'swag_target_source': {'values': ['local']},
        'swag_reverse_weight': {'distribution': 'uniform', 'min': 0.1, 'max': 1.0},
        'no_markov_interval_rescale': {'values': [False]},
        'early_stop_loss': {'value': True},
        'patience': {'values': [50]},
        'restore_best': {'value': True}
    }
}


sweep_config_homo = {
    'method': 'bayes',
    'metric': {'name': 'Sweep_Target/Validation_Score', 'goal': 'maximize'},
    'parameters': {
        'lr': {'values': [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]},
        'sheaf_lr': {'values': [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05]},
        'weight_decay': {'values': [5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1]},
        'sheaf_decay': {'values': [5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1]},
        'eval_lr': {'values': [0.05]},
        'eval_weight_decay': {'values': [1e-4]},
        'eval_patience': {'values': [100]},


        'temp': {'distribution': 'uniform', 'min': 0.05, 'max': 1.2}, 
        'num_prototypes': {'values': [8, 16, 32, 64, 128, 256]},
        'sk_iter': {'values': [3, 4]},
        'proj_hidden_dim': {'values': [512, 1024]},
        'proj_dim': {'values': [32, 64]},
        'eps': {'distribution': 'uniform', 'min': 0.01, 'max': 0.1}, 
        

        'layers': {'values': [1, 2]}, 
        'hidden_channels': {'values': [32, 64, 128, 256]},
        'd': {'values': [2, 4, 8]}, # d=1 makes it equivalent to a standard GCN
        'dropout': {'values': [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]},
        'input_dropout': {'values': [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]},
        'view_strategy': {'value': 'post_sheaf_transport'}, 
        'epochs': {'value': 1000},
        'target_rep_dim': {'values': [512]},
    

        'use_markov_stability': {'values': [True]},
        'markov_weight': {'distribution': 'uniform', 'min': 0.01, 'max': 0.4},
        'markov_times': {'values': [[1,2], [1, 2, 4], [1, 2, 4, 8]]},
        'markov_fast_cumulative': {'values': [True]},
        'markov_adj_norm': {'values': ['row']},
        'swag_target_source': {'values': ['local']},
        'swag_reverse_weight': {'distribution': 'uniform', 'min': 0.1, 'max': 1.0},
        'no_markov_interval_rescale': {'values': [False]},

        'early_stop_loss': {'value': True},
        'patience': {'values': [50]},
        'restore_best': {'value': True}
    } 
}



def train_sweep(custom_config=None):
    model = None
    optimizer = None
    base_encoder = None
    dataset_obj = None
    data = None
    feat = None
    edge_index = None
    label = None
    embeds_torch = None
    final_maps = None
    full_preds = None
    eval_label = None
    out_1 = None
    out_2 = None
    z1 = None
    z2 = None
    loss = None

    hard_cleanup()
    # nuke any ghost tensors from previous W&B runs that crashed

    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        #parser
        parser = argparse.ArgumentParser(description="SwAG for Heterophilic Graphs")
        parser.add_argument('--dataset', type=str, default='roman-empire')
        parser.add_argument('--cuda', type=int, default=0)
        parser.add_argument('--norm', type=str, default='group')
        parser.add_argument('--epochs', type=int, default=500)
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--sheaf_decay', type=float, default=5e-4)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--sheaf_lr', type=float, default=0.001)
        
        # for nsd
        parser.add_argument('--model', type=str, default='DiagSheaf', choices=['DiagSheaf', 'BundleSheaf', 'GeneralSheaf', "GCN", "GraphSAGE", "MLP"])
        parser.add_argument('--layers', type=int, default=2)
        parser.add_argument('--d', type=int, default=2)
        parser.add_argument('--sheaf_act', type=str, default='tanh')
        parser.add_argument('--second_linear', action='store_true')
        parser.add_argument('--orth', type=str, choices=['matrix_exp', 'cayley', 'householder', 'euler'], default='householder')
        parser.add_argument('--use_act', action='store_true', default=True)
        parser.add_argument('--add_lp', action='store_true', default=False)
        parser.add_argument('--add_hp', action='store_true', default=False)
        parser.add_argument('--dropout', type=float, default=0.4)
        parser.add_argument('--input_dropout', type=float, default=0.2)
        parser.add_argument('--left_weights', action='store_true', default=True)
        parser.add_argument('--right_weights', action='store_true', default=True)
        parser.add_argument('--linear', action='store_true', default=False)
        parser.add_argument('--normalised', action='store_true', default=True)
        parser.add_argument('--deg_normalised', action='store_true', default=False)
        parser.add_argument('--sparse_learner', action='store_true', default=False)
        parser.add_argument('--edge_weights', action='store_true', default=True)
        parser.add_argument('--proj_dim', type=int, default=64)
        parser.add_argument('--proj_hidden_dim', type=int, default=512)

        
        parser.add_argument('--early_stop_loss', action='store_true')
        parser.add_argument('--patience', type=int, default=100)
        parser.add_argument('--early_stop_min_delta', type=float, default=1e-4)
        parser.add_argument('--early_stop_warmup', type=int, default=100)
        parser.add_argument('--restore_best', action='store_true')
        parser.add_argument('--checkpoint_dir', type=str, default='pkl')

        parser.add_argument('--drop_prob', type=float, default=0.2)
        parser.add_argument('--sk_iter', type=int, default=3)
        parser.add_argument('--use_proj', type=bool, default=True)
        parser.set_defaults(use_proj=True)
        parser.add_argument("--seed", type=int, default=42)

        parser.add_argument("--deterministic", action="store_true")

        parser.add_argument('--num_prototypes', type=int, nargs='+', default=[64], 
                        help='Number of prototypes. Pass one (e.g., 64) or multiple (e.g., 64 128)')




        parser.add_argument('--run_mid_probe', action='store_true', help='Run fast mid-training eval (Proxy AUC/Acc)')
        parser.add_argument('--run_visuals', action='store_true', help='Run t-SNE, Sieve, and Semantic Grids')
        parser.add_argument('--run_theory_audits', action='store_true', help='Run Dirichlet, Section Error, and Lipschitz')
        parser.add_argument('--run_expensive_evals', action='store_true', help='Run Finetuning gap and Label Rate sensitivity')

        parser.add_argument('--loss_type', type=str, default='swag', choices=['swag', 'ntxent'])
        parser.add_argument('--disable_sinkhorn', action='store_true')
        parser.add_argument('--asymmetric_mode', type=str, default='symmetric', choices=['symmetric', 'v1_predicts_v2', 'v2_predicts_v1'])
        parser.add_argument('--wide_feature_ablation', action='store_true', help='Force d=1 and expand hidden channels')

        parser.add_argument('--target_rep_dim', type=int, default=None, help="Forces hidden_channels = target / d")
        parser.add_argument('--eval_on_proj', action='store_true', help="Run linear probe on the projection head output")
  


        parser.add_argument('--swag_target_source', type=str, default='local',
                            choices=['local', 'global', 'mixed'])
        parser.add_argument('--swag_target_mixed_alpha', type=float, default=0.5)
        parser.add_argument('--swag_reverse_weight', type=float, default=None,
                            help='Soft asymmetric swag. If set, loss = CE(q2,p1)+rho*CE(q1,p2).')


        parser.add_argument('--use_markov_stability', action='store_true')
        parser.add_argument('--markov_weight', type=float, default=0.0)
        parser.add_argument('--markov_times', type=int, nargs='+', default=[1, 2, 4])
        parser.add_argument('--markov_adj_norm', type=str, default='row', choices=['row', 'sym', 'none'])
        parser.add_argument('--markov_self_loops', action='store_true', default=True)
        parser.add_argument('--no_markov_self_loops', action='store_false', dest='markov_self_loops')
        parser.add_argument('--markov_fast_cumulative', action='store_true',
                            help='Exact cumulative Markov propagation: compute P^t S for all requested t in one pass up to max(t).')
        parser.add_argument('--markov_target_only', action='store_true',
                            help='Use Markov diffusion only to form stop-gradient targets; predictions remain local scores.')
        parser.add_argument('--markov_time_sampling', type=str, default='none',
                            choices=['none', 'cyclic', 'random'],
                            help='Use all Markov times, or one sampled/cycled Markov time per active epoch.')
        parser.add_argument('--markov_update_interval', type=int, default=1,
                            help='Compute the Markov branch every m epochs. m=1 means every epoch.')
        parser.add_argument('--markov_interval_rescale', action='store_true', default=True,
                            help='Scale active Markov loss by update interval to preserve expected contribution.')
        parser.add_argument('--no_markov_interval_rescale', action='store_false',
                            dest='markov_interval_rescale')
       

        parser.add_argument("--eval_lr", type=float, default=0.01)
        parser.add_argument("--eval_weight_decay", type=float, default=0.0)
        parser.add_argument("--eval_steps", type=int, default=2000)
        parser.add_argument("--eval_patience", type=int, default=50)

        
    
        if custom_config is None:
            # --- SWEEP MODE ---
            args, _ = parser.parse_known_args()
            
            # This seeds W&B with all your parser defaults
            wandb.init(config=vars(args)) 
            
            # wandb.config now contains the defaults + the specific sweep overrides.
            # We overwrite the args namespace with these updated values in one sweep:
            vars(args).update(wandb.config)
            
            # Set derived variable
            args.use_sinkhorn = not args.disable_sinkhorn

        
            args.run_mid_probe = False
            args.run_visuals = False
            args.run_theory_audits = False
            args.run_expensive_evals = False


        else:
            args, _ = parser.parse_known_args([]) 
            
            for key, val in custom_config.items():
                setattr(args, key, val)
            args.use_sinkhorn = not getattr(args, 'disable_sinkhorn', False)
            
            proj_name = os.getenv("WANDB_PROJECT", "My-Code-Graphswag-Sweep")
            run = wandb.init(project=proj_name, config=vars(args), name=f"run_{args.dataset}")
            log_run_id("SINGLE_RUN", args.dataset, run.id, proj_name)

            if not hasattr(args, "khop_include_self"):
                args.khop_include_self = True

            if not hasattr(args, "lg_sigreg_detach_target"):
                args.lg_sigreg_detach_target = True

            # Better: force default manually after parsing
            args.khop_include_self = getattr(args, "khop_include_self", True)


        if not isinstance(args.num_prototypes, list):
            args.num_prototypes = [args.num_prototypes]

        seed_everything(args.seed, deterministic=getattr(args, "deterministic", False))


        if getattr(args, 'target_rep_dim', None) is not None:
            args.hidden_channels = max(1, args.target_rep_dim // args.d)
            
            print(f"\n[!] CONSTANT CAPACITY ABLATION ACTIVE:")
            print(f"    -> Target Representation Dim: {args.target_rep_dim}")
            print(f"    -> Current d: {args.d} | Forced hidden_channels: {args.hidden_channels}\n")
            
            if wandb.run is not None:
                # 1. Log a secondary safe metric
                wandb.config.update({"actual_hidden_channels": args.hidden_channels}, allow_val_change=True)
                
                # 2. Hack W&B's internal lock (Version Agnostic)
                if hasattr(wandb.config, '_locked'):
                    if isinstance(wandb.config._locked, dict):
                        wandb.config._locked.pop('hidden_channels', None)
                    elif isinstance(wandb.config._locked, set):
                        wandb.config._locked.discard('hidden_channels')
                    
                # 3. Apply the actual update
                wandb.config.update({"hidden_channels": args.hidden_channels}, allow_val_change=True)
 
        if getattr(args, 'wide_feature_ablation', False):
            orig_d = args.d
            orig_hidden = args.hidden_channels
            
            # Flatten geometry to standard GCN (d=1)
            args.d = 1
            # Multiply channels to keep total representation capacity identical
            args.hidden_channels = orig_hidden * orig_d
            
            print(f"\n[!] WIDE FEATURE ABLATION ACTIVE:")
            print(f"    -> Original Capacity: d={orig_d}, hidden_channels={orig_hidden}")
            print(f"    -> Ablated Capacity:  d={args.d}, hidden_channels={args.hidden_channels}\n")
            
            # Force W&B to reflect the executed dimensions so your logs are accurate
            if wandb.run is not None:
                wandb.config.update({"d": args.d, "hidden_channels": args.hidden_channels}, allow_val_change=True)
 

        device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
        args.device = device

        dense_datasets = ['tolokers']
        if args.dataset.lower() in dense_datasets:
            args.sparse_learner = True

        dataset_obj = get_dataset(args.dataset)
        data = dataset_obj[0] if isinstance(dataset_obj, list) else dataset_obj


 
        data = data.to(device)
        data = data.contiguous()
        
        feat = data.x

        edge_index = data.edge_index.contiguous()
        label = data.y

        n_classes = len(torch.unique(label))

        is_binary_task = (n_classes == 2)
        
        train_rate, val_rate = 0.6, 0.2
        percls_trn = int(round(train_rate * len(label) / max(1, n_classes)))
        val_lb = int(round(val_rate * len(label)))
        
        input_dim = feat.shape[1]
        final_d = args.d + (1 if args.add_lp else 0) + (1 if args.add_hp else 0)
        rep_size = args.hidden_channels * final_d

        args.input_dim = input_dim
        args.output_dim = rep_size   
        args.nclass = n_classes 
        args.graph_size = feat.shape[0]

        args_dict = vars(args)

        base_encoder = build_nsd_encoder(args_dict, edge_index.cpu()).to(device)

        def compute_proxy_val_score():
            model.eval()

            with torch.no_grad():
                encoder_out_eval = model.forward_encoder(feat)

                if isinstance(encoder_out_eval, dict):
                    z_eval = encoder_out_eval["z"]
                elif isinstance(encoder_out_eval, tuple):
                    z_eval = encoder_out_eval[0]
                else:
                    z_eval = encoder_out_eval

                z_eval = F.normalize(z_eval.detach(), dim=1)

            X_train = z_eval[t_mask].cpu().numpy()
            y_train = label[t_mask].cpu().numpy()
            X_val = z_eval[v_mask].cpu().numpy()
            y_val = label[v_mask].cpu().numpy()

            clf = LogisticRegression(
                solver='lbfgs',
                max_iter=300,
                class_weight='balanced',
                random_state=42,
            )
            clf.fit(X_train, y_train)

            if is_binary_task:
                try:
                    y_proba = clf.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, y_proba)
                except ValueError:
                    score = 0.0
            else:
                score = clf.score(X_val, y_val)

            model.train()
            return float(score)

        def push_hidden_tensors_to_device(obj, target_device):
            """Recursively finds tensors stored as raw attributes and moves them."""
            for attr, val in vars(obj).items():
                if isinstance(val, torch.Tensor):
                    setattr(obj, attr, val.to(target_device))
                elif isinstance(val, (tuple, list)):
                    # Handle tuples/lists of tensors (like left_right_idx)
                    new_val = type(val)(v.to(target_device) if isinstance(v, torch.Tensor) else v for v in val)
                    setattr(obj, attr, new_val)

        # Apply to the encoder itself
        push_hidden_tensors_to_device(base_encoder, device)
        
        # Apply to the Laplacian Builder (where all the sparse indices live)
        if hasattr(base_encoder, 'laplacian_builder'):
            push_hidden_tensors_to_device(base_encoder.laplacian_builder, device)


        def edge_index_to_sparse_adj(
            edge_index: torch.Tensor,
            num_nodes: int,
            edge_weight: Optional[torch.Tensor] = None,
            device: Optional[torch.device] = None,
        ) -> torch.Tensor:
            device = device if device is not None else edge_index.device

            edge_index = edge_index.to(device)

            if edge_weight is None:
                edge_weight = torch.ones(edge_index.size(1), device=device)
            else:
                edge_weight = edge_weight.to(device)

            A = torch.sparse_coo_tensor(
                indices=edge_index,
                values=edge_weight,
                size=(num_nodes, num_nodes),
                device=device,
            )

            return A.coalesce()

        model = GraphSwAG(
            encoder=base_encoder,
            encoder_output_dim=rep_size,
            proj_dim=args.proj_dim,
            proj_hidden_dim=args.proj_hidden_dim,
            num_prototypes=args.num_prototypes, 
            sk_iter = args.sk_iter,
            temp=args.temp,
            loss_type=args.loss_type,
            use_sinkhorn=args.use_sinkhorn,
            use_proj=args.use_proj,
            eps=args.eps,
            asymmetric_mode=args.asymmetric_mode,
            swag_reverse_weight=args.swag_reverse_weight,
            use_markov_stability=args.use_markov_stability,
            markov_weight=args.markov_weight,
            markov_times=args.markov_times,
            markov_adj_norm=args.markov_adj_norm,
            markov_self_loops=args.markov_self_loops,
            markov_fast_cumulative=args.markov_fast_cumulative,
            markov_target_only=args.markov_target_only,
            markov_time_sampling=args.markov_time_sampling,
            markov_update_interval=args.markov_update_interval,
            markov_interval_rescale=args.markov_interval_rescale,
        ).to(device)


        sheaf_learner_params, other_params = base_encoder.grouped_parameters()

        if model.projection_head is not None:
            other_params.extend(list(model.projection_head.parameters()))

        if model.prototypes is not None:
            other_params.extend(list(model.prototypes.parameters()))

        if getattr(model, "view_alignment_module", None) is not None:
            other_params.extend(list(model.view_alignment_module.parameters()))

        # Remove frozen params
        sheaf_learner_params = [p for p in sheaf_learner_params if p.requires_grad]
        other_params = [p for p in other_params if p.requires_grad]

        param_groups = []

        if len(sheaf_learner_params) > 0:
            param_groups.append({
                "params": sheaf_learner_params,
                "lr": args.sheaf_lr,
                "weight_decay": args.sheaf_decay,
            })

        if len(other_params) > 0:
            param_groups.append({
                "params": other_params,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            })

        optimizer = torch.optim.Adam(param_groups)

        lr_scheduler = CosineDecayScheduler(
            args.lr,
            warmup_steps=50,
            total_steps=args.epochs,
        )

        sheaf_lr_ratio = args.sheaf_lr / args.lr if args.lr > 0 else 1.0



        best_loss = float("inf")
        best_epoch = 0
        cnt_wait = 0

        platonov_datasets = [
            'roman-empire', 'amazon-ratings', 'minesweeper', 
            'tolokers', 'questions', 'chameleon-filtered', 'squirrel-filtered'
        ]

        if args.dataset.lower() in platonov_datasets:
            if data.train_mask.dim() > 1:
                t_mask, v_mask = data.train_mask[:, 0], data.val_mask[:, 0]
            else:
                t_mask, v_mask = data.train_mask, data.val_mask
        else:
            t_mask_np, v_mask_np, _ = random_splits(label, n_classes, percls_trn, val_lb, seed=42)
            t_mask = torch.BoolTensor(t_mask_np).to(device)
            v_mask = torch.BoolTensor(v_mask_np).to(device)

        best_loss = float("inf")
        best_epoch = 0
        cnt_wait = 0

        os.makedirs(args.checkpoint_dir, exist_ok=True)

        run_tag = wandb.run.id if wandb.run is not None else str(int(time.time()))
        best_ckpt_path = os.path.join(
            args.checkpoint_dir,
            f"best_model_{args.dataset}_{run_tag}.pt"
)


        for epoch in tqdm(range(args.epochs)):
            model.train()


            base_lr = lr_scheduler.get(epoch)

            for param_group in optimizer.param_groups:
                if param_group.get("name") == "sheaf":
                    param_group["lr"] = base_lr * sheaf_lr_ratio
                else:
                    param_group["lr"] = base_lr

            optimizer.zero_grad(set_to_none=True)
            h = feat

      
            if args.view_strategy in ['post_stalk_mask', 'post_sheaf_transport', 'post_sheaf_laplacian', 'post_mean_pool', 'post_khop_agg']:
                encoder_out = model.forward_encoder(h)
                
                if isinstance(encoder_out, dict):
                    sheafx = encoder_out.get("z", None)
                    maps = encoder_out.get("maps", None)

                    view1 = encoder_out.get("view1_laplacian_diag", None)

                    raw_view2 = encoder_out.get("view2_laplacian_non_diag", None)
                    view2 = -raw_view2 if raw_view2 is not None else None

                elif isinstance(encoder_out, tuple):
                    sheafx = encoder_out[0] if len(encoder_out) > 0 else None
                    maps = encoder_out[1] if len(encoder_out) > 1 else None
                    view1 = encoder_out[2] if len(encoder_out) > 2 else None

                    raw_view2 = encoder_out[3] if len(encoder_out) > 3 else None
                    view2 = -raw_view2 if raw_view2 is not None else None

                else:
                    sheafx = encoder_out
                    maps = None
                    view1 = None
                    view2 = None

                if sheafx is None:
                    raise ValueError(
                        f"Encoder returned no node embeddings. "
                        f"type(encoder_out)={type(encoder_out)}, encoder_out={encoder_out}"
                    )
                
                if args.view_strategy == 'post_stalk_mask':
                    latent_views = stalk_masking(sheafx, final_d=final_d, emb_dim=args.hidden_channels)
                    ortho_penalty = 0.0 
    
                elif args.view_strategy == 'post_mean_pool':
                    from aug import generate_mean_pool_views
                    latent_views = generate_mean_pool_views(sheafx, edge_index, feat.size(0))
                    ortho_penalty = 0.0

                elif args.view_strategy == 'post_khop_agg':
                    from aug import generate_khop_aggregate_views

                    latent_views = generate_khop_aggregate_views(
                        x=sheafx,
                        edge_index=edge_index,
                        num_nodes=feat.size(0),
                        k=args.khop_k,
                        include_self=args.khop_include_self,
                        to_undirected_graph=args.khop_to_undirected,
                        exact_k=args.khop_exact,
                        combine=args.khop_combine,
                        residual_alpha=args.khop_residual_alpha,
                        normalize_each_step=args.khop_normalize_each_step,
                    )

                    ortho_penalty = 0.0
                else:
                    use_lap_diag = (args.view_strategy == 'post_sheaf_laplacian')
           
                    if use_lap_diag:
                        latent_views = [view1, view2]
                    else:
                        latent_views = [sheafx, view2]
                    
    
                z1, z1_raw = model.forward_projection(latent_views[0], return_raw=True)
                z2, z2_raw = model.forward_projection(latent_views[1], return_raw=True)
                x1=latent_views[0]
                x2=latent_views[1]
           
                if args.loss_type == "swag":
                    out_1 = model.prototypes(z1)
                    out_2 = model.prototypes(z2)
                   
                else:
                    out_1 = z1
                    out_2 = z2
            else:
                raise ValueError(f"Unknown strategy: {args.view_strategy}")

            A = edge_index_to_sparse_adj(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                device=x1.device,
            )

            loss_ssl, loss_parts = model.compute_loss(
                x1,
                x2,
                adj=A,
                epoch=epoch,
                return_parts=True,
            )

            loss = loss_ssl 


            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())

            log_dict = {
                "Train/Loss_Total": loss_value,
                "Train/LR_Other": base_lr,
                "Train/LR_Sheaf": base_lr * sheaf_lr_ratio,
            }
            if 'loss_parts' in locals() and loss_parts is not None:
                for k, v in loss_parts.items():
                    if torch.is_tensor(v):
                        log_dict[f"TrainParts/{k}"] = float(v.detach().cpu().item())
                    else:
                        log_dict[f"TrainParts/{k}"] = v
           


            stop_training = False

            if args.early_stop_loss and epoch >= args.early_stop_warmup:
                improved = loss_value < (best_loss - args.early_stop_min_delta)

                if improved:
                    best_loss = loss_value
                    best_epoch = epoch
                    cnt_wait = 0

                    if args.restore_best:
                        torch.save(model.state_dict(), best_ckpt_path)

                else:
                    cnt_wait += 1

                log_dict["EarlyStop/Best_Loss"] = best_loss
                log_dict["EarlyStop/Best_Epoch"] = best_epoch
                log_dict["EarlyStop/Counter"] = cnt_wait

                if cnt_wait >= args.patience:
                    print(
                        f"\n[EarlyStop] Loss plateau detected at epoch {epoch}. "
                        f"Best epoch={best_epoch}, best loss={best_loss:.6f}"
                    )
                    stop_training = True

            
     
            if epoch > 0 and epoch % 50 == 0 and args.run_mid_probe:
                model.eval()
                with torch.no_grad():
                    encoder_out_eval = model.forward_encoder(feat)
                    
                    if isinstance(encoder_out_eval, dict):
                        z_eval = encoder_out_eval["z"]
                        maps = encoder_out_eval.get("maps")
                    elif isinstance(encoder_out_eval, tuple):
                        z_eval = encoder_out_eval[0]
                        maps = encoder_out_eval[1]
                    else:
                        z_eval = encoder_out_eval
                        maps = None
                        
                    z_eval = z_eval.detach()
                    
                 

                        
                    X_train = F.normalize(z_eval[t_mask], dim=1).cpu().numpy()
                    y_train = label[t_mask].cpu().numpy()
                    X_val = F.normalize(z_eval[v_mask], dim=1).cpu().numpy()
                    y_val = label[v_mask].cpu().numpy()
                    
                    clf = LogisticRegression(solver='lbfgs', max_iter=300, class_weight='balanced', random_state=42)
                    clf.fit(X_train, y_train)
                    
                    if is_binary_task:
                        try:
                            y_proba = clf.predict_proba(X_val)[:, 1]
                            proxy_val_score = roc_auc_score(y_val, y_proba)
                        except ValueError:
                            proxy_val_score = 0.0
                        metric_name = "Epoch_Eval/Proxy_Val_AUC"
                    else:
                        proxy_val_score = clf.score(X_val, y_val)
                        metric_name = "Epoch_Eval/Proxy_Val_Acc"

                    log_dict[metric_name] = proxy_val_score

           
                    if args.loss_type == "swag":
                        # from analysis import compute_spectral_metrics_robust
                        spectral_metrics = compute_spectral_metrics_robust(z_eval)
                        log_dict["Epoch_Eval/SelfCluster"] = spectral_metrics["SelfCluster"]
                        log_dict["Epoch_Eval/RankMe"] = spectral_metrics["RankMe"]
                        
                        if maps is not None:
                            use_lap_diag = (args.view_strategy == 'post_sheaf_laplacian')
                            
                            views, _ = generate_sheaf_views(
                                z_eval, final_d=final_d, emb_dim=args.hidden_channels, 
                                maps=maps, edge_index=edge_index, num_nodes=feat.size(0),
                                use_laplacian_diagonal=use_lap_diag,
                                use_ricci_prior=args.use_ricci_prior
                            )
                            v2 = views[1] 
                            
                            section_error = torch.norm(z_eval - v2, p=2, dim=1).mean().item()
                            log_dict["Epoch_Eval/Section_Error_Mean"] = section_error


            wandb.log(log_dict, step=epoch)
            if stop_training:
                break
     
        if args.early_stop_loss and args.restore_best and os.path.exists(best_ckpt_path):
            print(
                f"[*] Restoring best model from epoch {best_epoch} "
                f"with loss={best_loss:.6f}"
            )
            state = torch.load(best_ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=True)


        model.eval()
        with torch.no_grad():
            out_final = model.forward_encoder(feat)
            
            if isinstance(out_final, dict):
                embeds_torch = out_final["z"].detach()
                final_maps = out_final.get("maps")
            elif isinstance(out_final, tuple):
                embeds_torch = out_final[0].detach()
                final_maps = out_final[1]
            else:
                embeds_torch = out_final.detach()
                final_maps = None
            

            if getattr(args, 'eval_on_proj', False) and model.use_projection:
                print("\n[*] Evaluating on Projection Head instead of Base Encoder...")
                embeds_torch = model.forward_projection(embeds_torch).detach()
                

                rep_size = embeds_torch.size(1) 
     
        from evaluator import run_linear_eval_protocol
        _, _, _, _, full_preds, eval_label = run_linear_eval_protocol(
            embeds_torch, label, data, n_classes, is_binary_task, rep_size, device, args.eval_lr, args.eval_weight_decay, args.eval_steps, args.eval_patience, dataset_name=args.dataset
        )
            

        model.encoder_input_features = feat 
        
        run_post_training_audits(
            model, embeds_torch, final_maps, label, eval_label, edge_index, full_preds, args.dataset, args
        )



    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("[OOM] CUDA out of memory caught. Cleaning before next sweep run.")
            hard_cleanup()
        raise

    finally:
        # Break references held by local variables.
        model = None
        optimizer = None
        base_encoder = None
        dataset_obj = None
        data = None
        feat = None
        edge_index = None
        label = None
        embeds_torch = None
        final_maps = None
        full_preds = None
        eval_label = None
        out_1 = None
        out_2 = None
        z1 = None
        z2 = None
        loss = None

        hard_cleanup()

if __name__ == "__main__":
    
    # 1. Set up a top-level "Router" parser
    router_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    
    # Force the user to choose exactly one execution mode
    group = router_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run_sweep', action='store_true', help="Start a full random wandb sweep")
    group.add_argument('--run_best', action='store_true', help="Run the single best config from JSON")
    group.add_argument('--run_hybrid_sweep', action='store_true', help="Sweep a specific subset of parameters while locking others to JSON best")
    
    # Require the dataset name
    router_parser.add_argument('--dataset', type=str, required=True, help="Dataset to use")
    router_parser.add_argument('--config_file', type=str, default='best_configs.json', help="Path to your JSON file")

    # NEW: Dynamic Hybrid Sweep Arguments (LEGACY)
    router_parser.add_argument('--sweep_param', type=str, help="Parameter name to sweep in hybrid mode (e.g., 'layers')")
    router_parser.add_argument('--sweep_values', nargs='+', help="Values to test (e.g., 1 2 4 8 16)")

    # NEW: Multi-grid support (A MORE GENERAL WAY THAN sweep_param and sweep_values)
    router_parser.add_argument('--grid', nargs='+', action='append', 
                               help="Usage: --grid temp 0.1 0.2 --grid lr 0.01 0.001")

    # 2. Intercept the routing arguments, leave the rest alone
    route_args, remaining_args = router_parser.parse_known_args()

    # 3. Rebuild sys.argv so the inner parser inside train_sweep() doesn't crash 
    sys.argv = [sys.argv[0], '--dataset', route_args.dataset] + remaining_args

    # Helper function to load and clean JSON config
    def get_clean_json_config(dataset, config_file):
        if not os.path.exists(config_file):
            print(f"Error: Could not find '{config_file}'.")
            sys.exit(1)
        with open(config_file, 'r') as f:
            try:
                all_configs = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: '{config_file}' is not formatted correctly.")
                sys.exit(1)
                
        lookup_key = dataset

            
        if lookup_key not in all_configs:
            print(f"\nError: Dataset '{lookup_key}' not found in {config_file}.")
            sys.exit(1)
            
        raw_config = all_configs[lookup_key]
        clean_config = {}
        for key, val in raw_config.items():
            if key == "_wandb": continue
            if isinstance(val, dict) and "value" in val: clean_config[key] = val["value"]
            else: clean_config[key] = val

       
        return clean_config
    
    def override_args(clean_config, remaining_args, dataset_name=None):
        if remaining_args:
            print(f"\n{'='*60}")
            if dataset_name:
                print(f"  REPRODUCTION MODE: {dataset_name.upper()}")
            print(f"  Applying CLI Overrides to JSON Config")
            print(f"{'='*60}")
            print(f"  {'PARAMETER':20s} | {'JSON VALUE':20s} -> {'CLI OVERRIDE'}")
            print(f"  {'-'*58}")

            i = 0
            while i < len(remaining_args):
                arg = remaining_args[i]
                if arg.startswith('--'):
                    key = arg[2:] 
                    val_list = []
                    i += 1
                    
                    while i < len(remaining_args) and not remaining_args[i].startswith('--'):
                        val_list.append(remaining_args[i])
                        i += 1

                    if not val_list:
                        new_val = True 
                    elif len(val_list) == 1:
                        v = val_list[0]
                        if v.lower() == 'true': new_val = True
                        elif v.lower() == 'false': new_val = False
                        else:
                            try: new_val = int(v)
                            except ValueError:
                                try: new_val = float(v)
                                except ValueError: new_val = v 
                    else:
                        parsed_list = []
                        for v in val_list:
                            try: parsed_list.append(int(v))
                            except ValueError:
                                try: parsed_list.append(float(v))
                                except ValueError: parsed_list.append(v)
                        new_val = parsed_list

    
                    old_val = clean_config.get(key, "N/A (Parser Default)")
                    print(f"  • {key:18s} | {str(old_val):20s} -> {str(new_val)}")
            
                    clean_config[key] = new_val
                else:
                    i += 1
            print(f"{'='*60}\n")

        return clean_config


    if route_args.run_best:
        clean_config = get_clean_json_config(route_args.dataset, route_args.config_file)
                
        if remaining_args:
            clean_config = override_args(clean_config, remaining_args, route_args.dataset)

                
        train_sweep(custom_config=clean_config)


    elif route_args.run_sweep:
        print(f"\n>>> Starting FULL W&B Sweep for '{route_args.dataset}' <<<\n")
        
        count = 180
        if route_args.dataset in ["cora", "citeseer", "pubmed", "amazon-photo", "amazon-computer", "coauthor-cs"]:
            current_sweep_config = sweep_config_homo
        elif route_args.dataset.startswith("csbm_"):
            current_sweep_config = sweep_config_csbm
        else:
            current_sweep_config = sweep_config

        proj_name = os.getenv("WANDB_PROJECT", "My-Code-Graphswag-Sweep")
        sweep_id = wandb.sweep(current_sweep_config, project=proj_name)
        log_run_id("FULL_SWEEP", route_args.dataset, sweep_id, proj_name)
        wandb.agent(sweep_id, train_sweep, count=count)


    elif route_args.run_hybrid_sweep:
        if not route_args.grid and (not route_args.sweep_param or not route_args.sweep_values):
            print("Error: --run_hybrid_sweep requires either (--sweep_param AND --sweep_values) OR --grid")
            sys.exit(1)
            
        base_config = get_clean_json_config(route_args.dataset, route_args.config_file)

        if remaining_args:
            base_config = override_args(base_config, remaining_args, route_args.dataset)
        
        hybrid_sweep_config = {
            'method': 'grid',  
            'metric': {'name': 'Eval/Test_Score_Mean', 'goal': 'maximize'},
            'parameters': {k: {'value': v} for k, v in base_config.items()}
        }


        grid_map = {}

        if route_args.sweep_param and route_args.sweep_values:
            grid_map[route_args.sweep_param] = route_args.sweep_values

        if route_args.grid:
            for item in route_args.grid:
                grid_map[item[0]] = item[1:]

        total_runs = 1
        sweep_names = []
        
        for param_name, raw_values in grid_map.items():
            parsed_vals = []
            for v in raw_values:
                try: parsed_vals.append(int(v))
                except ValueError:
                    try: parsed_vals.append(float(v))
                    except ValueError: parsed_vals.append(v)
            
            hybrid_sweep_config['parameters'][param_name] = {'values': parsed_vals}
            total_runs *= len(parsed_vals)
            sweep_names.append(param_name)

        print(f"\n>>> Launching Hybrid Sweep for '{route_args.dataset}'")
        print(f">>> Grid: {' x '.join([f'{k}({len(v)})' for k, v in grid_map.items()])}")
        print(f">>> Total Permutations: {total_runs}\n")
        
        proj_name = os.getenv("WANDB_PROJECT", "SwAG-Mechanistic-Audit")
        sweep_id = wandb.sweep(hybrid_sweep_config, project=proj_name)
        
        tag = "_".join(sweep_names)[:20]
        log_run_id(f"GRID_{tag}", route_args.dataset, sweep_id, proj_name)
        
        wandb.agent(sweep_id, train_sweep, count=total_runs)