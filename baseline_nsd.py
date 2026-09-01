import argparse
import torch
import numpy as np

from dataset import get_dataset
from encoder import build_nsd_encoder
from analysis import run_downstream_audit_robust, compute_spectral_metrics_robust, compute_manifold_metrics_robust

def get_random_nsd_baseline():
    parser = argparse.ArgumentParser(description="Randomly Initialized NSD Baseline")
    

    parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--model', type=str, default='DiagSheaf')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--norm', type=str, default='group')
    parser.add_argument('--sheaf_act', type=str, default='tanh')
    parser.add_argument('--orth', type=str, default='householder')

    parser.add_argument('--second_linear', action='store_true')
    parser.add_argument('--use_act', action='store_true', default=True)
    parser.add_argument('--add_lp', action='store_true')
    parser.add_argument('--add_hp', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.0) 
    parser.add_argument('--input_dropout', type=float, default=0.0)
    parser.add_argument('--left_weights', action='store_true', default=True)
    parser.add_argument('--right_weights', action='store_true', default=True)
    parser.add_argument('--normalised', action='store_true', default=True)
    parser.add_argument('--deg_normalised', action='store_true', default=False)
    parser.add_argument('--sparse_learner', action='store_true', default=False)
    
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    args.device = device
    

    print(f"Loading {args.dataset}...")
    dataset_obj = get_dataset(args.dataset)
    data = dataset_obj.to(device) if not isinstance(dataset_obj, list) else dataset_obj[0].to(device)

    final_d = args.d + (1 if args.add_lp else 0) + (1 if args.add_hp else 0)
    rep_size = args.hidden_channels * final_d
    
    args_dict = vars(args)
    args_dict['input_dim'] = data.x.shape[1]
    args_dict['output_dim'] = rep_size
    args_dict['graph_size'] = data.x.shape[0]
    
    print(f"Building randomly initialized {args.model}...")
    torch.manual_seed(42) 
    encoder = build_nsd_encoder(args_dict, data.edge_index).to(device)
    

    encoder.eval()
    with torch.no_grad():
        out = encoder(data.x)
        embeds = out["z"] if isinstance(out, dict) else out
        embeds = embeds.detach()
        
    print(f"Extracted embeddings shape: {embeds.shape}")
    
    print("\n--- Running Intrinsic Audits ---")
    spec_metrics = compute_spectral_metrics_robust(embeds)
    man_metrics = compute_manifold_metrics_robust(embeds, data.y, data.edge_index)
    
    print("\nSpectral:", spec_metrics)
    print("Manifold:", man_metrics)
    
    print("\n--- Running Linear Evaluation (10-split) ---")
    downstream_results = run_downstream_audit_robust(embeds, data.y, num_splits=10)
    
    for metric, (mean, std) in downstream_results.items():
        print(f"{metric.upper()}: {mean*100:.2f}% ± {std*100:.2f}%")

if __name__ == "__main__":
    get_random_nsd_baseline()
