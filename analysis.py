import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import wandb
import torch_sparse
import umap  
import copy

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    silhouette_score, accuracy_score, f1_score, roc_auc_score, 
    average_precision_score, confusion_matrix, 
    normalized_mutual_info_score, adjusted_rand_score
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from scipy.stats import linregress
from torch_geometric.utils import get_laplacian

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# =========================================================
# 1. VISUALIZATION & PLOTTING FUNCTIONS
# =========================================================
def analyze_and_plot_sheaf_transport(edge_index, maps, labels, dataset_name="graph"):
    """
    Plots the transport distributions PER STALK DIMENSION to see if individual 
    channels specialize in homophily vs. heterophily.
    """
    if maps is None: return
    print("\n--- Running Channel-wise Sieve Analysis ---")
    
    u, v = edge_index[0], edge_index[1]
    same_label = (labels[u] == labels[v]).cpu().numpy()
    
    with torch.no_grad():
        if isinstance(maps, tuple): 
            maps = maps[0] * maps[1]
        
        # Extract per-channel transport weights
        if maps.dim() == 2:  # DiagSheaf [E, d]
            p_uv = maps.cpu().numpy()
        elif maps.dim() == 3:  # Bundle/General Sheaf [E, d, d]
            p_uv = maps.diagonal(dim1=1, dim2=2).cpu().numpy()
        else:
            p_uv = maps.unsqueeze(-1).cpu().numpy()

    num_channels = p_uv.shape[1]
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Create a grid of KDE plots
    cols = min(4, num_channels)
    rows = int(np.ceil(num_channels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if num_channels == 1: axes = np.array([axes])
    axes = axes.flatten()

    for k in range(num_channels):
        ax = axes[k]
        homo_vals = p_uv[same_label, k]
        hetero_vals = p_uv[~same_label, k]
        
        if len(homo_vals) > 0 and np.std(homo_vals) > 1e-4:
            sns.kdeplot(homo_vals, ax=ax, color='blue', label='Homophilic', fill=True, alpha=0.3)
        if len(hetero_vals) > 0 and np.std(hetero_vals) > 1e-4:
            sns.kdeplot(hetero_vals, ax=ax, color='red', label='Heterophilic', fill=True, alpha=0.3)
            
        ax.axvline(x=0.0, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f"Stalk Channel $d_{k}$", fontweight='bold')
        if k == 0: ax.legend()

    for i in range(num_channels, len(axes)): axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"plots/sheaf_transport_channel_sieve_{dataset_name}.png", dpi=300)
    
    import wandb
    try: wandb.log({"Analysis/Channel_Sieve": wandb.Image(plt)})
    except Exception: pass
    plt.close()

def plot_tsne(embeddings, labels, dataset_name="Graph"):
    """
    Computes and plots a 2D t-SNE of the continuous node embeddings,
    colored by their ground-truth semantic labels.
    """
    print(f"\n--- Running t-SNE Dimensionality Reduction ({dataset_name}) ---")
    
    # 1. Convert to numpy
    X = embeddings.cpu().numpy()
    y = labels.cpu().numpy()
    
    # 2. Run t-SNE
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 3. Apply Publication-Ready Styling safely
    try:
        import scienceplots
        plt.style.use(['science', 'no-latex']) 
        plt.rcParams.update({'font.size': 10, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10})
    except ImportError:
        sns.set_theme(style="ticks", context="paper", font_scale=1.0)
        plt.rcParams['font.family'] = 'serif'

    df = pd.DataFrame({'tsne_1': X_tsne[:, 0], 'tsne_2': X_tsne[:, 1], 'Class': y})
    
    # 4. Plotting
    fig, ax = plt.subplots(figsize=(7, 5)) 
    
    num_classes = len(np.unique(y))
    # THE FIX: Use tab20 for high-cardinality datasets for better contrast
    if num_classes <= 10:
        palette = sns.color_palette("tab10", num_classes)
    elif num_classes <= 20:
        palette = sns.color_palette("tab20", num_classes)
    else:
        palette = sns.color_palette("husl", num_classes)
    
    sns.scatterplot(
        x='tsne_1', y='tsne_2', 
        hue='Class', 
        palette=palette, 
        data=df, 
        alpha=0.6,        # THE FIX: Lower alpha to see cluster density
        s=5,              # THE FIX: Smaller dots to combat overplotting
        linewidth=0,
        ax=ax,
        rasterized=True   # THE FIX: Keeps LaTeX PDF file sizes small!
    )
    
    # Clean up labels and title
    ax.set_title(f"Unsupervised Latent Space ({dataset_name})", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    
    # THE FIX: Split long legends into 2 columns
    ncols = 2 if num_classes > 10 else 1
    
    ax.legend(
        title="Class", 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', 
        frameon=True, 
        fontsize=9,
        title_fontsize=10,
        markerscale=3, 
        ncol=ncols       # 2 columns if > 10 classes
    )
    
    # Remove top and right borders
    sns.despine(ax=ax)
    plt.tight_layout()
    
    # 5. Save High-Res
    save_name = f"plots/tsne_{dataset_name.replace(' ', '_')}_publication.pdf"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.savefig(save_name.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    
    try:
        import wandb
        wandb.log({f"Analysis/t-SNE_{dataset_name}": wandb.Image(fig)})
    except Exception:
        pass
        
    plt.close()
    print(f">>> Publication-ready t-SNE saved as '{save_name}' <<<\n")

def plot_local_section_error(model, embeddings, edge_index, final_maps, labels, preds, dataset_name="Graph"):
    r"""
    TASK 3: Local Section Error (The Anchor Proof)
    Calculates \xi_v = || z_v - Transport(N_v) ||_2 
    and compares correctly vs incorrectly classified nodes.
    """
    if final_maps is None: return

    print("\n--- Computing Local Section Error (Task 3) ---")
    N = embeddings.size(0)
    
    # Extract the stalk dimension 'd'
    d = model.encoder.d if hasattr(model.encoder, 'd') else model.encoder.final_d
    Nd = N * d
    
    with torch.no_grad():
        L_output = model.encoder.laplacian_builder(final_maps)
        L_idx, L_val = L_output[0]
        
        # Remove self-loops to get pure neighbor transport
        mask = L_idx[0] != L_idx[1]
        A_idx = L_idx[:, mask]
        A_val = -L_val[mask] 
        
        # 1. Reshape embeddings to stalk space [N*d, hidden]
        Z_stalk = embeddings.view(Nd, -1)
        
        # 2. Compute sum of transported neighbors for each stalk (using Nd bounds)
        sum_transported_stalks = torch_sparse.spmm(A_idx, A_val, Nd, Nd, Z_stalk)
        
        # 3. Reshape back to node space [N, d*hidden]
        sum_transported_neighbors = sum_transported_stalks.view(N, -1)
        
        # 4. Compute degree of each node (using standard edge_index)
        row, col = edge_index
        deg = torch.bincount(row, minlength=N).view(-1, 1).float()
        deg[deg == 0] = 1.0 # Prevent division by zero
        
        mean_transported_neighbors = sum_transported_neighbors / deg
        
        # 5. Calculate xi_v (The Local Section Error)
        xi_v = torch.norm(embeddings - mean_transported_neighbors, p=2, dim=1).cpu().numpy()

    # Split by Classification
    y_true, y_pred = labels.cpu().numpy(), preds.cpu().numpy()
    correct_mask, incorrect_mask = (y_true == y_pred), (y_true != y_pred)
    
    errors_correct = xi_v[correct_mask]
    errors_incorrect = xi_v[incorrect_mask]
    
    # Plotting
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(8, 5))
    if len(errors_correct) > 0:
        sns.kdeplot(errors_correct, color='green', label='Correctly Classified', fill=True, alpha=0.4)
        plt.axvline(x=np.median(errors_correct), color='darkgreen', linestyle='--', linewidth=2)
    if len(errors_incorrect) > 0:
        sns.kdeplot(errors_incorrect, color='orange', label='Incorrectly Classified', fill=True, alpha=0.4)
        plt.axvline(x=np.median(errors_incorrect), color='darkorange', linestyle='--', linewidth=2)

    plt.title(f"Local Section Error (Task 3): {dataset_name}", fontsize=14, fontweight='bold')
    plt.xlabel(r"Local Section Error $\xi_v$", fontsize=12)
    plt.legend()
    sns.despine()
    
    plt.savefig(f"plots/task3_section_error_{dataset_name}.png", dpi=300)
    
    import wandb
    try:
        wandb.log({"Task3/Local_Section_Error": wandb.Image(plt)})
    except Exception:
        pass
    plt.close()

def audit_intercluster_transport(edge_index, maps, cluster_assignments):
    """
    TASK 4: Inter-Cluster Transport Audit
    Shows that edges crossing SwAV clusters are suppressed (near 0).
    """
    if maps is None: return
    
    print("\n--- Auditing Inter-Cluster Transport (Task 4) ---")
    
    u, v = edge_index[0], edge_index[1]
    
    # Categorize edges based on the unsupervised SwAV assignments!
    intra_cluster = (cluster_assignments[u] == cluster_assignments[v])
    inter_cluster = ~intra_cluster
    
    with torch.no_grad():
        if maps.dim() == 2: # DiagSheaf [E, d]
            p_uv_abs = torch.abs(maps).mean(dim=1).cpu().numpy()
        else: # Fallback
            p_uv_abs = torch.abs(maps.squeeze()).cpu().numpy()
            
    p_intra = p_uv_abs[intra_cluster.cpu().numpy()]
    p_inter = p_uv_abs[inter_cluster.cpu().numpy()]
    
    # Plotting
    plt.figure(figsize=(8, 5))
    sns.kdeplot(p_intra, color='blue', label='Intra-Cluster Edges', fill=True, alpha=0.4)
    sns.kdeplot(p_inter, color='red', label='Inter-Cluster Edges', fill=True, alpha=0.4)
    
    plt.title("Inter-Cluster Boundary Formation (Task 4)", fontsize=14, fontweight='bold')
    plt.xlabel("Absolute Transport Weight $|P_{uv}|$", fontsize=12)
    plt.legend()
    sns.despine()
    
    plt.savefig("plots/task4_intercluster_transport.png", dpi=300)
    wandb.log({"Task4/Inter_Cluster_Transport": wandb.Image(plt),
               "Task4/Mean_Intra_Transport": np.mean(p_intra) if len(p_intra)>0 else 0,
               "Task4/Mean_Inter_Transport": np.mean(p_inter) if len(p_inter)>0 else 0})
    plt.close()

def plot_semantic_grids(embeddings, labels, preds, cluster_assignments, dataset_name="Graph", dim_reduction='tsne'):
    """
    Generates:
    1. Confusion Grid: Subplots = Predicted Classes. Colored by True Label. (Precision Focus)
    2. Subclass Grid: Subplots = Actual Classes. Colored by Predicted Label. (Recall Focus)
    3. Confusion Matrix: Standard actual vs predicted heatmap.
    
    dim_reduction: 'tsne' or 'umap'
    """
    method_str = "UMAP" if dim_reduction.lower() == 'umap' else "t-SNE"
    print(f"\n--- Generating Semantic & Subclass {method_str} Grids + Confusion Matrix ---")
    
    # 1. Run Dimensionality Reduction once
    X = embeddings.cpu().numpy()
    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()
    
    if dim_reduction.lower() == 'umap':
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
        file_prefix = "umap"
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
        X_2d = reducer.fit_transform(X)
        file_prefix = "tsne"
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from sklearn.metrics import confusion_matrix
    
    df = pd.DataFrame({
        'dim_1': X_2d[:, 0],
        'dim_2': X_2d[:, 1],
        'True_Label': y_true,
        'Predicted_Label': y_pred
    })
    
    classes = np.unique(y_true)
    num_classes = len(classes)
    cols = min(4, num_classes)
    rows = int(np.ceil(num_classes / cols))
    
    # Create a consistent color dictionary for BOTH grids
    base_palette = sns.color_palette("husl", num_classes)
    color_dict = {cls: color for cls, color in zip(classes, base_palette)}
    
    # ==========================================
    # FIGURE 1: The Confusion Grid (Precision Focus)
    # ==========================================
    fig1, axes1 = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if num_classes == 1: axes1 = np.array([axes1])
    axes1 = axes1.flatten()
    
    # Loop over the PREDICTED classes
    for idx, c in enumerate(classes): 
        ax = axes1[idx]
        
        # Calculate Precision for this predicted class
        subset_pred = df[df['Predicted_Label'] == c]
        total_pred = len(subset_pred)
        correct_pred = len(subset_pred[subset_pred['True_Label'] == c])
        precision = (correct_pred / total_pred * 100) if total_pred > 0 else 0.0

        # Background
        sns.scatterplot(x='dim_1', y='dim_2', data=df[df['Predicted_Label'] != c], 
                        color='lightgrey', alpha=0.3, s=10, linewidth=0, ax=ax, legend=False)
        # Foreground (Colored by True Label)
        sns.scatterplot(x='dim_1', y='dim_2', hue='True_Label', palette=color_dict, 
                        data=subset_pred, 
                        alpha=0.9, s=25, linewidth=0.5, edgecolor='black', ax=ax, legend=False)
        
        ax.set_title(f"Predicted Class: {c}\n(Actual matches: {precision:.1f}%)", fontweight='bold')
        ax.axis('off')

    for i in range(num_classes, len(axes1)): axes1[i].axis('off')
    
    # Global Legend for Fig 1
    handles1 = [mpatches.Patch(color=color_dict[c], label=f"True Class {c}") for c in classes]
    fig1.legend(handles=handles1, loc='lower center', ncol=cols, bbox_to_anchor=(0.5, 0.0), title="Ground Truth Labels")
    fig1.tight_layout(rect=[0, 0.05, 1, 1]) 
    fig1.savefig(f"plots/{file_prefix}_confusion_grid_{dataset_name}.png", dpi=300, bbox_inches='tight')

    # ==========================================
    # FIGURE 2: The Subclass Grid (Recall Focus)
    # ==========================================
    fig2, axes2 = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows)) 
    if num_classes == 1: axes2 = np.array([axes2])
    axes2 = axes2.flatten()
    
    # Loop over the ACTUAL classes
    for idx, c in enumerate(classes):
        ax = axes2[idx]
        
        # Calculate Recall for this actual class
        subset_actual = df[df['True_Label'] == c]
        total_actual = len(subset_actual)
        correct_actual = len(subset_actual[subset_actual['Predicted_Label'] == c])
        recall = (correct_actual / total_actual * 100) if total_actual > 0 else 0.0

        # Background
        sns.scatterplot(x='dim_1', y='dim_2', data=df[df['True_Label'] != c], 
                        color='lightgrey', alpha=0.3, s=10, linewidth=0, ax=ax, legend=False)
        # Foreground (Colored by Predicted Label)
        sns.scatterplot(x='dim_1', y='dim_2', hue='Predicted_Label', palette=color_dict, 
                        data=subset_actual, 
                        alpha=0.9, s=25, linewidth=0.5, edgecolor='black', ax=ax, legend=False)
        
        ax.set_title(f"Actual Class: {c}\n(Predicted correctly: {recall:.1f}%)", fontweight='bold')
        ax.axis('off')

    for i in range(num_classes, len(axes2)): axes2[i].axis('off')
    
    # Global Legend for Fig 2
    handles2 = [mpatches.Patch(color=color_dict[c], label=f"Predicted Class {c}") for c in classes]
    fig2.legend(handles=handles2, loc='lower center', ncol=cols, bbox_to_anchor=(0.5, 0.0), title="Predicted Labels")
    fig2.tight_layout(rect=[0, 0.05, 1, 1])
    fig2.savefig(f"plots/{file_prefix}_subclass_grid_{dataset_name}.png", dpi=300, bbox_inches='tight')

    # ==========================================
    # FIGURE 3: Confusion Matrix
    # ==========================================
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax3)
    ax3.set_xlabel('Predicted Class', fontweight='bold')
    ax3.set_ylabel('Actual Class', fontweight='bold')
    ax3.set_title(f'Confusion Matrix: {dataset_name}', fontweight='bold')
    fig3.tight_layout()
    fig3.savefig(f"plots/confusion_matrix_{dataset_name}.png", dpi=300, bbox_inches='tight')
    
    import wandb
    try:
        wandb.log({
            f"Analysis/{method_str}_Confusion_Grid": wandb.Image(fig1),
            f"Analysis/{method_str}_Subclass_Grid": wandb.Image(fig2),
            "Analysis/Confusion_Matrix": wandb.Image(fig3)
        })
    except Exception:
        pass
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

# =========================================================
# 2. INTRINSIC & THEORETICAL METRICS
# =========================================================
def compute_and_log_dirichlet_energy(model, Z, X, final_maps, edge_index):
    """
    Calculates and logs the Normalized Dirichlet Energy 
    for the Raw Features, the Identity Sheaf (GCN), and the Learned Sheaf.
    """
    print("\n--- Computing Dirichlet Energies (Task 1) ---")
    N = Z.size(0)
    
    # Safely extract stalk dimension 'd'
    if hasattr(model.encoder, 'd'):
        d = model.encoder.d
    elif hasattr(model.encoder, 'final_d'):
        d = model.encoder.final_d
    else:
        d = 1
        
    def calc_energy(features, L_idx, L_val, matrix_size):
        L_Z = torch_sparse.spmm(L_idx, L_val, matrix_size, matrix_size, features)
        dirichlet_energy = torch.sum(features * L_Z).item()
        norm_squared = torch.norm(features, p='fro').item() ** 2
        return dirichlet_energy / (norm_squared + 1e-8)

    with torch.no_grad():
        # 1. Identity Sheaf Energy (Standard GCN Laplacian)
        L_id_idx, L_id_val = get_laplacian(edge_index, normalization='sym', num_nodes=N)
        identity_energy = calc_energy(Z, L_id_idx, L_id_val, N)

        # 2. Raw Feature Energy
        raw_feat_energy = calc_energy(X, L_id_idx, L_id_val, N)

        # 3. Learned Sheaf Energy (Only if maps exist)
        if final_maps is not None and hasattr(model.encoder, 'laplacian_builder'):
            Z_stalk = Z.view(N * d, -1)
            L_learned_output = model.encoder.laplacian_builder(final_maps)
            L_learned_idx, L_learned_val = L_learned_output[0]
            learned_energy = calc_energy(Z_stalk, L_learned_idx, L_learned_val, N * d)
        else:
            # For GCN/MLP, the "learned" geometry is just the identity geometry
            learned_energy = identity_energy 

    print(f"Normalized Energy (Learned Sheaf):  {learned_energy:.4f}")
    print(f"Normalized Energy (Identity Sheaf): {identity_energy:.4f}")
    print(f"Normalized Energy (Raw Features):   {raw_feat_energy:.4f}")

    # Log to Weights & Biases
    import wandb
    try:
        wandb.log({
            "Task1_Energy/Learned_Sheaf": learned_energy,
            "Task1_Energy/Identity_Sheaf": identity_energy,
            "Task1_Energy/Raw_Features": raw_feat_energy,
        })
    except Exception:
        pass

def verify_lemma1_lipschitz(model, embeddings, edge_index, dataset_name="Graph"):
    """
    Empirically verifies Lemma 1 by ensuring the empirical Lipschitz ratio 
    between Sinkhorn codes (Q) and embeddings (z) is bounded by the theoretical limit.
    """
    print("\n--- Empirically Verifying Lemma 1 (Lipschitz Continuity) ---")
    # NOTE: let's do it for all edges instead of max
    
    with torch.no_grad():
        # 1. Get projected embeddings (z)
        Z = model.forward_projection(embeddings)
        
        # 2. Get prototype matrix (C) and compute Operator Norm (L2 Spectral Norm)
        # NOTE: should this be normalized before computing the operator norm?
        # NOTE: is this why the theoretical bound too high?
        C_raw = model.prototypes.prototypes[0].weight
        C_norm = F.normalize(C_raw, p=2, dim=1)
        C_op_norm = torch.linalg.norm(C_norm, ord=2).item()
        
        # 3. Compute Theoretical Lipschitz Bound
        eps = model.criterion.eps
        L_theoretical = C_op_norm / (2 * eps)
        
        # 4. Get Sinkhorn codes (Q)
        # NOTE: to get this, we are first normalizing C and then computing scores in prototypes forward
        scores = model.prototypes(Z)[0] 
        Q = model.criterion.sinkhorn_knopp(scores) # <--- CHANGED FROM swav_loss
        
        # 5. Compute distances across all edges
        src, tgt = edge_index
        
        diff_Q = torch.norm(Q[src] - Q[tgt], p=2, dim=1)
        diff_Z = torch.norm(Z[src] - Z[tgt], p=2, dim=1)
        
        # Filter out identical nodes to avoid division by zero
        valid_mask = diff_Z > 1e-7
        diff_Q = diff_Q[valid_mask]
        diff_Z = diff_Z[valid_mask]
        
        empirical_ratios = (diff_Q / diff_Z).cpu().numpy()
        max_empirical = np.max(empirical_ratios)
        
        print(f"Theoretical Lipschitz Bound (L): {L_theoretical:.4f}")
        print(f"Max Empirical Ratio observed:    {max_empirical:.4f}")
        
        if max_empirical <= L_theoretical:
            print(">>> STATUS: LEMMA 1 HOLDS TRUE! <<<")
        else:
            print(">>> STATUS: BOUND VIOLATED! (Check epsilon or projection logic) <<<")

        # 6. Plot the Verification
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 5))
        sns.kdeplot(empirical_ratios, fill=True, color='purple', alpha=0.5, 
                    label=r'Empirical Ratio $\frac{||Q_u - Q_v||_2}{||z_u - z_v||_2}$')
        
        plt.axvline(x=L_theoretical, color='red', linestyle='--', linewidth=2.5, 
                    label=f'Theoretical Bound (L = {L_theoretical:.2f})')
        
        plt.title(f"Empirical Verification of Lemma 1 ({dataset_name})", fontsize=14, fontweight='bold')
        plt.xlabel("Lipschitz Ratio", fontsize=12)
        plt.ylabel("Density of Edges", fontsize=12)
        plt.legend(loc='upper right')
        sns.despine()
        
        plt.tight_layout()
        plt.savefig(f"plots/lemma1_verification_{dataset_name}.png", dpi=300)
        
        try:
            import wandb
            wandb.log({"Analysis/Lemma1_Verification": wandb.Image(plt),
                       "Analysis/Theoretical_Bound": L_theoretical,
                       "Analysis/Max_Empirical_Ratio": max_empirical})
        except Exception:
            pass
        plt.close()

def compute_prototype_dirichlet_energy(model, final_maps, num_nodes):
    """
    Evaluates how 'harmonic' the learned prototypes are by assigning 
    each prototype as a constant global signal across all nodes and 
    computing its Sheaf Dirichlet Energy.
    """
    print("\n--- Computing Prototype Dirichlet Energy ---")

    if final_maps is None or not hasattr(model.encoder, 'laplacian_builder'):
        return 0.0, [] # Or calculate using standard Identity Laplacian

    d = model.encoder.d if hasattr(model.encoder, 'd') else model.encoder.final_d
    
    with torch.no_grad():
        # Get the normalized prototypes [K, d]
        C = model.prototypes.prototypes[0].weight.data
        C_norm = F.normalize(C, p=2, dim=1)
        K = C_norm.size(0)
        
        # Build the learned Sheaf Laplacian
        L_output = model.encoder.laplacian_builder(final_maps)
        L_idx, L_val = L_output[0]
        
        prototype_energies = []
        for k in range(K):
            # Project prototype k onto all nodes [N, d]
            # c_k = C_norm[k].unsqueeze(0).expand(num_nodes, d)
            # c_k = C_norm[k].unsqueeze(0).expand(num_nodes, -1)
            c_k = C_norm[k].view(1, -1, d).expand(num_nodes, -1, -1)
            c_k_flat = c_k.reshape(num_nodes * d, 1)
            
            # Compute energy: c_k^T * L * c_k
            L_c_k = torch_sparse.spmm(L_idx, L_val, num_nodes * d, num_nodes * d, c_k_flat)
            energy = torch.sum(c_k_flat * L_c_k).item()
            
            # Normalize by the norm squared of the constant signal
            norm_sq = torch.norm(c_k_flat, p='fro').item() ** 2
            prototype_energies.append(energy / (norm_sq + 1e-8))
            
        mean_energy = np.mean(prototype_energies)
        print(f"Mean Prototype Dirichlet Energy: {mean_energy:.6f}")
        return mean_energy, prototype_energies

def compute_edge_asymmetry(maps, edge_index, labels):
    """
    Computes || F_{v<-e} - F_{u<-e} ||_F for each edge.
    This measures the geometric ASYMMETRY of the learned transport, 
    comparing how it behaves on homophilic vs heterophilic edges.
    """
    if maps is None: return None, None
    
    with torch.no_grad():
        E = edge_index.shape[1] // 2
        r_src = maps[:E]
        r_tgt = maps[E:]
        
        # Compute norm of the difference between the restriction maps
        if r_src.dim() == 3: # Bundle/General Sheaf
            asymmetry = torch.linalg.matrix_norm(r_src - r_tgt, ord='fro').cpu().numpy()
        else: # Diag Sheaf
            asymmetry = torch.norm(r_src - r_tgt, p=2, dim=1).cpu().numpy()
            
    # Split by edge type
    u, v = edge_index[0, :E], edge_index[1, :E]
    same_label = (labels[u] == labels[v]).cpu().numpy()
    
    homo_asym = asymmetry[same_label]
    hetero_asym = asymmetry[~same_label]
    
    return homo_asym, hetero_asym

def compute_true_edge_disagreement(model, embeddings, maps, edge_index, labels):
    """
    Computes the true Sheaf Disagreement across each edge:
    || F_{v <- e} z_v - F_{u <- e} z_u ||_2
    This represents how much two connected nodes "disagree" in the shared discourse space.
    """
    if maps is None: return None, None

    with torch.no_grad():
        src, tgt = edge_index
        E = src.shape[0] // 2
        N = embeddings.size(0)
        
        src_ = src[:E]
        tgt_ = tgt[:E]
        r_src = maps[:E]
        r_tgt = maps[E:]
        
        # Safely extract stalk dimension 'd' and feature dimension 'f'
        d = model.encoder.d if hasattr(model.encoder, 'd') else model.encoder.final_d
        emb_dim = embeddings.size(1) // d
        
        # Reshape embeddings to [N, d, f]
        Z_reshaped = embeddings.view(N, d, emb_dim)
        
        # Project nodes into the edge discourse space
        if r_src.dim() == 3:
            # Bundle/General Sheaf: BMM [E, d, d] @ [E, d, f] -> [E, d, f]
            s_src = torch.bmm(r_src, Z_reshaped[src_])
            s_tgt = torch.bmm(r_tgt, Z_reshaped[tgt_])
        else:
            # Diag Sheaf: Element-wise multiply [E, d, 1] * [E, d, f] -> [E, d, f]
            s_src = r_src.unsqueeze(-1) * Z_reshaped[src_]
            s_tgt = r_tgt.unsqueeze(-1) * Z_reshaped[tgt_]
            
        # The true disagreement is the norm of the difference in the discourse space
        # Calculate norm over the stalk and feature dimensions
        disagreement = torch.norm(s_src - s_tgt, p=2, dim=(1, 2)).cpu().numpy()
        
    # Split by ground-truth labels
    same_label = (labels[src_] == labels[tgt_]).cpu().numpy()
    
    homo_disagree = disagreement[same_label]
    hetero_disagree = disagreement[~same_label]
    
    return homo_disagree, hetero_disagree

def compute_spectral_metrics_robust(embeddings):
    """
    Computes spectral utilization metrics efficiently for large N.
    """
    Z = embeddings.detach().cpu()
    N, D = Z.shape
    
    # 1. Fast SVD: Computes U (N x D) and S (D), avoiding full N x N matrices
    U, S, Vh = torch.linalg.svd(Z, full_matrices=False)

    # RankMe (Entropy of singular values)
    S_norm = S / torch.sum(S)
    entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-12))
    rankme = torch.exp(entropy).item()

    # Alpha-ReQ (Generalization/Decay rate)
    log_k = np.log(np.arange(1, len(S) + 1))
    log_sigma = np.log(S.numpy() + 1e-12)
    slope, _, _, _, _ = linregress(log_k, log_sigma)
    alpha_req = -slope

    # Stable Rank
    stable_rank = (torch.sum(S ** 2) / (torch.max(S) ** 2)).item()

    # Pseudo-Condition Number (Ratio of largest to smallest singular value)
    cond_num = (S[0] / (S[-1] + 1e-12)).item()

    # NESum (Sum of negative exponentials of singular values)
    nesum = torch.sum(torch.exp(-S)).item()

    # Coherence (Max leverage score scaled by N / rank)
    # Estimate rank as number of singular values > 1e-5
    rank = (S > 1e-5).sum().item()
    rank = max(rank, 1) # Prevent division by zero
    max_row_norm_sq = torch.max(torch.sum(U**2, dim=1)).item()
    coherence = (N / rank) * max_row_norm_sq

    # 2. SelfCluster
    Z_norm = F.normalize(Z, p=2, dim=1)
    gram_feature = Z_norm.T @ Z_norm
    Q_obs = torch.norm(gram_feature, p='fro')**2
    Q_obs = Q_obs.item()
    
    exp_random = N + (N * (N - 1) / D)
    max_val = N**2
    self_cluster = (Q_obs - exp_random) / (max_val - exp_random)

    return {
        "RankMe": rankme,
        "Alpha_ReQ": alpha_req,
        "Stable_Rank": stable_rank,
        "Condition_Number": cond_num,
        "NESum": nesum,
        "Coherence": coherence,
        "SelfCluster": self_cluster
    }

def compute_manifold_metrics_robust(embeddings, y, edge_index):
    """
    Manifold metrics with unit-sphere normalization and deterministic sampling.
    """
    print("Computing manifold utilization")

    # Unit-Sphere Normalization: Essential for fair Dirichlet Energy comparison
    emb_norm_torch = F.normalize(embeddings, p=2, dim=1)
    emb_norm_np = emb_norm_torch.cpu().numpy()
    y_np = y.cpu().numpy()
    
    # 1. Silhouette Score (Deterministic Sampling for N > 10k)
    # Using 'cosine' metric because embeddings are normalized
    if len(emb_norm_np) > 10000:
        rng = np.random.default_rng(42) # Reproducibility
        indices = rng.choice(len(emb_norm_np), 10000, replace=False)
        sil = silhouette_score(emb_norm_np[indices], y_np[indices], metric='cosine')
    else:
        sil = silhouette_score(emb_norm_np, y_np, metric='cosine')

    # 2. Latent KNN Purity (Full Graph)
    nbrs = NearestNeighbors(n_neighbors=11, metric='cosine', n_jobs=-1).fit(emb_norm_np)
    _, indices = nbrs.kneighbors(emb_norm_np)
    y_neighbors = y_np[indices[:, 1:]]
    y_self = y_np.reshape(-1, 1)
    knn_hom = (y_neighbors == y_self).mean()

    # 3. Dirichlet Energy (Normalized Scale)
    src, dst = edge_index
    diff = emb_norm_torch[src] - emb_norm_torch[dst]
    energy = torch.norm(diff, p=2, dim=1).pow(2).mean().item()

    # 4. Distribution Overlap (Bhattacharyya Coefficient)
    cos_sim = F.cosine_similarity(emb_norm_torch[src], emb_norm_torch[dst], dim=1)
    cos_sim_mapped = (cos_sim + 1) / 2 # Scale [-1, 1] -> [0, 1]
    label_same = (y[src] == y[dst])
    
    h_homo, _ = np.histogram(cos_sim_mapped[label_same].cpu().numpy(), bins=100, range=(0,1), density=True)
    h_hetero, _ = np.histogram(cos_sim_mapped[~label_same].cpu().numpy(), bins=100, range=(0,1), density=True)
    overlap = np.sum(np.sqrt(h_homo * h_hetero)) / 100
    
    return {
        "Silhouette": sil,
        "kNN_Purity": knn_hom,
        "Dirichlet_Energy": energy,
        "Overlap": overlap
    }

# =========================================================
# 3. DOWNSTREAM & FINE-TUNING AUDITS
# =========================================================
class FineTuneWrapper(nn.Module):
    """
    Wraps the pre-trained Neural Sheaf Diffusion encoder and a linear classifier 
    for end-to-end fine-tuning.
    """
    def __init__(self, encoder: nn.Module, rep_size: int, num_classes: int):
        super(FineTuneWrapper, self).__init__()
        self.encoder = encoder
        
        # The classification head
        out_dim = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(rep_size, out_dim)
        
        # Initialize head
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x, **kwargs):
        # Extract embeddings just like in swav.py
        out = self.encoder(x, **kwargs)
        embeddings = out["z"] if isinstance(out, dict) else out
        
        # Pass through classification head
        logits = self.classifier(embeddings)
        return logits

def run_finetuning_audit_robust(pretrained_encoder, feat, label, rep_size, num_splits=10, lr=0.005, weight_decay=5e-4, epochs=1000, patience=100):
    """
    Executes the End-to-End Fine-tuning protocol across multiple splits.
    Compares directly to run_downstream_audit_robust (Linear Probe).
    """
    print("\n--- Running End-to-End Fine-Tuning Audit ---")
    device = feat.device
    classes = torch.unique(label).cpu().numpy()
    num_classes = len(classes)
    is_binary = (num_classes == 2)
    
    y = label.to(torch.float if is_binary else torch.long).to(device)
    loss_fn = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    
    results = {'acc': [], 'f1_macro': [], 'roc_auc': [], 'avg_precision': []}

    for split_idx in range(num_splits):
        print(f"Fine-tuning Split {split_idx + 1}/{num_splits}...")
        
        # 1. Generate Split (Stratified 60/20/20 to match standard GCL protocols)
        indices = np.arange(len(label.cpu()))
        idx_train, idx_rest, y_train, y_rest = train_test_split(
            indices, label.cpu().numpy(), train_size=0.6, random_state=42 + split_idx, stratify=label.cpu().numpy()
        )
        idx_val, idx_test = train_test_split(
            idx_rest, test_size=0.5, random_state=42 + split_idx, stratify=y_rest
        )
        
        t_mask = torch.tensor(idx_train, dtype=torch.long, device=device)
        v_mask = torch.tensor(idx_val, dtype=torch.long, device=device)
        te_mask = torch.tensor(idx_test, dtype=torch.long, device=device)

        # 2. Deepcopy the pre-trained encoder so we don't leak fine-tuned weights to the next split
        encoder_clone = copy.deepcopy(pretrained_encoder).to(device)
        model = FineTuneWrapper(encoder_clone, rep_size, num_classes).to(device)
        
        # 3. Setup Optimizer
        # Note: Often in fine-tuning, you apply a smaller LR to the encoder and a larger one to the head
        optimizer = torch.optim.Adam([
            {'params': model.encoder.parameters(), 'lr': lr * 0.1}, # Smaller LR for pre-trained backbone
            {'params': model.classifier.parameters(), 'lr': lr}     # Standard LR for new head
        ], weight_decay=weight_decay)

        best_val_metric = 0.0
        test_at_best_val = {}
        bad_counter = 0

        # 4. Training Loop
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            logits = model(feat)
            loss = loss_fn(logits[t_mask].squeeze(-1), y[t_mask]) if is_binary else loss_fn(logits[t_mask], y[t_mask])
            
            loss.backward()
            optimizer.step()

            # 5. Evaluation Loop
            if epoch % 10 == 0 or epoch == epochs - 1:
                model.eval()
                with torch.no_grad():
                    eval_logits = model(feat)
                    
                    if is_binary:
                        val_probs = torch.sigmoid(eval_logits[v_mask]).squeeze(-1).cpu().numpy()
                        test_probs = torch.sigmoid(eval_logits[te_mask]).squeeze(-1).cpu().numpy()
                        
                        try:
                            val_metric = roc_auc_score(y[v_mask].cpu().numpy(), val_probs)
                            test_auc = roc_auc_score(y[te_mask].cpu().numpy(), test_probs)
                            test_ap = average_precision_score(y[te_mask].cpu().numpy(), test_probs)
                            test_acc = ((test_probs > 0.5) == y[te_mask].cpu().numpy()).mean()
                            test_f1 = f1_score(y[te_mask].cpu().numpy(), (test_probs > 0.5), average='macro')
                        except ValueError:
                            val_metric, test_auc, test_ap, test_acc, test_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
                            
                        current_test_metrics = {'acc': test_acc, 'roc_auc': test_auc, 'avg_precision': test_ap, 'f1_macro': test_f1}
                    else:
                        val_preds = torch.argmax(eval_logits[v_mask], dim=1).cpu().numpy()
                        test_preds = torch.argmax(eval_logits[te_mask], dim=1).cpu().numpy()
                        test_probs = torch.softmax(eval_logits[te_mask], dim=1).cpu().numpy()
                        
                        val_metric = (val_preds == y[v_mask].cpu().numpy()).mean()
                        test_acc = (test_preds == y[te_mask].cpu().numpy()).mean()
                        test_f1 = f1_score(y[te_mask].cpu().numpy(), test_preds, average='macro')
                        
                        y_te_np = y[te_mask].cpu().numpy()
                        try:
                            test_auc = roc_auc_score(y_te_np, test_probs, multi_class='ovr')
                            y_te_bin = label_binarize(y_te_np, classes=classes)
                            test_ap = average_precision_score(y_te_bin, test_probs, average='macro')
                        except ValueError:
                            test_auc, test_ap = 0.0, 0.0
                            
                        current_test_metrics = {'acc': test_acc, 'roc_auc': test_auc, 'avg_precision': test_ap, 'f1_macro': test_f1}

                # Early Stopping Logic
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    test_at_best_val = current_test_metrics
                    bad_counter = 0
                else:
                    bad_counter += 1

                if bad_counter >= patience:
                    break
        
        # Log the best test metrics for this split
        for k, v in test_at_best_val.items():
            results[k].append(v)
            
    # Compute Final Mean & Std
    final_results = {k: (np.mean(v), np.std(v)) for k, v in results.items()}
    
    print("\n--- Fine-Tuning Results ---")
    print(f"Accuracy: {final_results['acc'][0]*100:.2f} ± {final_results['acc'][1]*100:.2f}")
    print(f"ROC-AUC:  {final_results['roc_auc'][0]*100:.2f} ± {final_results['roc_auc'][1]*100:.2f}")
    
    return final_results

def run_downstream_audit_robust(embeddings, y, num_splits=10):
    """
    10-split evaluation including Macro-AP for imbalanced multiclass sets.
    """
    X = F.normalize(embeddings, dim=1).cpu().numpy()
    y = y.cpu().numpy()
    classes = np.unique(y)
    num_classes = len(classes)
    
    results = {'acc': [], 'f1_macro': [], 'roc_auc': [], 'avg_precision': []}
    # param_grid = {'C': [2**i for i in range(-10, 11)]}
    param_grid = {'C': [1.0]}


    for i in range(num_splits):
        print(f"split {i+1}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42+i, stratify=y
        )
        
        # lr = LogisticRegression(solver='liblinear', max_iter=1000)
        # print("performing grid search")
        # clf = GridSearchCV(lr, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        # clf.fit(X_train, y_train)

        # Use balanced class weights to counteract Minesweeper/Tolokers imbalance
        lr = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced', seed=42)
        
        print("performing grid search")
        # Ensure we optimize for AUC on binary tasks, otherwise GridSearch picks a bad 'C'
        scoring_metric = 'roc_auc' if num_classes == 2 else 'accuracy'
        
        clf = GridSearchCV(lr, param_grid, cv=3, scoring=scoring_metric, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        print("found the best estimator")
        y_pred = clf.best_estimator_.predict(X_test)
        y_proba = clf.best_estimator_.predict_proba(X_test)

        results['acc'].append(accuracy_score(y_test, y_pred))
        results['f1_macro'].append(f1_score(y_test, y_pred, average='macro'))
        
        if num_classes == 2:
            results['roc_auc'].append(roc_auc_score(y_test, y_proba[:, 1]))
            results['avg_precision'].append(average_precision_score(y_test, y_proba[:, 1]))
        else:
            results['roc_auc'].append(roc_auc_score(y_test, y_proba, multi_class='ovr'))
            # Multiclass Macro Average Precision
            y_test_bin = label_binarize(y_test, classes=classes)
            results['avg_precision'].append(average_precision_score(y_test_bin, y_proba, average='macro'))

    return {k: (np.mean(v), np.std(v)) for k, v in results.items()}

def run_label_rate_sensitivity(embeddings, y, num_splits=5, is_binary_task=False): # device='cpu'
    """Evaluates the frozen linear probe across varying label rates using the EXACT PyTorch pipeline and split logic."""
    from evaluator import evaluate_mask_list, random_splits
    import wandb
    import torch

    device = embeddings.device
    embeddings = embeddings.to(device)
    eval_label = y.to(torch.float if is_binary_task else torch.long).to(device)

    # Ensure labels are formatted identically to the main linear probe
    eval_label = y.to(torch.float if is_binary_task else torch.long).to(device)
    n_classes = len(torch.unique(eval_label))
    rep_size = embeddings.size(1)
    
    rates = [0.05, 0.10, 0.20, 0.40, 0.60]
    val_rate = 0.20 # Lock validation size to exactly 20%, just like the main pipeline
    
    for rate in rates:
        print(f"Evaluating at {rate*100}% label rate...")
        masks_list = []
        
        for i in range(num_splits):
            # 1. Calculate EXACT PolyGCL class counts based on the current rate
            percls_trn = int(round(rate * len(eval_label) / max(1, n_classes)))
            val_lb = int(round(val_rate * len(eval_label)))
            
            # 2. Generate identical PolyGCL splits using your custom function
            # We offset the seed by `i` to ensure different random splits per iteration
            t_mask, v_mask, te_mask = random_splits(eval_label, n_classes, percls_trn, val_lb, seed=42+i)
            
            masks_list.append((t_mask.to(device), v_mask.to(device), te_mask.to(device)))
            
        # 3. Use the exact same PyTorch evaluator as the main baseline
        mean_score, std_score, _, _, _ = evaluate_mask_list(
            embeddings, eval_label, masks_list, f"Rate_{rate}", rep_size, n_classes, is_binary_task, device
        )
        
        # 4. Log the MEAN correctly as a decimal
        if wandb.run is not None:
            wandb.log({f"Sensitivity/Rate_{rate}": mean_score / 100.0})
            
def compute_and_log_cluster_purity(model, embeddings, label, device):
    """Computes Purity, NMI, ARI, and Silhouette for SwAV prototypes."""
    print("\n--- Final Cluster Purity Analysis ---")
    with torch.no_grad():
        z = model.forward_projection(embeddings)
        proto_scores = model.prototypes(z)[0] 
        
        if hasattr(model, 'get_masks'):
            mask = model.get_masks()[0]
            proto_scores_masked = proto_scores.clone()
            proto_scores_masked[:, ~mask] = -1e9
            cluster_assignments = torch.argmax(proto_scores_masked, dim=1)
            active_clusters = mask.nonzero(as_tuple=True)[0]
        else:
            cluster_assignments = torch.argmax(proto_scores, dim=1)
            active_clusters = torch.arange(proto_scores.shape[1], device=device)

        total_assigned, correct_nodes = 0, 0
        cluster_sizes = []
        
        for c in active_clusters:
            c_idx = c.item()
            node_indices = (cluster_assignments == c_idx).nonzero(as_tuple=True)[0]
            cluster_size = len(node_indices)
            cluster_sizes.append(cluster_size)

            if cluster_size == 0:
                print(f"Cluster {c_idx:>3}: Size = {cluster_size:>4} | Empty")
                continue
            
            cluster_labels = label[node_indices]
            unique_labels, counts = torch.unique(cluster_labels, return_counts=True)
            max_count_idx = torch.argmax(counts)
            majority_class = unique_labels[max_count_idx].item()
            
            max_count = counts.max().item()
            correct_nodes += max_count
            total_assigned += cluster_size

            cluster_purity = max_count / cluster_size
            print(f"Cluster {c_idx:>3}: Size = {cluster_size:>4} | Majority Class = {majority_class:>2} | Purity = {cluster_purity:.4f}")
            
        if total_assigned > 0:
            overall_purity = correct_nodes / total_assigned
            y_true, y_pred = label.cpu().numpy(), cluster_assignments.cpu().numpy()
            nmi_score = normalized_mutual_info_score(y_true, y_pred)
            ari_score = adjusted_rand_score(y_true, y_pred)
            
            sil_score = 0.0
            if len(np.unique(y_pred)) > 1:
                sil_score = silhouette_score(z.cpu().numpy(), y_pred, metric='cosine')
            else:
                print("\n[!] WARNING: Severe Representation Collapse Detected! All nodes in 1 cluster.")
            
            print(f"Weighted Purity: {overall_purity:.4f}")
            print(f"NMI:             {nmi_score:.4f}")
            print(f"ARI:             {ari_score:.4f}")
            print(f"Silhouette:      {sil_score:.4f} (Cosine)")
            print(f"Active Clusters: {len(active_clusters)}")

            import wandb
            if wandb.run is not None:
                wandb.log({
                    "Final_Manifold/Cluster_Purity": overall_purity,
                    "Final_Manifold/Cluster_NMI": nmi_score,
                    "Final_Manifold/Cluster_ARI": ari_score,
                    "Final_Manifold/Cluster_Silhouette": sil_score,
                    "Final_Manifold/Cluster_Size_Min": min(cluster_sizes),
                    "Final_Manifold/Cluster_Size_Max": max(cluster_sizes),
                    "Final_Manifold/Cluster_Size_Avg": total_assigned / len(active_clusters),
                    "Final_Manifold/Active_Clusters": len(active_clusters)
                })
                
    return cluster_assignments
# =========================================================
# 4. MASTER EXECUTION WRAPPER
# =========================================================
def run_post_training_audits(model, embeds_torch, final_maps, label, eval_label, edge_index, full_preds, dataset_name, args):
    """Master function to execute all visualizations and intrinsic audits after training."""
    
    # 1. Base Intrinsic Metrics (Always run unless it's a fast sweep)
    # We can just assume if we called this function, we want the basics.
    print("\n--- Running Intrinsic Embedding Audit ---")
    spectral_metrics = compute_spectral_metrics_robust(embeds_torch)
    manifold_metrics = compute_manifold_metrics_robust(embeds_torch, label, edge_index)
    if wandb.run is not None:
        wandb.log({f"Final_Spectral/{k}": v for k, v in spectral_metrics.items()})
        wandb.log({f"Final_Manifold/{k}": v for k, v in manifold_metrics.items()})

    # 2. Cluster Purity (Only for SwAV)
    cluster_assignments = None
    if args.loss_type == 'swav':
        cluster_assignments = compute_and_log_cluster_purity(model, embeds_torch, label, args.device)

    # 3. Visualizations
    if getattr(args, 'run_visuals', False):
        print("\n--- Generating Visualizations ---")
        plot_tsne(embeds_torch, label, dataset_name=dataset_name)
        analyze_and_plot_sheaf_transport(edge_index, final_maps, label, dataset_name=dataset_name)
        if args.loss_type == 'swav' and full_preds is not None and cluster_assignments is not None:
            plot_semantic_grids(embeds_torch, eval_label, full_preds, cluster_assignments, dataset_name=dataset_name)

    # 4. Theoretical Audits (The Math proofs)
    if getattr(args, 'run_theory_audits', False):
        print("\n--- Running Theoretical Audits ---")
        compute_and_log_dirichlet_energy(model, embeds_torch, model.encoder_input_features if hasattr(model, 'encoder_input_features') else None, final_maps, edge_index)
        
        # ---> NEW: Run Edge/Geometry Diagnostics
        homo_asym, hetero_asym = compute_edge_asymmetry(final_maps, edge_index, label)
        homo_disagree, hetero_disagree = compute_true_edge_disagreement(model, embeds_torch, final_maps, edge_index, label)
        
        if wandb.run is not None and homo_asym is not None:
            wandb.log({
                "Theory/Edge_Asymmetry_Homo": np.mean(homo_asym),
                "Theory/Edge_Asymmetry_Hetero": np.mean(hetero_asym),
                "Theory/Sheaf_Disagreement_Homo": np.mean(homo_disagree),
                "Theory/Sheaf_Disagreement_Hetero": np.mean(hetero_disagree)
            })
        
        if full_preds is not None:
            plot_local_section_error(model, embeds_torch, edge_index, final_maps, eval_label, full_preds, dataset_name=dataset_name)
            
        if args.loss_type == 'swav':
            audit_intercluster_transport(edge_index, final_maps, cluster_assignments)
            verify_lemma1_lipschitz(model, embeds_torch, edge_index, dataset_name=dataset_name)
            
            # ---> NEW: Run Prototype Energy
            num_nodes = embeds_torch.size(0)
            # compute_prototype_dirichlet_energy(model, final_maps, num_nodes)

    # 5. Expensive Downstream Evals
    if getattr(args, 'run_expensive_evals', False):
        if hasattr(model, 'encoder_input_features'):
            rep_size = embeds_torch.size(1)
            finetune_results = run_finetuning_audit_robust(
                model.encoder, model.encoder_input_features, label, rep_size, num_splits=10, epochs=500 
            )
            if wandb.run is not None:
                wandb.log({
                    "Eval/Finetune_Acc_Mean": finetune_results['acc'][0] * 100,
                    "Eval/Finetune_Acc_Std": finetune_results['acc'][1] * 100,
                })

        print("\n--- Running Label Rate Sensitivity ---")
        run_label_rate_sensitivity(embeds_torch, eval_label, num_splits=5)
        
    print("\n>>> All Post-Training Audits Completed! <<<")