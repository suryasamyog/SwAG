import torch
import torch.nn as nn
import numpy as np
import wandb
from sklearn.metrics import roc_auc_score
from model import LogReg # Assuming LogReg is still in model.py

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def random_splits(label, num_classes, percls_trn, val_lb, seed=42):
    """Original PolyGCL Split Logic"""
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

# def evaluate_mask_list(embeds_torch, eval_label, masks_list, split_name, rep_size, n_classes, is_binary_task, device, eval_lr=0.01, eval_weight_decay=0.0, eval_steps=2000, eval_patience=50):
# 
#     # --- THE FIX: Force the device parameter to match the embeddings ---
#     device = embeds_torch.device
# 
#     results = []
#     results_val = [] # <--- RESTORED: Track validation scores
#     
#     out_dim = 1 if is_binary_task else n_classes
#     loss_fn = nn.BCEWithLogitsLoss() if is_binary_task else nn.CrossEntropyLoss()
#     
#     # We will store the full graph predictions from the best performing seed model
#     best_overall_full_preds = None
#     best_overall_score = -1.0
# 
#     for idx, (t_mask, v_mask, te_mask) in enumerate(masks_list):
#         train_embs = embeds_torch[t_mask].to(device)
#         val_embs   = embeds_torch[v_mask].to(device)
#         test_embs  = embeds_torch[te_mask].to(device)
# 
#         train_labels = eval_label[t_mask].to(device)
#         val_labels   = eval_label[v_mask].to(device)
#         test_labels  = eval_label[te_mask].to(device)
# 
#         logreg = LogReg(rep_size, out_dim).to(device)
#         opt = torch.optim.Adam(logreg.parameters(), lr=eval_lr, weight_decay=eval_weight_decay) 
# 
#         best_val_metric, test_at_best_val = 0, 0
#         eval_bad_counter = 0
#         best_state = None
# 
#         for step in range(eval_steps):
#             logreg.train()
#             opt.zero_grad()
#             logits = logreg(train_embs)
#             
#             loss = loss_fn(logits.squeeze(-1), train_labels) if is_binary_task else loss_fn(logits, train_labels)
#             loss.backward()
#             opt.step()
# 
#             if step % 10 == 0 or step == 1999:
#                 logreg.eval()
#                 with torch.no_grad():
#                     val_logits, test_logits = logreg(val_embs), logreg(test_embs)
#                     
#                     if is_binary_task:
#                         try:
#                             val_metric = roc_auc_score(val_labels.cpu().numpy(), torch.sigmoid(val_logits).squeeze(-1).cpu().numpy())
#                             test_metric = roc_auc_score(test_labels.cpu().numpy(), torch.sigmoid(test_logits).squeeze(-1).cpu().numpy())
#                         except ValueError:
#                             val_metric, test_metric = 0.0, 0.0 
#                     else: 
#                         val_preds, test_preds = torch.argmax(val_logits, dim=1), torch.argmax(test_logits, dim=1)
#                         val_metric = (val_preds == val_labels).float().mean().item()
#                         test_metric = (test_preds == test_labels).float().mean().item()
# 
#                 if val_metric > best_val_metric:
#                     best_val_metric = val_metric
#                     test_at_best_val = test_metric
#                     eval_bad_counter = 0
# 
#                     best_state = {
#                         k: v.detach().cpu().clone()
#                         for k, v in logreg.state_dict().items()
#                     }
#                 else:
#                     eval_bad_counter += 1
# 
#             if eval_bad_counter >= eval_patience:
#                 break
# 
#         if best_state is not None:
#             logreg.load_state_dict(
#                 {k: v.to(device) for k, v in best_state.items()}
#             )
# 
#         print(f"[{split_name}] Split {idx+1}/10 | Val: {best_val_metric:.4f} | Test: {test_at_best_val:.4f}")
#         results.append(test_at_best_val)
#         results_val.append(best_val_metric) # <--- RESTORED: Save val metric
#         
#         # Keep track of the full predictions of the best overall model for later visualizations
#         if test_at_best_val > best_overall_score:
#             best_overall_score = test_at_best_val
#             logreg.eval()
#             with torch.no_grad():
#                 full_logits = logreg(embeds_torch)
#                 best_overall_full_preds = (torch.sigmoid(full_logits).squeeze(-1) > 0.5).float() if is_binary_task else torch.argmax(full_logits, dim=1)
# 
#     mean_score, std_score = np.mean(results) * 100, np.std(results) * 100
#     mean_val, std_val = np.mean(results_val) * 100, np.std(results_val) * 100 # <--- RESTORED
#     
#     print(f"[{split_name}] Final Val = {mean_val:.2f}% | Final Test = {mean_score:.2f}%\n")
#     return mean_score, std_score, mean_val, std_val, best_overall_full_preds # <--- RESTORED: returning val metrics

def evaluate_mask_list(
    embeds_torch,
    eval_label,
    masks_list,
    split_name,
    rep_size,
    n_classes,
    is_binary_task,
    device,
    eval_lr=0.01,
    eval_weight_decay=0.0,
    eval_steps=2000,
    eval_patience=50,
):
    device = embeds_torch.device

    results = []
    results_val = []

    out_dim = 1 if is_binary_task else n_classes
    loss_fn = nn.BCEWithLogitsLoss() if is_binary_task else nn.CrossEntropyLoss()

    best_overall_full_preds = None
    best_overall_score = -1.0

    for idx, (t_mask, v_mask, te_mask) in enumerate(masks_list):
        train_embs = embeds_torch[t_mask].to(device)
        val_embs = embeds_torch[v_mask].to(device)
        test_embs = embeds_torch[te_mask].to(device)

        train_labels = eval_label[t_mask].to(device)
        val_labels = eval_label[v_mask].to(device)
        test_labels = eval_label[te_mask].to(device)

        logreg = LogReg(rep_size, out_dim).to(device)
        opt = torch.optim.Adam(
            logreg.parameters(),
            lr=eval_lr,
            weight_decay=eval_weight_decay,
        )

        best_val_metric, test_at_best_val = 0.0, 0.0
        eval_bad_counter = 0
        best_state = None

        for step in range(eval_steps):
            logreg.train()
            opt.zero_grad()

            logits = logreg(train_embs)
            loss = (
                loss_fn(logits.squeeze(-1), train_labels)
                if is_binary_task
                else loss_fn(logits, train_labels)
            )

            loss.backward()
            opt.step()

            if step % 10 == 0 or step == eval_steps - 1:
                logreg.eval()
                with torch.no_grad():
                    val_logits = logreg(val_embs)
                    test_logits = logreg(test_embs)

                    if is_binary_task:
                        try:
                            val_metric = roc_auc_score(
                                val_labels.cpu().numpy(),
                                torch.sigmoid(val_logits).squeeze(-1).cpu().numpy(),
                            )
                            test_metric = roc_auc_score(
                                test_labels.cpu().numpy(),
                                torch.sigmoid(test_logits).squeeze(-1).cpu().numpy(),
                            )
                        except ValueError:
                            val_metric, test_metric = 0.0, 0.0
                    else:
                        val_preds = torch.argmax(val_logits, dim=1)
                        test_preds = torch.argmax(test_logits, dim=1)

                        val_metric = (val_preds == val_labels).float().mean().item()
                        test_metric = (test_preds == test_labels).float().mean().item()

                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    test_at_best_val = test_metric
                    eval_bad_counter = 0
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in logreg.state_dict().items()
                    }
                else:
                    eval_bad_counter += 1

            if eval_bad_counter >= eval_patience:
                break

        if best_state is not None:
            logreg.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        print(
            f"[{split_name}] Split {idx + 1}/{len(masks_list)} | "
            f"Val: {best_val_metric:.4f} | Test: {test_at_best_val:.4f}"
        )

        results.append(test_at_best_val)
        results_val.append(best_val_metric)

        if test_at_best_val > best_overall_score:
            best_overall_score = test_at_best_val
            logreg.eval()
            with torch.no_grad():
                full_logits = logreg(embeds_torch)
                if is_binary_task:
                    best_overall_full_preds = (
                        torch.sigmoid(full_logits).squeeze(-1) > 0.5
                    ).float()
                else:
                    best_overall_full_preds = torch.argmax(full_logits, dim=1)

    results = np.asarray(results, dtype=np.float64)
    results_val = np.asarray(results_val, dtype=np.float64)

    mean_score = results.mean() * 100.0
    std_score = results.std(ddof=1) * 100.0 if len(results) > 1 else 0.0
    ci95_score = 1.96 * std_score / np.sqrt(len(results)) if len(results) > 1 else 0.0

    mean_val = results_val.mean() * 100.0
    std_val = results_val.std(ddof=1) * 100.0 if len(results_val) > 1 else 0.0
    ci95_val = 1.96 * std_val / np.sqrt(len(results_val)) if len(results_val) > 1 else 0.0

    print(
        f"[{split_name}] Final Val = {mean_val:.2f} ± {ci95_val:.2f}% CI95 | "
        f"Final Test = {mean_score:.2f} ± {ci95_score:.2f}% CI95\n"
    )

    return (
        mean_score,
        std_score,
        ci95_score,
        mean_val,
        std_val,
        ci95_val,
        best_overall_full_preds,
    )


def run_linear_eval_protocol(embeds_torch, label, data, n_classes, is_binary_task, rep_size, device, eval_lr=0.01, eval_weight_decay=0.0, eval_steps=2000, eval_patience=50, dataset_name=""):
    """Executes the dual evaluation protocol (computes both if possible)."""
    print(f"\n--- Starting Linear Evaluation for {dataset_name} ---")
    
    # ==========================================================
    # 1. cSBM OVERRIDE (Accuracy instead of ROC-AUC)
    # ==========================================================
    if dataset_name.lower().startswith("csbm"):
        print("[*] cSBM detected: Overriding binary task flag to compute Accuracy instead of ROC-AUC.")
        is_binary_task = False 
    
    eval_label = label.to(torch.float if is_binary_task else torch.long).to(device)
    
    pub_mean, pub_std, poly_mean, poly_std = None, None, None, None
    pub_val_mean, poly_val_mean = None, None
    best_full_preds = None

    # --- 1. Evaluate on PyG Public Splits (If Available) ---
    if hasattr(data, 'train_mask') and data.train_mask is not None and data.train_mask.dim() > 1:
        print("Evaluating on PyG Public Splits...")
        public_masks_gpu = [
            (data.train_mask[:, i].to(device), data.val_mask[:, i].to(device), data.test_mask[:, i].to(device))
            for i in range(data.train_mask.shape[1])
        ]
        
#         pub_mean, pub_std, pub_val_mean, pub_val_std, best_full_preds = evaluate_mask_list(
#             embeds_torch, eval_label, public_masks_gpu, "Public-Splits", rep_size, n_classes, is_binary_task, device, eval_lr, eval_weight_decay, eval_steps, eval_patience
#         ) 

        pub_mean, pub_std, pub_ci95, pub_val_mean, pub_val_std, pub_val_ci95, best_full_preds = evaluate_mask_list(
            embeds_torch,
            eval_label,
            public_masks_gpu,
            "Public-Splits",
            rep_size,
            n_classes,
            is_binary_task,
            device,
            eval_lr,
            eval_weight_decay,
            eval_steps,
            eval_patience,
        )
        wandb.log({
            "Eval/Public_Score_Mean": pub_mean, 
            "Eval/Public_Score_Std": pub_std,
            "Eval/Public_Score_CI95": pub_ci95,
            "Eval/Public_Val_Mean": pub_val_mean,
            "Eval/Public_Val_Std": pub_val_std,
            "Eval/Public_Val_CI95": pub_val_ci95,
        }) 

    # --- 2. Evaluate on PolyGCL 60-20-20 Splits ---
    platonov_datasets = [
        'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions',
        'chameleon-filtered', 'squirrel-filtered'
    ]
    
    if dataset_name.lower() in platonov_datasets:
        print(f"Skipping PolyGCL 60-20-20 splits for '{dataset_name}' (Platonov/Filtered dataset).")
    else:
        print("Evaluating on PolyGCL 60-20-20 Random Splits...")
        SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 
                 2108550661, 1648766618, 629014539, 3212139042, 2424918363]

        train_rate, val_rate = 0.6, 0.2
        percls_trn = int(round(train_rate * len(label) / max(1, n_classes)))
        val_lb = int(round(val_rate * len(label)))

        poly_masks_gpu = [
            tuple(torch.BoolTensor(m).to(device) for m in random_splits(eval_label, n_classes, percls_trn, val_lb, seed=seed))
            for seed in SEEDS
        ]
        
#         poly_mean, poly_std, poly_val_mean, poly_val_std, poly_preds = evaluate_mask_list(
#             embeds_torch, eval_label, poly_masks_gpu, "PolyGCL-Splits", rep_size, n_classes, is_binary_task, device, eval_lr, eval_weight_decay, eval_steps, eval_patience
#         )

        poly_mean, poly_std, poly_ci95, poly_val_mean, poly_val_std, poly_val_ci95, poly_preds = evaluate_mask_list(
            embeds_torch,
            eval_label,
            poly_masks_gpu,
            "PolyGCL-Splits",
            rep_size,
            n_classes,
            is_binary_task,
            device,
            eval_lr,
            eval_weight_decay,
            eval_steps,
            eval_patience,
        ) 
        wandb.log({
            "Eval/PolyGCL_Score_Mean": poly_mean, 
            "Eval/PolyGCL_Score_Std": poly_std,
            "Eval/PolyGCL_Score_CI95": poly_ci95,
            "Eval/PolyGCL_Val_Mean": poly_val_mean,
            "Eval/PolyGCL_Val_Std": poly_val_std,
            "Eval/PolyGCL_Val_CI95": poly_val_ci95,
        })
        
        # If public splits weren't computed, we use PolyGCL for the full graph predictions
        if pub_mean is None:
            best_full_preds = poly_preds

    # --- 3. SET THE OFFICIAL TEST AND SWEEP TARGETS ---
#     if dataset_name.lower() in platonov_datasets:
#         # Platonov & Filtered datasets strictly optimize and report on fixed public splits
#         final_test_mean = pub_mean
#         final_test_std = pub_std
#         final_val_mean = pub_val_mean
#         best_full_preds = best_full_preds # Already set from public eval
#     else:
#         # Everything else (Cora, cSBM, etc.) strictly optimizes and reports on PolyGCL 60-20-20 splits
#         final_test_mean = poly_mean
#         final_test_std = poly_std
#         final_val_mean = poly_val_mean
#         best_full_preds = poly_preds
        
    if dataset_name.lower() in platonov_datasets:
        final_test_mean = pub_mean
        final_test_std = pub_std
        final_test_ci95 = pub_ci95
        final_val_mean = pub_val_mean
        final_val_std = pub_val_std
        final_val_ci95 = pub_val_ci95
        best_full_preds = best_full_preds
    else:
        final_test_mean = poly_mean
        final_test_std = poly_std
        final_test_ci95 = poly_ci95
        final_val_mean = poly_val_mean
        final_val_std = poly_val_std
        final_val_ci95 = poly_val_ci95
        best_full_preds = poly_preds

#     wandb.log({
#         "Eval/Test_Score_Mean": final_test_mean, 
#         "Eval/Test_Score_Std": final_test_std,
#         "Sweep_Target/Validation_Score": final_val_mean
#     })

    wandb.log({
        "Eval/Test_Score_Mean": final_test_mean,
        "Eval/Test_Score_Std": final_test_std,
        "Eval/Test_Score_CI95": final_test_ci95,
        "Sweep_Target/Validation_Score": final_val_mean,
        "Sweep_Target/Validation_CI95": final_val_ci95,
    })

    return poly_mean, poly_std, pub_mean, pub_std, best_full_preds, eval_label