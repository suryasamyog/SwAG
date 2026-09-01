#!/bin/bash
export WANDB_PROJECT="SwAG-Core-Ablations-Final"
export HIP_VISIBLE_DEVICES=5

# Create a logs directory if it doesn't exist
mkdir -p console_logs

# DATASETS=("cora" "roman-empire" "amazon-ratings")
DATASETS=("cora" "citeseer" "pubmed" "roman-empire" "amazon-ratings" "tolokers" "questions" "amazon-photo" "amazon-computer" "coauthor-cs" "minesweeper" "chameleon-filtered" "squirrel-filtered")

echo "Starting Table 2 Ablations on GPU 6..."
for DATASET in "${DATASETS[@]}"; do
    # 1. Feature Contrast (No Clusters)
    python3 sweep_train.py --run_best --dataset $DATASET --loss_type ntxent > console_logs/${DATASET}_ablation_ntxent.log 2>&1

    # 2. No Sinkhorn (Softmax only)
    python3 sweep_train.py --run_best --dataset $DATASET --disable_sinkhorn > console_logs/${DATASET}_ablation_no_sinkhorn.log 2>&1

    # 3. Asymmetric (v2 predicts v1)
    python3 sweep_train.py --run_best --dataset $DATASET --asymmetric_mode v2_predicts_v1 > console_logs/${DATASET}_ablation_asym.log 2>&1

    # 4. Asymmetric + No Sinkhorn
    python3 sweep_train.py --run_best --dataset $DATASET --asymmetric_mode v2_predicts_v1 --disable_sinkhorn True > console_logs/${DATASET}_ablation_asym_no_sinkhorn.log 2>&1

    # 5. Augmentation Before Encoding (Edge Drop / Feature Mask)
    python3 sweep_train.py --run_best --dataset $DATASET --view_strategy pre_feature_drop --drop_prob 0.2 > console_logs/${DATASET}_ablation_pre_drop.log 2>&1
    
    # 6. True Laplacian Diagonal Split
    python3 sweep_train.py --run_best --dataset $DATASET --view_strategy post_sheaf_laplacian > console_logs/${DATASET}_ablation_laplacian.log 2>&1

    # 7. Asymmetric (v1 predicts v2) - Completeness for Appendix
    python3 sweep_train.py --run_best --dataset $DATASET --asymmetric_mode v1_predicts_v2 > console_logs/${DATASET}_ablation_asym_v1_to_v2.log 2>&1

    # 8. Pre-Encoder Edge Drop Ablation
    python3 sweep_train.py --run_best --dataset $DATASET --view_strategy pre_edge_drop --drop_prob 0.2 > console_logs/${DATASET}_ablation_pre_edge_drop.log 2>&1

    python3 sweep_train.py --run_best --dataset $DATASET --use_proj False > console_logs/${DATASET}_ablation_drop_projector.log 2>&1

    python3 sweep_train.py --run_best --dataset $DATASET --wide_feature_ablation True > console_logs/${DATASET}_ablation_wide_feature.log 2>&1

done
echo "Core Ablations Completed!"