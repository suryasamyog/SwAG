#!/bin/bash

GPU_ID=0


export WANDB_PROJECT="SwAG-Full-Sweeps-V3"


DATASETS=(
    # Heterophilic
    # "roman-empire"
    "amazon-ratings"
    "minesweeper"
    "tolokers"
    "questions"
)

echo "Starting W&B Sweeps for ${#DATASETS[@]} datasets..."
echo "Project: $WANDB_PROJECT"
echo "GPU: $GPU_ID"
echo "======================================================"

for dataset in "${DATASETS[@]}"
do
  echo " "
  echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] Launching Sweep for: $dataset <<<"
  
  # Launch the sweep! (This will run for the 60 iterations defined in your python script)
  HIP_VISIBLE_DEVICES=$GPU_ID python3 sweep_train_swag.py --run_sweep --dataset "$dataset"
  
  echo ">>> Finished Sweep for $dataset"
  echo "------------------------------------------------------"
done

echo " "
echo "🎉 ALL SWEEPS COMPLETED SUCCESSFULLY! 🎉"