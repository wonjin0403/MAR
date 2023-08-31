MAR_PATH="./src/output/baseline"
FOLD_PATH="/app/home/jhk22/MAR/HN-CT-MAR/codes/final_all_data_1_fold.json"

python /app/data2/jhk22/MAR/src/core/inference/main.py \
model_checkpoint_path="$MAR_PATH/lightning_logs/version_0/checkpoints/epoch=63-step=197120.ckpt" \
module.dataset.infer.fold_path="$FOLD_PATH" \
inference_save_path="$MAR_PATH/inference_results"
