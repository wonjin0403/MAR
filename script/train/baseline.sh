MAR_PATH="./src/output/baseline"
FOLD_PATH="/app/home/jhk22/MAR/HN-CT-MAR/codes/final_all_data_1_fold.json"
MODEL_PATH="'/app/data2/jhk22/MAR/src/output/baseline/lightning_logs/version_0/checkpoints/epoch=18-step=58520.ckpt'"

python /app/data2/jhk22/MAR/src/core/train/main.py \
save_path="$MAR_PATH" \
Trainer.devices=\"2,3\" \
module.batch_size=8 \
module.save_path="$MODEL_PATH" \
module.criterion.device="cuda:2" \
module.dataset.train.fold_path="$FOLD_PATH" \
module.dataset.validation.fold_path="$FOLD_PATH" \
module.dataset.test.fold_path="$FOLD_PATH" \
module.test_save_path="$MAR_PATH"

