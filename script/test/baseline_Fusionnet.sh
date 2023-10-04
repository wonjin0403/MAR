MAR_PATH="./src/output/baseline_Fusionnet"
FOLD_PATH="/app/home/jhk22/MAR/HN-CT-MAR/codes/final_all_data_1_fold.json"
MODEL_PATH="'./src/output/baseline_Fusionnet/lightning_logs/version_9/checkpoints/epoch=92-step=286440.ckpt'"

python ./src/core/test/main.py \
save_path="$MAR_PATH" \
Trainer.devices=\"3,\" \
module=baseline_Fusionnet \
module.batch_size=1 \
module.save_path="$MODEL_PATH" \
module.criterion.device="cuda:3" \
module.dataset.train.fold_path="$FOLD_PATH" \
module.dataset.validation.fold_path="$FOLD_PATH" \
module.dataset.test.fold_path="$FOLD_PATH" \
module.test_save_path="$MAR_PATH" 