MAR_PATH="./src/output/masked_mse"
FOLD_PATH="/app/home/jhk22/MAR/HN-CT-MAR/codes/final_all_data_1_fold.json"
MODEL_PATH="'./src/output/masked_mse/lightning_logs/version_0/checkpoints/epoch=82-step=255640.ckpt'"

python ./src/core/predict/main.py \
save_path="$MAR_PATH" \
Trainer.devices=\"0,\" \
module=masked_mse \
module.batch_size=1 \
module.save_path="$MODEL_PATH" \
module.criterion.device="cuda:0" \
module.dataset.train.fold_path="$FOLD_PATH" \
module.dataset.validation.fold_path="$FOLD_PATH" \
module.dataset.test.fold_path="$FOLD_PATH" \
module.test_save_path="$MAR_PATH" 