MAR_PATH="./src/output/baseline"
FOLD_PATH="/app/home/jhk22/MAR/HN-CT-MAR/codes/final_all_data_1_fold.json"
MODEL_PATH="'./src/output/baseline/lightning_logs/version_2/checkpoints/epoch=96-step=298760.ckpt'"

python ./src/core/test/main.py \
save_path="$MAR_PATH" \
Trainer.devices=\"0,\" \
module=baseline \
module.batch_size=1 \
module.save_path="$MODEL_PATH" \
module.criterion.device="cuda:0" \
module.dataset.train.fold_path="$FOLD_PATH" \
module.dataset.validation.fold_path="$FOLD_PATH" \
module.dataset.test.fold_path="$FOLD_PATH" \
module.test_save_path="$MAR_PATH" 