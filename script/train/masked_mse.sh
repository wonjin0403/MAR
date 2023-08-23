MAR_PATH="./src/output/masked_mse"
FOLD_PATH="/app/MAR/data/123_fold.json"

python src/core/train/main.py \
save_path="$MAR_PATH" \
Trainer.devices=\"6,7\" \
module=masked_mse \
module.batch_size=8 \
module.criterion.device="cuda:6" \
module.dataset.train.fold_path="$FOLD_PATH" \
module.dataset.validation.fold_path="$FOLD_PATH" \
module.dataset.test.fold_path="$FOLD_PATH" \
module.test_save_path="$MAR_PATH" 