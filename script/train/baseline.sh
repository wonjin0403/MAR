MAR_PATH="./src/output/baseline"
FOLD_PATH="/app/MAR/data/123_fold.json"

python /app/Final_MAR/src/core/train/main.py \
save_path="$MAR_PATH" \
Trainer.devices=\"1,2\" \
module.batch_size=8 \
module.criterion.device="cuda:1" \
module.dataset.train.fold_path="$FOLD_PATH" \
module.dataset.validation.fold_path="$FOLD_PATH" \
module.dataset.test.fold_path="$FOLD_PATH" \
module.test_save_path="$MAR_PATH"