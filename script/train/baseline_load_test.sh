MAR_PATH="./src/output/baseline"
FOLD_PATH="/app/MAR/data/123_fold.json"
MODEL_PATH="'/app/Final_MAR/src/output/baseline/lightning_logs/version_0/checkpoints/epoch=1-step=596.ckpt'"

python /app/Final_MAR/src/core/train/main.py \
save_path="$MAR_PATH" \
Trainer.devices=\"1,2\" \
module.batch_size=8 \
module.save_path="$MODEL_PATH" \
module.criterion.device="cuda:1" \
module.dataset.train.fold_path="$FOLD_PATH" \
module.dataset.validation.fold_path="$FOLD_PATH" \
module.dataset.test.fold_path="$FOLD_PATH" \
module.test_save_path="$MAR_PATH"