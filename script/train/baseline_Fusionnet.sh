MAR_PATH="./src/output/baseline_Fusionnet"
FOLD_PATH="/app/home/jhk22/MAR/HN-CT-MAR/codes/final_all_data_1_fold.json"

python /app/data2/jhk22/MAR/src/core/train/main.py \
save_path="$MAR_PATH" \
Trainer.devices=\"0,1\" \
Trainer.max_epochs=200 \
Trainer.accumulate_grad_batches=1 \
module=baseline_Fusionnet \
module.batch_size=8 \
module.criterion.device="cuda:0" \
module.optimizer.lr=0.001 \
module.dataset.train.fold_path="$FOLD_PATH" \
module.dataset.validation.fold_path="$FOLD_PATH" \
module.dataset.test.fold_path="$FOLD_PATH" \
module.test_save_path="$MAR_PATH" 