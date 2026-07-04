DEVICES=0,1
PORT=29616
NUM_PROCESSES=2
CONFIG_LR="config_nus_224_depth_infl_lr_bal"
CONFIG_HR="config_nus_448_depth_infl_hr_bal"

RUN_ID=$(date +%Y%m%d%H%M%S)${RANDOM}
echo "Generated unique Run ID for this session: ${RUN_ID}"
# OR RESUME FROM LR
# RUN_ID=2025092223140530713
# echo "Resuming with ID ${RUN_ID}"

echo "Step 1: Starting Low-Resolution (LR) Training..."

accelerate launch \
    --mixed_precision fp16 \
    --gpu_ids ${DEVICES} \
    --multi_gpu \
    --main_process_port ${PORT} \
    --num_processes ${NUM_PROCESSES} \
    --num_machines 1 tools/train_lr2hr.py \
    --config-name=${CONFIG_LR} \
    runner=2gpus \
    runner.train_batch_size=2 \
    runner.max_train_steps=150000 \
    runner.checkpointing_steps=10000 \
    runner.validation_steps=5000 \
    runner.save_model_per_step=5000 \
    runner.validation_before_run=true \
    runner.validation_index=[0,4000,15000,23000,32632] \
    +runner.run_id=${RUN_ID} \
    # resume_from_checkpoint=./dreamer-log/SDv1.5_mv_single_ref_... \

INFO_FILE="/tmp/run_info_${RUN_ID}.txt"
LR_OUTPUT_DIR=$(cat "${INFO_FILE}")
# rm "${INFO_FILE}"
LR_CHECKPOINT_PATH=$(cat "${LR_OUTPUT_DIR}/last_checkpoint.txt")

if [ -n "$LR_CHECKPOINT_PATH" ]; then
    echo "LR Training successful. Checkpoint at: $LR_CHECKPOINT_PATH"
    echo ""
    echo "Step 2: Starting High-Resolution (HR) Fine-tuning..."

    accelerate launch \
        --mixed_precision fp16 \
        --gpu_ids ${DEVICES} \
        --multi_gpu \
        --main_process_port ${PORT} \
        --num_processes ${NUM_PROCESSES} \
        --num_machines 1 tools/train_lr2hr.py \
        --config-name=${CONFIG_HR} \
        runner=2gpus \
        runner.train_batch_size=2 \
        runner.max_train_steps=300000 \
        resume_from_checkpoint=$LR_CHECKPOINT_PATH \
        runner.checkpointing_steps=5000 \
        runner.validation_steps=2500 \
        runner.save_model_per_step=2500 \
        runner.validation_before_run=true \
        runner.validation_index=[0,4000,15000,23000,32632] \
        # resume_from_checkpoint=./dreamer-log/SDv1.5_mv_single_ref_... \

else
    echo "LR Training failed. Halting."
    exit 1
fi

echo "All stages complete!"
