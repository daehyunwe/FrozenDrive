DEVICES=0
CONFIG=test_config_nus_448_depth_infl_ref_attn
CHECKPOINT=./dreamer-log/SDv1.5_mv_single_ref_nus/weight-E5-S200000


CUDA_VISIBLE_DEVICES=${DEVICES} python tools/test_occ.py \
    --config-name=${CONFIG} \
    log_root_prefix=./dreamer-log/test \
    resume_from_checkpoint=${CHECKPOINT} \
    runner=default \
    runner.validation_index=all \
    +text_prompt="Rainy weather. Heavy rain. Wet surface." \
    # +e2e_eval_mode=true \  # Save only rgb images
    # +use_first_frame=true \  # Use first frame from nuscenes as reference
