GPT_CKPT="./ckpts/cond_gpt_ucf_128_488_29999__epoch=21-step=1349999-train.ckpt"
VQGAN_CKPT="./ckpts/vqgan_ucf_128_488_epoch=1-step=29999.ckpt"
DATAPATH="./data/UCF101"
SAVEPATH="./output/ucf_short_videos"
DATANAME="ucf101"

CUDA_VISIBLE_DEVICES=3 python -m scripts.sample_vqgan_transformer_short_videos \
    --gpt_ckpt ${GPT_CKPT} --vqgan_ckpt ${VQGAN_CKPT} --class_cond \
    --save ${SAVEPATH} --data_path ${DATAPATH} --batch_size 16 --resolution 128 \
    --top_k 2048 --top_p 0.8 --dataset ${DATANAME} --compute_fvd --save_videos
