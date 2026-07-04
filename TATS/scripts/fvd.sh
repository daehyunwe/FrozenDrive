CUDA_VISIBLE_DEVICES=1 python -m scripts.fvd \
--splits test \
--nusc_dataroot PATH_TO_NUSCENES \
--arena_dataroot /path/to/your/log/.../frames_split_interp_group \
--output_dir /path/to/your/outdir \
--batch_size 1 \
--max_frames 16 \
--num_videos_per_scene 30 \
--shift_stride 1 \
--pad \
--load_real_embeddings_from ./output/shift30_1_pad/val_real_embeddings_max16.npz \
# --load_fake_embeddings_from ./output/shift30_1_pad/interp/fake_embeddings_max16.npz \
# --crop \
# --cameras_to_evaluate \
