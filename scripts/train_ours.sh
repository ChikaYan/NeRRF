CUDA_LAUNCH_BLOCKING=5 \
python train/train.py \
-n ficus_test \
-c NeRRF.conf \
-D /home/tw554/NeRRF/data_ours/ficus \
--gpu_id=0 \
--visual_path tet_visual \
--stage 1 \
--tet_scale 3.8 \
--sphere_radius 2.40 \
--resume \
--enable_refl \
--ior 1.2

# --use_cone
# --use_progressive_encoder
# --use_grid \