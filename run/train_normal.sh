#!/bin/bash
[ -z $1 ] && exit 1

# dataset name device
dataset="${1}"
name="${2}"
gpu_ids="${3}"

model='nerf_atlas_radiance'

dataset_name='custom'
data_root=$dataset

# random_sample='balanced'
random_sample='patch'
random_sample_size=32
sample_num=256

geometry_embedding_dim=64
primitive_type='sphere'
# primitive_type='square'
primitive_count=1
points_per_primitive=2048
texture_decoder_type='texture_view_mlp_mix'
atlasnet_activation='relu'

loss_color_weight=1
loss_bg_weight=1
loss_chamfer_weight=-1
loss_inverse_uv_weight=-1
loss_inverse_selection_weight=-1
loss_inverse_mapping_weight=1
loss_density_weight=-1

# training
batch_size=1

lr=0.0001

checkpoints_dir='./checkpoints/'
resume_checkpoints_dir=$checkpoints_dir

save_iter_freq=25000
niter=500000
niter_decay=0

n_threads=0

train_and_test=1
test_num=1

print_freq=500
test_freq=2500

loss_normal=1
loss_smooth=0.1
freeze_all_except_normal=1
bias=1
scale_uv_weight=1
seed=1337
python3 train.py  \
        --name=$name  \
        --loss_normal=$loss_normal  \
        --loss_smooth=$loss_smooth  \
        --freeze_all_except_normal=$freeze_all_except_normal  \
        --bias=$bias  \
        --seed=$seed  \
        --scale_uv_weight=$scale_uv_weight  \
        --loss_inverse_mapping_weight=$loss_inverse_mapping_weight  \
        --points_per_primitive=$points_per_primitive  \
        --model=$model  \
        --dataset_name=$dataset_name  \
        --data_root=$data_root  \
        --random_sample=$random_sample  \
        --random_sample_size=$random_sample_size  \
        --sample_num=$sample_num  \
        --geometry_embedding_dim=$geometry_embedding_dim  \
        --primitive_type=$primitive_type  \
        --primitive_count=$primitive_count  \
        --texture_decoder_type=$texture_decoder_type  \
        --atlasnet_activation=$atlasnet_activation  \
        --loss_color_weight=$loss_color_weight  \
        --loss_bg_weight=$loss_bg_weight  \
        --loss_chamfer_weight=$loss_chamfer_weight  \
        --loss_inverse_uv_weight=$loss_inverse_uv_weight  \
        --loss_inverse_selection_weight=$loss_inverse_selection_weight  \
        --loss_density_weight=$loss_density_weight  \
        --batch_size=$batch_size  \
        --lr=$lr  \
        --gpu_ids=$gpu_ids  \
        --checkpoints_dir=$checkpoints_dir  \
        --save_iter_freq=$save_iter_freq  \
        --niter=$niter  \
        --niter_decay=$niter_decay  \
        --n_threads=$n_threads  \
        --train_and_test=$train_and_test  \
        --test_num=$test_num  \
        --print_freq=$print_freq  \
        --test_freq=$test_freq  \
        --verbose  \
        --texture_decoder_width=64  \
        --texture_decoder_depth=2,2  \
        --resume_dir=$resume_checkpoints_dir/$name

