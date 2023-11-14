find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'

python train.py --model_name mono_model --data_path /mnt/test/kitti_data --log_dir models --split eigen_zhou --batch_size 12 --num_epochs 20


# check disk usage
df -T

# check devices