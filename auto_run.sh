CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  \
    --nproc_per_node 4   \
    --master_port 5558   \
    -m train   \
    --from_pretrained result/FGvln_1phase/data/29.bin \
    --save_name FGvln_baseline_4  \
    --shuffle_visual_features   \
    --ranking   \
    --batch_size 8    \
    --num_epochs 10 \
    --mini \
    --Full

echo "Baseline 4 Done..."


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  \
    --nproc_per_node 4   \
    --master_port 5558   \
    -m train   \
    --from_pretrained result/FGvln_1phase/data/29.bin \
    --save_name FGvln_3e5_2FG_5it_1_seed   \
    --shuffle_visual_features   \
    --ranking   \
    --batch_size 8    \
    --num_epochs 10 \
    --FGN \
    --trial_type 1 \
    --num_FGN 2 \
    --trial_iter 8 \
    --FG_style 1 \
    --mini \
    --Full

echo "FGvln_3e5_2FG_5it_mini 1 Done..."


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch  \
    --nproc_per_node 8   \
    --master_port 5558   \
    -m train   \
    --from_pretrained result/FGvln_1phase/data/29.bin \
    --save_name FGvln_3e5_2FG_5it_1_seed_full   \
    --shuffle_visual_features   \
    --ranking   \
    --batch_size 16    \
    --num_epochs 30 \
    --FGN \
    --trial_type 1 \
    --num_FGN 2 \
    --trial_iter 5 \
    --FG_style 1 \
    --Full
