CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  \
    --nproc_per_node 4   \
    --master_port 5558   \
    -m train   \
    --from_pretrained result/FGvln_1phase/data/29.bin \
    --save_name FGvln_baseline_1  \
    --shuffle_visual_features   \
    --ranking   \
    --batch_size 8    \
    --num_epochs 10 \
    --mini

echo "Baseline 1 Done..."


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  \
    --nproc_per_node 4   \
    --master_port 5558   \
    -m train   \
    --from_pretrained result/FGvln_1phase/data/29.bin \
    --save_name FGvln_baseline_2  \
    --shuffle_visual_features   \
    --ranking   \
    --batch_size 8    \
    --num_epochs 10 \
    --mini

echo "Baseline 2 Done..."

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  \
    --nproc_per_node 4   \
    --master_port 5558   \
    -m train   \
    --from_pretrained result/FGvln_1phase/data/29.bin \
    --save_name FGvln_3e5_2FG_5it_1   \
    --shuffle_visual_features   \
    --ranking   \
    --batch_size 8    \
    --num_epochs 10 \
    --FGN \
    --trial_type 1 \
    --num_FGN 2 \
    --trial_iter 5 \
    --FG_style 1 \
    --mini

echo "FGvln_3e5_2FG_5it_mini 1 Done..."

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  \
    --nproc_per_node 4   \
    --master_port 5558   \
    -m train   \
    --from_pretrained result/FGvln_1phase/data/29.bin \
    --save_name FGvln_3e5_2FG_5it_2   \
    --shuffle_visual_features   \
    --ranking   \
    --batch_size 8    \
    --num_epochs 10 \
    --FGN \
    --trial_type 1 \
    --num_FGN 2 \
    --trial_iter 5 \
    --FG_style 1 \
    --mini

echo "FGvln_3e5_2FG_5it_mini 2 Done..."


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  \
    --nproc_per_node 4   \
    --master_port 5558   \
    -m train   \
    --from_pretrained result/FGvln_1phase/data/29.bin \
    --save_name FGvln_3e5_2FG_3it_out-domain_one_frame_mini  \
    --shuffle_visual_features   \
    --ranking   \
    --batch_size 8    \
    --num_epochs 10 \
    --FGN \
    --trial_type 1 \
    --num_FGN 2 \
    --FG_style 1 \
    --trial_iter 5 \
    --one_frame \
    --mini

echo "FGvln_3e5_2FG_3it_out-domain_one_frame_mini Done..."