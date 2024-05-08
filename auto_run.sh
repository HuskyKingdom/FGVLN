CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  \
    --nproc_per_node 4   \
    --master_port 5558   \
    -m train   \
    --from_pretrained result/FGvln_1phase/data/29.bin \
    --save_name FGvln_3e5_1FG_5it_mini  \
    --shuffle_visual_features   \
    --ranking   \
    --batch_size 8    \
    --num_epochs 10 \
    --FGN \
    --trial_type 1 \
    --num_FGN 1 \
    --trial_iter 5 \
    --mini

echo "FGvln_3e5_1FG_5it_mini Done..."

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  \
    --nproc_per_node 4   \
    --master_port 5558   \
    -m train   \
    --from_pretrained result/FGvln_1phase/data/29.bin \
    --save_name FGvln_3e5_2FG_5it   \
    --shuffle_visual_features   \
    --ranking   \
    --batch_size 8    \
    --num_epochs 10 \
    --FGN \
    --trial_type 1 \
    --num_FGN 2 \
    --trial_iter 5 \
    --mini

echo "FGvln_3e5_2FG_5it_mini Done..."


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  \
    --nproc_per_node 4   \
    --master_port 5558   \
    -m train   \
    --from_pretrained result/FGvln_1phase/data/29.bin \
    --save_name FGvln_3e5_2FG_3it_mini  \
    --shuffle_visual_features   \
    --ranking   \
    --batch_size 8    \
    --num_epochs 10 \
    --FGN \
    --trial_type 1 \
    --num_FGN 2 \
    --mini


echo "FGvln_3e5_2FG_3it_mini Done..."

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  \
    --nproc_per_node 4   \
    --master_port 5558   \
    -m train   \
    --from_pretrained result/FGvln_1phase/data/29.bin \
    --save_name FGvln_3e5_2FG_3it_out-domain_mini  \
    --shuffle_visual_features   \
    --ranking   \
    --batch_size 8    \
    --num_epochs 10 \
    --FGN \
    --trial_type 1 \
    --num_FGN 2 \
    --FG_style 1 \
    --mini


echo "FGvln_3e5_2FG_3it_out-domain_mini Done..."

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