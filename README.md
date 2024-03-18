# NIPS24_TIA
Official Implementation of [xxx].

python pretrain.py     --pre_dataset ytb     --from_pretrained data/trained/pretrain_LILY.bin     --save_name ytbvln_2e5_500_MRT     --prefix merge+     --separators     --masked_vision     --masked_language     --ranking     --traj_judge     --batch_size 8     --learning_rate 2e-5     --num_epochs 500     --save_epochs 100


python vis_exp.py --pre_dataset ytb     --from_pretrained data/trained/pretrain_LILY.bin     --save_name ytbvln_2e5_500_MRT     --prefix merge+     --separators     --masked_vision     --masked_language     --ranking     --traj_judge     --batch_size 8     --learning_rate 2e-5     --num_epochs 500     --save_epochs 100


python train.py --from_pretrained --from_pretrained data/trained/trained_best_unseen.bin --save_name ytbvln_2e5_500_MRT_ranking_30M_30RS --shuffle_visual_features --ranking --batch_size 16 --num_epochs 30




CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node 4 \
    --master_port 1234 \
    -m pretrain \
    --pre_dataset ytb \
    --from_pretrained data/YouTube-VLN/pretrained_model.bin \
    --save_name ytbvln_2e5_500_MRT \
    --prefix merge+ \
    --separators \
    --masked_vision \
    --masked_language \
    --ranking \
    --traj_judge \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_epochs 500 \
    --save_epochs 100