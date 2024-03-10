# NIPS24_TIA
Official Implementation of [xxx].

python pretrain.py     --pre_dataset ytb     --from_pretrained data/YouTube-VLN/pretrained_model.bin     --save_name ytbvln_2e5_500_MRT     --prefix merge+     --separators     --masked_vision     --masked_language     --ranking     --traj_judge     --batch_size 8     --learning_rate 2e-5     --num_epochs 500     --save_epochs 100