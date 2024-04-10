# NIPS24_TIA
Official Implementation of [xxx].


## Training with Masking loss

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  \
    --nproc_per_node 4   \
    --master_port 5558   \
    -m train   \
    --from_pretrained data/trained/pretrain_lily_new.bin \
    --save_name FGvln_1phase  \
    --masked_vision \
    --masked_language \
    --batch_size 12    \
    --num_epochs 30 
```

## Training with Path Ranking loss

Random Sample FGN:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  \
    --nproc_per_node 8   \
    --master_port 5558   \
    -m train   \
    --from_pretrained data/trained/pretrain_lily_new.bin \
    --save_name FGvln_2e5_Rnd_1FG_2phase  \
    --shuffle_visual_features   \
    --ranking   \
    --batch_size 16    \
    --num_epochs 30 \
    --trial_type 0 \
    --num_FGN 1
```

BO-based Sample FGN:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  \
    --nproc_per_node 8   \
    --master_port 5558   \
    -m train   \
    --from_pretrained data/trained/pretrain_lily_new.bin \
    --save_name FGvln_2e5_BO_1FG_2phase  \
    --shuffle_visual_features   \
    --ranking   \
    --batch_size 16    \
    --num_epochs 30 \
    --trial_type 0 \
    --num_FGN 1
```

## Training with Path Ranking loss and Aug Data

Random Sample FGN:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  \
    --nproc_per_node 8   \
    --master_port 5558   \
    -m train   \
    --from_pretrained data/trained/pretrain_lily_new.bin \
    --save_name FGvln_2e5_Rnd_1FG_2phase  \
    --prefix aug+ \
    --beam_prefix aug_ \
    --shuffle_visual_features   \
    --ranking   \
    --batch_size 16    \
    --num_epochs 30 \
    --trial_type 0 \
    --num_FGN 1
```

BO-based Sample FGN:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  \
    --nproc_per_node 8   \
    --master_port 5558   \
    -m train   \
    --from_pretrained data/trained/pretrain_lily_new.bin \
    --save_name FGvln_2e5_BO_1FG_2phase  \
    --prefix aug+ \
    --beam_prefix aug_ \
    --shuffle_visual_features   \
    --ranking   \
    --batch_size 16    \
    --num_epochs 30 \
    --trial_type 0 \
    --num_FGN 1
```