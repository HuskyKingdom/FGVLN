# NIPS24_TIA
Official Implementation of [xxx].

Fine-grained crossmodality alignment between trajectory and instruction via Bayes filtering.

[Upstream](https://github.com/JeremyLinky/YouTube-VLN)

## Install (Offline Data Only For Downstream Task)

1. Download the checkpoint of VilBERT pre-trained on [Conceptual Captions](https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin) and then put it into *data/*.

<!-- *data/pretrained_model.bin* -->

2. Download the [matterport-ResNet-101-faster-rcnn features](https://dl.dropbox.com/s/67k2vjgyjqel6og/matterport-ResNet-101-faster-rcnn-genome.lmdb.zip) and unzip it and then put it into *data/*.

3. Download the [instruction template](https://drive.google.com/file/d/1skdU4Kvs3E1jvqBSBvtsLsxMXYbtQ7fp/view?usp=sharing) and then put it into *data/task*.

4. Follow [download.py](scripts/download.py) to download the other data of tasks.
```bash
python scripts/download.py
```

5. Download [model](https://drive.google.com/file/d/1reRM3yKULDEHuxamcmx0enn9fh147rWs/view?usp=sharing) pretrained on upstream YoutubeVLN task, and put it into *data/trained/pretrain_lily_new.bin*.

6. There are 3 files you need to download mannuly again:

    [data/config/bert_base_6_layer_6_connect.json](https://drive.google.com/uc?id=17mL0qCWnIjqL2GNku8A7CKAi6A8Scogh)

    [data/task/aug+R2R_train.json](https://drive.google.com/uc?id=1cA2GRF_EGB8cw_XIxk8b6TXSEaWZEDk7)

    [data/beamsearch/aug_beams_train.json](https://drive.google.com/uc?id=1ukpTRI6LelEl0_gk10azW_Td95XANL2e)


The final file structure should looks like the following:

```
data/
  beamsearch/
  config/
  connectivity/
  distances/
  matterport-ResNet-101-faster-rcnn-genome.lmdb
  task/
  trained/
  pretrained_model.bin
```

## Training Downsteam Task

### Single GPU Training For Debuging

```
python train.py     \
--from_pretrained data/trained/pretrain_lily_new.bin     \
--save_name FGvln_2e5_500_MRT_ranking_30M_30RSA     \
--prefix aug+     \
--beam_prefix aug_    \    
--shuffle_visual_features   \     
--ranking     \
--batch_size 2   \    
--num_epochs 30   
```


### Distributed GPU Training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node 8 \
    --master_port 5555 \
    -m train \
    --from_pretrained data/trained/pretrain_lily_new.bin \
    --save_name FGvln_2e5_500_MRT_ranking_30M_30RSA \
    --prefix aug+ \
    --beam_prefix aug_ \
    --shuffle_visual_features \
    --ranking \
    --batch_size 16 \
    --num_epochs 30 
```
