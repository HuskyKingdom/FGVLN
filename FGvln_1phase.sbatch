#!/bin/bash

# Generic options:

#SBATCH --account=bdliv07  # Run job under project <project>
#SBATCH --time=48:0:0     

# Node resources:
# (choose between 1-4 gpus per node)

#SBATCH --partition=gpu    # Choose either "gpu" or "infer" node type
#SBATCH --nodes=1          # Resources from a single node
#SBATCH --gres=gpu:4       # One GPU per node (plus 25% of node CPU and RAM per GPU)

# Specify when we should receive e-mail about the job - in this case if it ends or fails
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sgyson10@liverpool.ac.uk

# Run commands:
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  \
    --nproc_per_node 4   \
    --master_port 5558   \
    -m train   \
    --from_pretrained data/trained/pretrain_lily_new.bin \
    --save_name FGvln_1phase  \
    --masked_vision \
    --masked_language \
    --batch_size 12    \
    --num_epochs 30 \
    --learning_rate 0.5e-5
# Place other commands here
echo "end of job"
