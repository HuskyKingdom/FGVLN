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
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch     --nproc_per_node 4     --master_port 1234     -m pretrain     --pre_dataset ytb     --from_pretrained data/YouTube-VLN/pretrained_model.bin     --save_name ytbvln_2e5_500_MRT     --prefix merge+     --separators     --masked_vision     --masked_language     --ranking     --traj_judge     --batch_size 8     --learning_rate 2e-5     --num_epochs 500     --save_epochs 100

# Place other commands here

echo "end of job"
~                                                                                             
~                                                                                             
~                                                                                             
~                                                                                             
~                                                                                             
~                          