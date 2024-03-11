# python vis_exp.py --pre_dataset ytb     --from_pretrained data/trained/pretrain_LILY.bin     --save_name ytbvln_2e5_500_MRT     --prefix merge+     --separators     --masked_vision     --masked_language     --ranking     --traj_judge     --batch_size 8     --learning_rate 2e-5     --num_epochs 500     --save_epochs 100

from utils.misc import get_output_dir, set_seed, NoneLogger, logo_print, exp_saver, get_logger
from pathlib import Path
from utils.cli import get_parser
from pretrain import set_cuda,get_local_rank
from utils.dataset.dataset_init import load_dataloader
from utils.misc import get_output_dir, set_seed, NoneLogger, logo_print, exp_saver, get_logger
from lily import Lily, BERT_CONFIG_FACTORY
import torch.distributed as dist
from utils.dataset.all_dataset import YTbDataset
import torch
import random
import numpy as np
from utils.dataset.common import (
    load_json_data,
    perm2num,
    generate_negative_trajectories,
    load_shuffler,
    ytb_get_key,
    _check_enough_images,
    load_trajectories,
    ytb_generate_trajectory_from_listing,
    randomize_regions,
    randomize_tokens,
    load_tokens,
    generate_trajectory_out_listing,
    generate_trajectory_from_listing,
    merge_images,
    merge_frames,
    get_headings,
    shuffle_different,
    shuffle_non_adjacent,
    load_nav_graphs,
    load_distances,
    get_viewpoints,
    save_json_data,
    tokenize,
    InstructionGenerator,
    RephraseInstructionGenerator,
    ConcatenateInstructionGenerator,  
    YTBRephraseInstructionGenerator,
)
from utils.dataset.common import pad_packed
from utils.distributed import set_cuda, get_local_rank, wrap_distributed_model
from tqdm import tqdm
from vilbert.vilbert_init import get_optimization
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader

from transformers import BertTokenizer
from utils.dataset.features_reader import FeaturesReader, BnBFeaturesReader, YTbFeaturesReader, PanoFeaturesReader
from utils.utils_init import get_loss_correct
from typing import List, Dict, Tuple

def get_device(batch):
    return batch[0].device

def compute_metrics_independent(batch: List[torch.Tensor], outputs: Dict[str, torch.Tensor], task, args, logger, reduced_metrics) -> torch.Tensor:
    device = get_device(batch)
    local_rank = get_local_rank(args)
    batch_size, target, loss, correct = get_loss_correct(batch, outputs, task, args, logger, True) 

    # calculate accumulated stats
    reduced_loss = loss.detach().float()
    reduced_correct = correct.detach().float()
    reduced_batch_size = torch.tensor(batch_size, device=device).detach().float()

    # TODO: skip this `all_reduce` to speed-up runtime
    if local_rank != -1 and not args.skip_all_reduce:
        world_size = float(dist.get_world_size())
        reduced_loss /= world_size
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM) # type: ignore
        dist.all_reduce(reduced_correct, op=dist.ReduceOp.SUM) # type: ignore
        dist.all_reduce(reduced_batch_size, op=dist.ReduceOp.SUM) # type: ignore
    
    reduced_metrics["loss"][f"{task}"] = reduced_loss
    if not (task == 'vision' or task == 'language'):
        reduced_metrics["accuracy"][f"{task}"] = reduced_correct / reduced_batch_size

    return loss


def get_ranking_target(batch):
    return batch[0]

def get_mask_options(batch) -> torch.Tensor:
    return batch[13]

def get_model_input(batch):
    (
        _,
        image_features,
        image_locations,
        image_mask,
        _,
        _,
        instr_tokens,
        instr_mask,
        _,
        instr_highlights,
        segment_ids,
        co_attention_mask,
        _,
        opt_mask,
        _,
        attend_order_visual_feature,
    ) = batch



    # remove padding samples
    image_features = image_features[opt_mask]
    image_locations = image_locations[opt_mask]
    image_mask = image_mask[opt_mask]
    instr_tokens = instr_tokens[opt_mask]
    instr_mask = instr_mask[opt_mask]
    instr_highlights = instr_highlights[opt_mask]
    segment_ids = segment_ids[opt_mask]
    # transform batch shape
    co_attention_mask = co_attention_mask.view(
        -1, co_attention_mask.size(2), co_attention_mask.size(3)
    )



    return (
        instr_tokens,
        image_features,
        image_locations,
        segment_ids,
        instr_mask,
        image_mask,
        co_attention_mask,
        instr_highlights,
        attend_order_visual_feature,
    )

class VisDataset(YTbDataset):
    
    def __getitem__(self, index: int):

        
        # get a random listing_id
        
        listing_id = self._listing_ids[index]


        # select negative and positive photo ids
        (
            positive_ids,
            negative_captions,
            negative_images,
            negative_random,
            order_labels
        ) = self._pick_photo_ids(listing_id)
        
        # new_list = [positive_ids[:-1]] 
        # new_list[0].append(positive_ids[0])

        # new_list_1 = positive_ids[:-2] 
        # new_list_1.append(positive_ids[0])
        # new_list_1.append(positive_ids[0])

        # new_list.append(new_list_1)

        # negative_captions = new_list # replacement

        
        # print("positive_ids {} , \n negative_captions {} , \n \n".format(positive_ids,negative_captions))

        # get the order label of trajectory
        ordering_target = []
        order_atteneded_visual_feature = 1
        
        prob_order = 1
            
        for key in order_labels:
            if key == "normal_idx" or key == "negative_captions_idx":
                # Skip normal_idx and negative_captions_idx and consider only negative_images_idx
                continue
            else:
                for random_order_path in range(len(order_labels[key])):
                    if prob_order < 0.7:
                        order_atteneded_visual_feature = 1 # 1 indicates random and 0 indicates normal
                        temp = [v for v in order_labels[key][random_order_path] ]
                        # If the path length is too short, it is automatically filled to the longest path
                        temp +=  [-1] * (self.args.max_path_length - len(positive_ids))
                        ordering_target.append(temp)
                    else:
                        order_atteneded_visual_feature = 0 # 1 indicates random and 0 indicates normal
                        ordering_target.append([i for i in range(len(positive_ids))] + \
                                                [-1] * (self.args.max_path_length - len(positive_ids)))

        # get the positive pair
        build_instruction = random.choice(self._build_instructions)
        self.templete = None
        
        instructions = [self.generate_instruction(build_instruction,positive_ids)]
        f, b, p, m = self._get_visual_features(positive_ids)
        features, boxes, probs, masks = [f], [b], [p], [m] # This feature will patch to the longest length (8)

        
        
        if self._traj_judge: # Trajectory judgment task
            negative_traj = negative_captions + negative_images + negative_random
            for traj in negative_traj:
                instructions += [instructions[0]]
                f, b, p, m = self._get_visual_features(traj)
                features += [f]
                boxes += [b]
                probs += [p]
                masks += [m]

        else:
            # get the negative captions
            for traj in negative_captions:
                instructions += [self.generate_instruction(build_instruction,traj)]
                features += [features[0]]
                boxes += [boxes[0]]
                probs += [probs[0]]
                masks += [masks[0]]

            if self.args.negative_style == 'shuffle_instruction':
                # get the negative captions
                for traj in negative_images:
                    instructions += [self.generate_instruction(build_instruction,traj)]
                    features += [features[0]]
                    boxes += [boxes[0]]
                    probs += [probs[0]]
                    masks += [masks[0]]
            else:
                # get the negative images
                for traj in negative_images:
                    instructions += [instructions[0]]
                    f, b, p, m = self._get_visual_features(traj)
                    features += [f]
                    boxes += [b]
                    probs += [p]
                    masks += [m]

            # get the random images
            for traj in negative_random:
                instructions += [instructions[0]]
                f, b, p, m = self._get_visual_features(traj)
                features += [f]
                boxes += [b]
                probs += [p]
                masks += [m]


        # convert data into tensors
        image_features = torch.from_numpy(np.array(features)).float()
        image_boxes = torch.from_numpy(np.array(boxes)).float()
        image_probs = torch.from_numpy(np.array(probs)).float()
        image_masks = torch.from_numpy(np.array(masks)).long()
        instr_tokens = torch.from_numpy(np.array(instructions)).long()
        instr_mask = instr_tokens > 0
        segment_ids = torch.zeros_like(instr_tokens)
        instr_highlights = torch.zeros((image_features.shape[0], 0)).long()


        # randomly mask image features
        if self._masked_vision:
            image_features, image_targets, image_targets_mask = randomize_regions(
                image_features, image_probs, image_masks
            )
        else:
            image_targets = torch.ones_like(image_probs) / image_probs.shape[-1]
            image_targets_mask = torch.zeros_like(image_masks)

        # randomly mask instruction tokens
        if self._masked_language:
            instr_tokens, instr_targets = randomize_tokens(
                instr_tokens, instr_mask, self._tokenizer, self.args
            )
        else:
            instr_targets = torch.ones_like(instr_tokens) * -1

        # construct null return items
        co_attention_mask = torch.zeros(
            2, self.args.max_path_length * self.args.max_num_boxes, self.args.max_instruction_length
        ).long()
        
        ordering_target = torch.tensor(ordering_target)
        if self._training:
            ranking_target = torch.tensor(0)
        else:
            ranking_target = torch.zeros(image_features.shape[0]).bool()
            ranking_target[0] = 1
        
        return (
            ranking_target,
            image_features,
            image_boxes,
            image_masks,
            image_targets,
            image_targets_mask,
            instr_tokens,
            instr_mask,
            instr_targets,
            instr_highlights,
            segment_ids,
            co_attention_mask,
            torch.tensor(self.get_listing_ids(listing_id)).long(),
            torch.ones(image_features.shape[0]).bool(),
            ordering_target,
            order_atteneded_visual_feature,
        )



# command line parsing
parser = get_parser()
parser.add_argument("--final", default=False, action="store_true")
args = parser.parse_args()


set_seed(args)


# get device settings
default_gpu, _, device = set_cuda(args)
logger = NoneLogger()


# create data loaders
local_rank = get_local_rank(args)
train_data_loader, test_data_loader, val_seen_data_loader, val_unseen_data_loader = load_dataloader(args, default_gpu, logger, local_rank)

# load pre-trained model

# Loading model
logger.info(f"Loading model")
config = BERT_CONFIG_FACTORY[args.model_name].from_json_file(args.config_file)


config.args = args

if len(args.from_pretrained) == 0:  # hack for catching --from_pretrained ""
    model = Lily(config)
else:
    model = Lily.from_pretrained(
        args.from_pretrained, config, default_gpu=True
    )

model.to(device)
model = wrap_distributed_model(model, local_rank)


optimizer, scheduler, model, start_epoch = get_optimization(args, model, len(train_data_loader), logger)



def load_features_reader(args) -> FeaturesReader:
    if args.pre_dataset == 'ytb':
        return YTbFeaturesReader(args.ytb_feature)
    
def get_testset_path(args) -> str:
    testset_path = {}
    if args.ranking or args.not_traj_judge_data:
        if args.negative_style == "normal":
            negative_style = ""
        else:
            negative_style = args.negative_style + "_"
        testset_path["ranking"] = get_path(args, negative_style)
    if args.traj_judge and not args.ranking:
        # when ranking and traj_judge work simultaneously, use ranking's testset
        testset_path["traj"] =  get_path(args, "traj_")
    
    return testset_path

def get_path(args, task_prefix) ->str:
    return f"data/YouTube-VLN/{args.pre_dataset}/{args.prefix}{task_prefix}testset{args.feather_note}.json"



# construct model inputs
caption_path = f"data/YouTube-VLN/{args.pre_dataset}/{args.prefix}{args.pre_dataset}_train{args.feather_note}.json"
tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
features_reader = load_features_reader(args)
separators = ("then", "and", ",", ".") if args.separators else ("[SEP]",)
testset_path = get_testset_path(args)


Datset = VisDataset(
    args = args,
    caption_path=caption_path,
    tokenizer=tokenizer,
    features_reader=features_reader,
    masked_vision=False,
    masked_language=False,
    training=True,
    separators=separators,
    testset_path=testset_path,
)

train_sampler = RandomSampler(Datset)

batch_size = args.batch_size // args.gradient_accumulation_steps
if local_rank != -1:
    batch_size = batch_size // dist.get_world_size()


train_data_loader, test_data_loader, val_seen_data_loader, val_unseen_data_loader = load_dataloader(args, default_gpu, logger, local_rank)

model.train()   # CHANGE
model.zero_grad()

for step, batch in enumerate(tqdm(train_data_loader, disable= not (default_gpu))):

    
    model.zero_grad()


    batch = tuple(
            t.cuda(device=device, non_blocking=True) if hasattr(t, "cuda") else t
            for t in batch
        )

    outputs = model(*get_model_input(batch))

    opt_mask = get_mask_options(batch)
    prediction = pad_packed(outputs["ranking"].squeeze(1), opt_mask)
    target = get_ranking_target(batch)
    correct = torch.sum(torch.argmax(prediction, 1) == target).float()
    
    model.zero_grad()

    reduced_metrics = {}
    reduced_metrics["loss"] = {}
    reduced_metrics["accuracy"] = {}

    compute_metrics_independent(batch, outputs, 'ranking', args, logger, reduced_metrics)
    # print("Prediction: {} \n Target: {} \n Correct: {} \n\n".format(prediction,target,correct))
