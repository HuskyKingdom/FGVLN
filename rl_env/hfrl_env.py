# Standalone app
import getpass

# RL & Machine Learning
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from utils.cli import get_parser
from utils.dataset.dataset_init import load_dataloader
from utils.misc import NoneLogger
from transformers import BertTokenizer
from utils.dataset.all_dataset import FGRLDataset
from utils.dataset.features_reader import FeaturesReader, BnBFeaturesReader, YTbFeaturesReader, PanoFeaturesReader
from torch.utils.data import RandomSampler, DataLoader

def get_path(args, task_prefix) ->str:
    return f"data/YouTube-VLN/{args.pre_dataset}/{args.prefix}{task_prefix}testset{args.feather_note}.json"

def load_features_reader(args) -> FeaturesReader:
    if args.pre_dataset == 'ytb':
        return YTbFeaturesReader(args.ytb_feature)
    elif args.pre_dataset == 'bnb':
        return BnBFeaturesReader(args.bnb_feature)
    elif not args.pretrain:
        return PanoFeaturesReader(args.img_feature)
    
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

class TLAlignmentTask(gym.Env):

    def __init__(self,args):


        # set the action and observation space for RL
        self.action_space = spaces.Box(low=-50, high=50,
						    shape=(1), dtype=np.uint8)
        self.observation_space = spaces.Dict({
            "instr_tokens":spaces.Box(low=float('-inf'), high=float('inf'),
						    shape=(56, 60), dtype=np.uint8),
            "image_features":spaces.Box(low=float('-inf'), high=float('inf'),
						    shape=(56, 808, 2048), dtype=np.uint8),
            "image_locations":spaces.Box(low=float('-inf'), high=float('inf'),
                shape=(56, 808, 12), dtype=np.uint8),
            "segment_ids":spaces.Box(low=float('-inf'), high=float('inf'),
                shape=(56, 60), dtype=np.uint8),
            "instr_mask":spaces.Box(low=float('-inf'), high=float('inf'),
                shape=(56, 60), dtype=np.uint8),
            "image_mask":spaces.Box(low=float('-inf'), high=float('inf'),
                shape=(56, 808), dtype=np.uint8),
            "co_attention_mask":spaces.Box(low=float('-inf'), high=float('inf'),
                shape=(16, 808, 60), dtype=np.uint8),
        })

        
        # command line parsing
        parser = get_parser()
        parser.add_argument("--final", default=False, action="store_true")
        args = parser.parse_args()
        self.args = args

        # env dataset
        caption_path = f"data/YouTube-VLN/{args.pre_dataset}/{args.prefix}{args.pre_dataset}_train{args.feather_note}.json"
        tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
        features_reader = load_features_reader(args)
        separators = ("then", "and", ",", ".") if args.separators else ("[SEP]",)
        testset_path = get_testset_path(args)

        dataset = FGRLDataset(
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
        train_sampler = RandomSampler(dataset)

        self.data_loader = DataLoader(
            dataset,
            sampler=train_sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )


    def get_model_input(self,batch):
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

    def step(self, action):
        ...
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        
        self.current_bs = next(iter(self.data_loader))
        self.current_bs = self.get_model_input(self.current_bs)



        return observation, info

    def render(self):
        ...

    def close(self):
        ...
