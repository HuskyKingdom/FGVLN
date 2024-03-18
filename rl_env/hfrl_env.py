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
            "img":spaces.Box(low=0, high=255,
						    shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            "map":spaces.Box(low=0, high=255,
						    shape=(self.obs_size, self.obs_size,1), dtype=np.uint8),
            "text_emb":spaces.Box(low=-1, high=1,
						    shape=(512,), dtype=np.float16),
            "img_emb":spaces.Box(low=-1, high=1,
						    shape=(512,), dtype=np.float16)
        })

        
        # command line parsing
        parser = get_parser()
        parser.add_argument("--final", default=False, action="store_true")
        args = parser.parse_args()
        self.args = args

        # # construct model inputs
        caption_path = f"data/YouTube-VLN/{args.pre_dataset}/{args.prefix}{args.pre_dataset}_train{args.feather_note}.json"
        tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
        features_reader = load_features_reader(args)
        separators = ("then", "and", ",", ".") if args.separators else ("[SEP]",)
        testset_path = get_testset_path(args)

        self.dataset = FGRLDataset(
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



        



    def step(self, action):
        ...
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        ...
        return observation, info

    def render(self):
        ...

    def close(self):
        ...
