# Standalone app
import getpass

# RL & Machine Learning
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from utils.cli import get_parser
from utils.dataset.dataset_init import load_dataloader
from utils.misc import NoneLogger

from utils.dataset.all_dataset import FGRLDataset

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

        self.dataset = FGRLDataset(
            args = args,
            caption_path=f"data/YouTube-VLN/{args.pre_dataset}/{args.prefix}{args.pre_dataset}_test{args.feather_note}.json",
            tokenizer=tokenizer,
            features_reader=features_reader,
            masked_vision=False,
            masked_language=False,
            training=False,
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
