# python vis_exp.py --from_pretrained data/trained/pretrain_LILY.bin  --pre_dataset ytb --prefix merge+

from pathlib import Path
from utils.cli import get_parser
from pretrain import set_cuda,get_local_rank
from utils.dataset.dataset_init import load_dataloader
from utils.misc import get_output_dir, set_seed, NoneLogger, logo_print, exp_saver, get_logger
from lily import Lily, BERT_CONFIG_FACTORY

from utils.dataset.all_dataset import YTbDataset


def get_input():

    positive_ids = []

    pass




# command line parsing
parser = get_parser()
parser.add_argument("--final", default=False, action="store_true")
args = parser.parse_args()

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
        args.from_pretrained, config, default_gpu=default_gpu
    )

model.to(device)




# construct model inputs