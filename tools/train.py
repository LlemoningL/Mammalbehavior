from TRAINER import TRAINER
from argparse import ArgumentParser
from yacs.config import CfgNode as CN
from mmengine.config import Config, DictAction


def train_default():

    _C = CN()
    _C.PRE = False
    _C.TRAIN = CN()

    _C.TRAIN.FACE = CN()
    _C.TRAIN.FACE.cfg = ''

    _C.TRAIN.BODY = CN()
    _C.TRAIN.BODY.cfg = ''

    _C.TRAIN.POSE = CN()
    _C.TRAIN.POSE.cfg = ''

    _C.TRAIN.BEHAVIOR = CN()
    _C.TRAIN.BEHAVIOR.cfg = ''

    _C.TRAIN.FACEID = CN()
    _C.TRAIN.FACEID.cfg = ''

    _C.freeze()
    config = _C.clone()

    return config

def parser_args():

    parser = ArgumentParser(description='Process video.')
    parser.add_argument('cfg', help='Path to config file'),
    parser.add_argument('--task',
                        choices=['face', 'track', 'pose', 'behavior', 'faceid'],
                        help='Choose a task to train')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
             'specify, try to auto resume from the latest checkpoint '
             'in the work directory.')
    parser.add_argument(
        '--work_dir',
        default='work_dir',
        type=str,
        help='Output directory to save results')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        default=True,
        help='whether to auto scale the learning rate according to the '
             'actual batch size and the original batch size.')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-rank-seed',
        action='store_true',
        help='whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='whether to disable the pin_memory option in dataloaders.')
    parser.add_argument(
        '--no-persistent-workers',
        action='store_true',
        help='whether to disable the persistent_workers option in dataloaders.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    cfgs = train_default()
    cfgs.merge_from_file(args.cfg)
    cfgs.freeze()

    return cfgs, args


def main():
    cfg, args = parser_args()
    if args.task == 'pose':
        cfgs = cfg.TRAIN.POSE.cfg
    elif args.task == 'faceid':
        cfgs = cfg.TRAIN.FACEID.cfg
    elif args.task == 'behavior':
        cfgs = cfg.TRAIN.BEHAVIOR.cfg
    elif args.task == 'face':
        cfgs = cfg.TRAIN.FACE.cfg
    elif args.task == 'track':
        cfgs = cfg.TRAIN.BODY.cfg
    else:
        raise NotImplementedError

    trainer = TRAINER(cfgs, args, args.task)
    trainer.train()


if __name__ == '__main__':
    main()