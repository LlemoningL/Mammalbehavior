import os.path as osp
from copy import deepcopy
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.runner import Runner
from mmaction.registry import RUNNERS
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime


class TRAINER:
    def __init__(self, config, args, task):
        self.now_time = datetime.now().strftime('%Y%m%d%H%M')
        self.cfg = config
        self.args = args
        self.task = task
        if task != 'face' and task != 'track':
            self.cfg = Config.fromfile(self.cfg)
            self.merge_args(self.task)
        self.output_dir = Path(self.args.work_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_pose(self):
        if self.task != 'pose':
            raise ValueError('task must be [pose]')

        self.cfg = self.merge_args(self.task)
        if 'preprocess_cfg' in self.cfg:
            self.cfg.model.setdefault('data_preprocessor',
                                      self.cfg.get('preprocess_cfg', {}))
        self.runner(self.cfg)

    def train_faceid(self):
        if self.task != 'faceid':
            raise ValueError('task must be [faceid]')
        self.cfg = self.merge_args(self.task)
        self.runner(self.cfg)

    def train_behavior(self):
        if self.task != 'behavior':
            raise ValueError('task must be [behavior]')
        self.cfg = self.merge_args(self.task)
        self.runner(self.cfg)

    def train_face(self):
        if self.task != 'face':
            raise ValueError('task must be [face]')
        self.yolo()

    def train_track(self):
        if self.task != 'track':
            raise ValueError('task must be [track]')
        self.yolo()

    def merge_args(self, task):
        """Merge CLI arguments to config."""
        if self.args.no_validate:
            self.cfg.val_cfg = None
            self.cfg.val_dataloader = None
            self.cfg.val_evaluator = None

        self.cfg.launcher = self.args.launcher

        # work_dir is determined in this priority: CLI > segment in file > filename
        if self.args.work_dir is not None:
            # update configs according to CLI args if args.work_dir is not None
            self.cfg.work_dir = osp.join(self.args.work_dir, f'{task}_{self.now_time}')
        elif self.cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            self.cfg.work_dir = osp.join('./work_dirs',
                                         osp.splitext(osp.basename(self.args.config))[0])
            # resume training
        if self.args.resume == 'auto':
            self.cfg.resume = True
            self.cfg.load_from = None
        elif self.args.resume is not None:
            self.cfg.resume = True
            self.cfg.load_from = self.args.resume

        # enable auto scale learning rate
        if self.args.auto_scale_lr:
            self.cfg.auto_scale_lr.enable = True

        if task == 'pose':
            if self.args.amp is True:
                from mmengine.optim import AmpOptimWrapper, OptimWrapper
                optim_wrapper = self.cfg.optim_wrapper.get('type', OptimWrapper)
                assert optim_wrapper in (OptimWrapper, AmpOptimWrapper,
                                         'OptimWrapper', 'AmpOptimWrapper'), \
                    '`--amp` is not supported custom optimizer wrapper type ' \
                    f'`{optim_wrapper}.'
                self.cfg.optim_wrapper.type = 'AmpOptimWrapper'
                self.cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')
            # visualization
            if self.args.show or (self.args.show_dir is not None):
                assert 'visualization' in self.cfg.default_hooks, \
                    'PoseVisualizationHook is not set in the ' \
                    '`default_hooks` field of config. Please set ' \
                    '`visualization=dict(type="PoseVisualizationHook")`'

                self.cfg.default_hooks.visualization.enable = True
                self.cfg.default_hooks.visualization.show = self.args.show
                if self.args.show:
                    self.cfg.default_hooks.visualization.wait_time = self.args.wait_time
                self.cfg.default_hooks.visualization.out_dir = self.args.show_dir
                self.cfg.default_hooks.visualization.interval = self.args.interval
        elif task == 'behavior':
            if self.args.amp is True:
                optim_wrapper = self.cfg.optim_wrapper.get('type', 'OptimWrapper')
                assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper'], \
                    '`--amp` is not supported custom optimizer wrapper type ' \
                    f'`{optim_wrapper}.'
                self.cfg.optim_wrapper.type = 'AmpOptimWrapper'
                self.cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')
            if self.cfg.get('randomness', None) is None:
                self.cfg.randomness = dict(
                    seed=self.args.seed,
                    diff_rank_seed=self.args.diff_rank_seed,
                    deterministic=self.args.deterministic)
        elif task == 'faceid':
            # enable automatic-mixed-precision training
            if self.args.amp is True:
                self.cfg.optim_wrapper.type = 'AmpOptimWrapper'
                self.cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')
            default_dataloader_cfg = ConfigDict(
                pin_memory=True,
                persistent_workers=True,
                collate_fn=dict(type='default_collate'),
            )
            if digit_version(TORCH_VERSION) < digit_version('1.8.0'):
                default_dataloader_cfg.persistent_workers = False

            def set_default_dataloader_cfg(cfg, field):
                if cfg.get(field, None) is None:
                    return
                dataloader_cfg = deepcopy(default_dataloader_cfg)
                dataloader_cfg.update(self.cfg[field])
                cfg[field] = dataloader_cfg
                if self.args.no_pin_memory:
                    cfg[field]['pin_memory'] = False
                if self.args.no_persistent_workers:
                    cfg[field]['persistent_workers'] = False

            set_default_dataloader_cfg(self.cfg, 'train_dataloader')
            set_default_dataloader_cfg(self.cfg, 'val_dataloader')
            set_default_dataloader_cfg(self.cfg, 'test_dataloader')

        if self.args.cfg_options is not None:
            self.cfg.merge_from_dict(self.args.cfg_options)

        return self.cfg
    
    def runner(self, cfg):
        if 'runner_type' not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        # start training
        runner.train()

    def yolo(self):
        model = YOLO('yolov8.yaml').load('yolov8m.pt')
        if self.args.resume == 'auto':
            self.cfg.resume = True
        model.train(data=self.cfg,
                    epochs=100,
                    # imgsz=(1152, 768),
                    batch=128,
                    device=0,
                    pretrained=True,
                    val=True,
                    # save_json=True,
                    resume=self.args.resume,
                    project=f'{self.output_dir}',
                    name=f'{self.task}_{self.now_time}')

    def train(self):
        if self.task == 'pose':
            self.train_pose()
        elif self.task == 'faceid':
            self.train_faceid()
        elif self.task == 'behavior':
            self.train_behavior()
        elif self.task == 'face':
            self.train_face()
        elif self.task == 'track':
            self.train_track()
        else:
            raise NotImplementedError(f'{self.task} not implemented yet')
        
  
