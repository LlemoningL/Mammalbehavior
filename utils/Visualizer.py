import warnings
import mmengine
from typing import Optional
from mmengine.config import Config
# from ..dependencies.mmaction2.mmaction.registry import VISUALIZERS
from mmaction.registry import VISUALIZERS
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from utils.util import vis_box, visualize_frame, show_img



class Visualizer:
    def __init__(self, cfgs):
        pose_config = mmengine.Config.fromfile(cfgs.MODEL.POSE.cfg)
        self.visualizer = VISUALIZERS.build(pose_config.visualizer)
        dataset_meta = None
        if dataset_meta is None:
            dataset_meta = self.dataset_meta_from_config(pose_config, dataset_mode='train')
        if dataset_meta is None:
            warnings.simplefilter('once')
            warnings.warn('Can not load dataset_meta from the checkpoint or the '
                          'model config. Use COCO metainfo by default.')
            dataset_meta = parse_pose_metainfo(
                dict(from_file='configs/_base_/datasets/coco.py'))
        self.visualizer.set_dataset_meta(dataset_meta)

    def visualize(self, frame, frame_coordinates, id_bbox_colors, data_sample):
        frame = vis_box(frame, frame_coordinates, id_bbox_colors)
        vis_img = visualize_frame(self.visualizer, frame, data_sample)

        return vis_img

    def show(self, vis_img):
        show_img(vis_img)

    def dataset_meta_from_config(self, config: Config,
                                 dataset_mode: str = 'train') -> Optional[dict]:
        """Get dataset metainfo from the model config.

        Args:
            config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
                :obj:`Path`, or the config object.
            dataset_mode (str): Specify the dataset of which to get the metainfo.
                Options are ``'train'``, ``'val'`` and ``'test'``. Defaults to
                ``'train'``

        Returns:
            dict, optional: The dataset metainfo. See
            ``mmpose.datasets.datasets.utils.parse_pose_metainfo`` for details.
            Return ``None`` if failing to get dataset metainfo from the config.
        """
        try:
            if dataset_mode == 'train':
                dataset_cfg = config.train_dataloader.dataset
            elif dataset_mode == 'val':
                dataset_cfg = config.val_dataloader.dataset
            elif dataset_mode == 'test':
                dataset_cfg = config.test_dataloader.dataset
            else:
                raise ValueError(
                    f'Invalid dataset {dataset_mode} to get metainfo. '
                    'Should be one of "train", "val", or "test".')

            if 'metainfo' in dataset_cfg:
                metainfo = dataset_cfg.metainfo
            else:
                import mmpose.datasets.datasets  # noqa: F401, F403
                from mmpose.registry import DATASETS

                dataset_class = dataset_cfg.type if isinstance(
                    dataset_cfg.type, type) else DATASETS.get(dataset_cfg.type)
                metainfo = dataset_class.METAINFO

            metainfo = parse_pose_metainfo(metainfo)

        except AttributeError:
            metainfo = None

        return metainfo

