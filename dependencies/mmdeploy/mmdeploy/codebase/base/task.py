# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
from mmengine import Config
from mmengine.model import BaseDataPreprocessor
from torch.utils.data import DataLoader, Dataset

from mmdeploy.utils import (get_backend_config, get_codebase,
                            get_codebase_config, get_root_logger)
from mmdeploy.utils.config_utils import (get_codebase_external_module,
                                         get_rknn_quantization)
from mmdeploy.utils.dataset import is_can_sort_dataset, sort_dataset


class BaseTask(metaclass=ABCMeta):
    """Wrap the processing functions of a Computer Vision task.

    Args:
        model_cfg (str | Config): Model config file.
        deploy_cfg (str | Config): Deployment config file.
        device (str): A string specifying device type.
        experiment_name (str, optional): Name of current experiment.
            If not specified, timestamp will be used as
            ``experiment_name``. Defaults to ``None``.
    """

    def __init__(self,
                 model_cfg: Config,
                 deploy_cfg: Config,
                 device: str,
                 experiment_name: str = 'BaseTask'):

        self.model_cfg = model_cfg
        self.deploy_cfg = deploy_cfg
        self.device = device

        self.codebase = get_codebase(deploy_cfg)
        self.experiment_name = experiment_name

        # init scope
        from .. import import_codebase
        custom_module_list = get_codebase_external_module(deploy_cfg)
        import_codebase(self.codebase, custom_module_list)

        from mmengine.registry import DefaultScope
        if not DefaultScope.check_instance_created(self.experiment_name):
            self.scope = DefaultScope.get_instance(
                self.experiment_name,
                scope_name=str(self.model_cfg.get('default_scope')))
        else:
            self.scope = DefaultScope.get_instance(self.experiment_name)

        # lazy build visualizer
        self.visualizer = self.model_cfg.get('visualizer', None)

    @abstractmethod
    def build_backend_model(
            self,
            model_files: Sequence[str] = None,
            data_preprocessor_updater: Optional[Callable] = None,
            **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files.
            data_preprocessor_updater (Callable | None): A function to update
                the data_preprocessor. Defaults to None.

        Returns:
            nn.Module: An initialized backend model.
        """
        pass

    def build_data_preprocessor(self):
        """build data preprocessor.

        Returns:
            BaseDataPreprocessor:
                Initialized instance of :class:`BaseDataPreprocessor`.
        """
        model = deepcopy(self.model_cfg.model)
        preprocess_cfg = model['data_preprocessor']

        from mmengine.registry import MODELS
        data_preprocessor = MODELS.build(preprocess_cfg)
        data_preprocessor.to(self.device)

        return data_preprocessor

    def build_pytorch_model(self,
                            model_checkpoint: Optional[str] = None,
                            cfg_options: Optional[Dict] = None,
                            **kwargs) -> torch.nn.Module:
        """Initialize torch model.

        Args:
            model_checkpoint (str): The checkpoint file of torch model,
                defaults to `None`.
            cfg_options (dict): Optional config key-pair parameters.

        Returns:
            nn.Module: An initialized torch model generated by other OpenMMLab
                codebases.
        """
        from mmengine.model import revert_sync_batchnorm
        from mmengine.registry import MODELS

        model = deepcopy(self.model_cfg.model)
        model.pop('pretrained', None)
        preprocess_cfg = deepcopy(self.model_cfg.get('preprocess_cfg', {}))
        preprocess_cfg.update(
            deepcopy(self.model_cfg.get('data_preprocessor', {})))
        model.setdefault('data_preprocessor', preprocess_cfg)
        model = MODELS.build(model)
        if model_checkpoint is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(model, model_checkpoint, map_location=self.device)

        model = revert_sync_batchnorm(model)
        if hasattr(model, 'backbone') and hasattr(model.backbone,
                                                  'switch_to_deploy'):
            model.backbone.switch_to_deploy()

        if hasattr(model, 'switch_to_deploy') and callable(
                model.switch_to_deploy):
            model.switch_to_deploy()

        model = model.to(self.device)
        model.eval()
        return model

    def build_dataset(self,
                      dataset_cfg: Union[str, Config],
                      is_sort_dataset: bool = False,
                      **kwargs) -> Dataset:
        """Build dataset for different codebase.

        Args:
            dataset_cfg (str | Config): Dataset config file or Config
                object.
            is_sort_dataset (bool): When 'True', the dataset will be sorted
                by image shape in ascending order if 'dataset_cfg'
                contains information about height and width.
                Default is `False`.

        Returns:
            Dataset: The built dataset.
        """
        backend_cfg = get_backend_config(self.deploy_cfg)
        from mmdeploy.utils import load_config
        dataset_cfg = load_config(dataset_cfg)[0]
        if 'pipeline' in backend_cfg:
            dataset_cfg.pipeline = backend_cfg.pipeline
        from mmengine.registry import DATASETS
        dataset = DATASETS.build(dataset_cfg)
        logger = get_root_logger()
        if is_sort_dataset:
            if is_can_sort_dataset(dataset):
                sort_dataset(dataset)
            else:
                logger.info('Sorting the dataset by \'height\' and \'width\' '
                            'is not possible.')
        return dataset

    @staticmethod
    def build_dataloader(dataloader: Union[DataLoader, Dict],
                         seed: Optional[int] = None) -> DataLoader:
        """Build PyTorch dataloader. A wrap of Runner.build_dataloader.

        Args:
            dataloader (DataLoader or dict): A Dataloader object or a dict to
                build Dataloader object. If ``dataloader`` is a Dataloader
                object, just returns itself.
            seed (int, optional): Random seed. Defaults to None.

        Returns:
            Dataloader: DataLoader build from ``dataloader_cfg``.
        """
        from mmengine.runner import Runner
        return Runner.build_dataloader(dataloader, seed)

    def build_test_runner(self,
                          model: torch.nn.Module,
                          work_dir: str,
                          log_file: Optional[str] = None,
                          show: bool = False,
                          show_dir: Optional[str] = None,
                          wait_time: int = 0,
                          interval: int = 1,
                          dataloader: Optional[Union[DataLoader,
                                                     Dict]] = None):

        def _merge_cfg(cfg):
            """Merge CLI arguments to config."""
            # -------------------- visualization --------------------
            if show or (show_dir is not None):
                assert 'visualization' in cfg.default_hooks, \
                    'VisualizationHook is not set in the `default_hooks`'\
                    ' field of config. Please set '\
                    '`visualization=dict(type="VisualizationHook")`'

                cfg.default_hooks.visualization.draw = True
                cfg.default_hooks.visualization.show = show
                cfg.default_hooks.visualization.wait_time = wait_time
                cfg.default_hooks.visualization.test_out_dir = show_dir
                cfg.default_hooks.visualization.interval = interval

            return cfg

        model_cfg = deepcopy(self.model_cfg)
        if dataloader is None:
            dataloader = model_cfg.test_dataloader
        if not isinstance(dataloader, DataLoader):
            if type(dataloader) == list:
                dataloader = [self.build_dataloader(dl) for dl in dataloader]
            else:
                dataloader = self.build_dataloader(dataloader)

        model_cfg = _merge_cfg(model_cfg)

        visualizer = self.get_visualizer(work_dir, work_dir)
        from .runner import DeployTestRunner
        runner = DeployTestRunner(
            model=model,
            work_dir=work_dir,
            log_file=log_file,
            device=self.device,
            visualizer=visualizer,
            default_hooks=model_cfg.default_hooks,
            test_dataloader=dataloader,
            test_cfg=model_cfg.test_cfg,
            test_evaluator=model_cfg.test_evaluator,
            default_scope=model_cfg.default_scope)
        return runner

    def update_data_preprocessor(self, data_preprocessor: Config):
        """Update data_preprocessor.

        Args:
            data_preprocessor (mmengine.Config): The data preprocessor.
        Returns:
            mmengine.Config: The updated data preprocessor.
        """
        data_preprocessor = deepcopy(data_preprocessor)
        if get_rknn_quantization(self.deploy_cfg):
            if data_preprocessor is not None:
                data_preprocessor['mean'] = [0, 0, 0]
                data_preprocessor['std'] = [1, 1, 1]
        return data_preprocessor

    @abstractmethod
    def create_input(self,
                     imgs: Union[str, np.ndarray],
                     input_shape: Optional[Sequence[int]] = None,
                     data_preprocessor: Optional[BaseDataPreprocessor] = None,
                     **kwargs) -> Tuple[Dict, torch.Tensor]:
        """Create input for model.

        Args:
            imgs (str | np.ndarray | Sequence): Input image(s),
                accepted data types are `str`, `np.ndarray`.
            input_shape (list[int]): Input shape of image in (width, height)
                format, defaults to `None`.

        Returns:
            tuple: (data, img), meta information for the input image and input
                image tensor.
        """
        pass

    def get_visualizer(self, name: str, save_dir: str):
        """Get the visualizer instance.

        Args:
            name (str): The name of the visualizer.
            save_dir (str): The save directory of visualizer.
        """
        cfg = deepcopy(self.visualizer)
        cfg.name = name
        cfg.save_dir = save_dir
        from mmengine.registry import VISUALIZERS, DefaultScope
        with DefaultScope.overwrite_default_scope(cfg.pop('_scope_', None)):
            # get the global default scope
            default_scope = DefaultScope.get_current_instance()
            if default_scope is not None:
                scope_name = default_scope.scope_name
                root = VISUALIZERS._get_root_registry()
                registry = root._search_child(scope_name)
                if registry is None:
                    registry = VISUALIZERS
            else:
                registry = VISUALIZERS
            VisualizerClass = registry.get(cfg.type)
            if VisualizerClass.check_instance_created(cfg.name):
                return VisualizerClass.get_instance(cfg.name)
            else:
                return registry.build_func(cfg, registry=registry)

    def visualize(self,
                  image: Union[str, np.ndarray],
                  result: list,
                  output_file: str,
                  window_name: str = '',
                  show_result: bool = False,
                  draw_gt: bool = False,
                  **kwargs):
        """Visualize predictions of a model.

        Args:
            model (nn.Module): Input model.
            image (str | np.ndarray): Input image to draw predictions on.
            result (list): A list of predictions.
            output_file (str): Output file to save drawn image.
            window_name (str): The name of visualization window. Defaults to
                an empty string.
            show_result (bool): Whether to show result in windows, defaults
                to `False`.
            draw_gt (bool): Whether to show ground truth in windows, defaults
                to `False`.
        """
        save_dir, save_name = osp.split(output_file)
        visualizer = self.get_visualizer(window_name, save_dir)

        name = osp.splitext(save_name)[0]
        if isinstance(image, str):
            image = mmcv.imread(image, channel_order='rgb')
        assert isinstance(image, np.ndarray)

        visualizer.add_datasample(
            name,
            image,
            data_sample=result,
            draw_gt=draw_gt,
            show=show_result,
            out_file=output_file)

    @staticmethod
    @abstractmethod
    def get_partition_cfg(partition_type: str, **kwargs) -> Dict:
        """Get a certain partition config.

        Args:
            partition_type (str): A string specifying partition type.

        Returns:
            dict: A dictionary of partition config.
        """
        pass

    @staticmethod
    def get_tensor_from_input(input_data: Dict[str, Any],
                              **kwargs) -> torch.Tensor:
        """Get input tensor from input data.

        Args:
            input_data (dict): Input data containing meta info and image
                tensor.
        Returns:
            torch.Tensor: An image in `Tensor`.
        """
        return input_data['inputs']

    @abstractmethod
    def get_preprocess(self, *args, **kwargs) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        pass

    @abstractmethod
    def get_postprocess(self, *args, **kwargs) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Composed of the postprocess information.
        """
        pass

    @abstractmethod
    def get_model_name(self, *args, **kwargs) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        pass

    @property
    def from_mmrazor(self) -> bool:
        """Whether the codebase from mmrazor.

        Returns:
            bool: From mmrazor or not.

        Raises:
            TypeError: An error when type of `from_mmrazor` is not boolean.
        """
        codebase_config = get_codebase_config(self.deploy_cfg)
        from_mmrazor = codebase_config.get('from_mmrazor', False)
        if not isinstance(from_mmrazor, bool):
            raise TypeError('`from_mmrazor` attribute must be boolean type! '
                            f'but got: {from_mmrazor}')

        return from_mmrazor
