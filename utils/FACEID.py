import torch
import numpy as np
import mmengine.dist as dist
from pathlib import Path
from mmcv.image import imread
from mmengine.dataset import BaseDataset, Compose, default_collate
from mmpretrain.registry import TRANSFORMS
from mmengine.fileio import get_file_backend
from mmpretrain.structures import DataSample
from mmpretrain.apis.model import list_models
from mmengine.model import BaseModel
from abc import abstractmethod
from typing import Callable, Iterable, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
from mmpretrain.registry import MODELS
from mmengine.config import Config
from torch2trt import TRTModule
from mmengine.runner import Runner
from mmpretrain.models.retrievers.base import BaseRetriever
from mmpretrain.utils import track_on_main_process
ModelType = Union[str, Config]
InputType = Union[str, np.ndarray, list]
torch.cuda.set_device(0)


class ImageToImageRetrieverTRT(BaseRetriever):
    def __init__(self,
                 image_encoder,
                 prototype: Union[DataLoader, dict, str, torch.Tensor],
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 similarity_fn: Union[str, Callable] = 'cosine_similarity',
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 topk: int = -1,
                 init_cfg: Optional[dict] = None):

        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        data_preprocessor.setdefault('type', 'mmpretrain.ClsDataPreprocessor')

        if train_cfg is not None and 'augments' in train_cfg:
            # Set batch augmentations by `train_cfg`
            data_preprocessor['batch_augments'] = train_cfg

        super(ImageToImageRetrieverTRT, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if head is not None and not isinstance(head, torch.nn.Module):
            head = MODELS.build(head)

        model_trt = TRTModule()
        # model_trt.load_state_dict(torch.load(image_encoder))
        model_trt.load_state_dict(torch.load(image_encoder))
        model_trt.to('cuda')

        self.image_encoder = model_trt
        self.head = head

        self.similarity = similarity_fn

        assert isinstance(prototype, (str, torch.Tensor, dict, DataLoader)), (
            'The `prototype` in  `ImageToImageRetriever` must be a path, '
            'a torch.Tensor, a dataloader or a dataloader dict format config.')
        self.prototype = prototype
        self.prototype_inited = False
        self.topk = topk

    @property
    def similarity_fn(self):
        """Returns a function that calculates the similarity."""
        # If self.similarity_way is callable, return it directly
        if isinstance(self.similarity, Callable):
            return self.similarity

        if self.similarity == 'cosine_similarity':
            # a is a tensor with shape (N, C)
            # b is a tensor with shape (M, C)
            # "cosine_similarity" will get the matrix of similarity
            # with shape (N, M).
            # The higher the score is, the more similar is
            return lambda a, b: torch.cosine_similarity(
                a.unsqueeze(1), b.unsqueeze(0), dim=-1)
        else:
            raise RuntimeError(f'Invalid function "{self.similarity_fn}".')

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor without any
          post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`DataSample`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor, tuple): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor.
            - If ``mode="predict"``, return a list of
              :obj:`mmpretrain.structures.DataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'tensor':
            return self.extract_feat(inputs)
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs):
        """Extract features from the input tensor with shape (N, C, ...).

        Args:
            inputs (Tensor): A batch of inputs. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
        Returns:
            Tensor: The output of encoder.
        """


        inputs = inputs.cuda()
        # x3 = torch.ones((1, 3, 448, 448)).cuda()
        # feat1 = self.image_encoder(x3)
        feat = self.image_encoder(inputs)

        return feat

    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats = self.extract_feat(inputs)
        return self.head.loss(feats, data_samples)

    def matching(self, inputs: torch.Tensor):
        """Compare the prototype and calculate the similarity.

        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C).
        Returns:
            dict: a dictionary of score and prediction label based on fn.
        """
        sim = self.similarity_fn(inputs, self.prototype_vecs)
        sorted_sim, indices = torch.sort(sim, descending=True, dim=-1)
        predictions = dict(
            score=sim, pred_label=indices, pred_score=sorted_sim)
        return predictions

    def predict(self,
                inputs: tuple,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> List[DataSample]:
        """Predict results from the extracted features.

        Args:
            inputs (tuple): The features extracted from the backbone.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        Returns:
            List[DataSample]: the raw data_samples with
                the predicted results
        """
        if not self.prototype_inited:
            self.prepare_prototype()

        feats = self.extract_feat(inputs)
        if isinstance(feats, tuple):
            feats = feats[-1]

        # Matching of similarity
        result = self.matching(feats)
        return self._get_predictions(result, data_samples)

    def _get_predictions(self, result, data_samples):
        """Post-process the output of retriever."""
        pred_scores = result['score']
        pred_labels = result['pred_label']
        if self.topk != -1:
            topk = min(self.topk, pred_scores.size()[-1])
            pred_labels = pred_labels[:, :topk]

        if data_samples is not None:
            for data_sample, score, label in zip(data_samples, pred_scores,
                                                 pred_labels):
                data_sample.set_pred_score(score).set_pred_label(label)
        else:
            data_samples = []
            for score, label in zip(pred_scores, pred_labels):
                data_samples.append(
                    DataSample().set_pred_score(score).set_pred_label(label))
        return data_samples

    def _get_prototype_vecs_from_dataloader(self, data_loader):
        """get prototype_vecs from dataloader."""
        self.eval()
        num = len(data_loader.dataset)

        prototype_vecs = None
        for data_batch in track_on_main_process(data_loader,
                                                'Prepare prototype'):
        # for data_batch in data_loader:
            data = self.data_preprocessor(data_batch, False)
            feat = self(**data)
            if isinstance(feat, tuple):
                feat = feat[-1]

            if prototype_vecs is None:
                dim = feat.shape[-1]
                prototype_vecs = torch.zeros(num, dim)
            for i, data_sample in enumerate(data_batch['data_samples']):
                sample_idx = data_sample.get('sample_idx')
                # print(sample_idx)
                torch.cuda.synchronize()
                prototype_vecs[sample_idx] = feat[i]

        assert prototype_vecs is not None
        dist.all_reduce(prototype_vecs)
        return prototype_vecs

    def _get_prototype_vecs_from_path(self, proto_path):
        """get prototype_vecs from prototype path."""
        data = [None]
        if dist.is_main_process():
            data[0] = torch.load(proto_path)
        dist.broadcast_object_list(data, src=0)
        prototype_vecs = data[0]
        assert prototype_vecs is not None
        return prototype_vecs

    @torch.no_grad()
    def prepare_prototype(self):
        """Used in meta testing. This function will be called before the meta
        testing. Obtain the vector based on the prototype.

        - torch.Tensor: The prototype vector is the prototype
        - str: The path of the extracted feature path, parse data structure,
            and generate the prototype feature vector set
        - Dataloader or config: Extract and save the feature vectors according
            to the dataloader
        """
        device = 'cuda:0'
        if isinstance(self.prototype, torch.Tensor):
            prototype_vecs = self.prototype
        elif isinstance(self.prototype, str):
            prototype_vecs = self._get_prototype_vecs_from_path(self.prototype)
        elif isinstance(self.prototype, (dict, DataLoader)):
            loader = Runner.build_dataloader(self.prototype)
            prototype_vecs = self._get_prototype_vecs_from_dataloader(loader)

        self.register_buffer(
            'prototype_vecs', prototype_vecs.to(device), persistent=False)
        self.prototype_inited = True

    def dump_prototype(self, path):
        """Save the features extracted from the prototype to specific path.

        Args:
            path (str): Path to save feature.
        """
        if not self.prototype_inited:
            self.prepare_prototype()
        torch.save(self.prototype_vecs, path)


class NewInferencer:
    """Base inferencer for various tasks.

    The BaseInferencer provides the standard workflow for inference as follows:

    1. Preprocess the input data by :meth:`preprocess`.
    2. Forward the data to the model by :meth:`forward`. ``BaseInferencer``
       assumes the model inherits from :class:`mmengine.models.BaseModel` and
       will call `model.test_step` in :meth:`forward` by default.
    3. Visualize the results by :meth:`visualize`.
    4. Postprocess and return the results by :meth:`postprocess`.

    When we call the subclasses inherited from BaseInferencer (not overriding
    ``__call__``), the workflow will be executed in order.

    All subclasses of BaseInferencer could define the following class
    attributes for customization:

    - ``preprocess_kwargs``: The keys of the kwargs that will be passed to
      :meth:`preprocess`.
    - ``forward_kwargs``: The keys of the kwargs that will be passed to
      :meth:`forward`
    - ``visualize_kwargs``: The keys of the kwargs that will be passed to
      :meth:`visualize`
    - ``postprocess_kwargs``: The keys of the kwargs that will be passed to
      :meth:`postprocess`

    All attributes mentioned above should be a ``set`` of keys (strings),
    and each key should not be duplicated. Actually, :meth:`__call__` will
    dispatch all the arguments to the corresponding methods according to the
    ``xxx_kwargs`` mentioned above.

    Subclasses inherited from ``BaseInferencer`` should implement
    :meth:`_init_pipeline`, :meth:`visualize` and :meth:`postprocess`:

    - _init_pipeline: Return a callable object to preprocess the input data.
    - visualize: Visualize the results returned by :meth:`forward`.
    - postprocess: Postprocess the results returned by :meth:`forward` and
      :meth:`visualize`.

    Args:
        model (BaseModel | str | Config): A model name or a path to the config
            file, or a :obj:`BaseModel` object. The model name can be found
            by ``cls.list_models()`` and you can also query it in
            :doc:`/modelzoo_statistics`.
        pretrained (str, optional): Path to the checkpoint. If None, it will
            try to find a pre-defined weight from the model you specified
            (only work if the ``model`` is a model name). Defaults to None.
        device (str | torch.device | None): Transfer the model to the target
            device. Defaults to None.
        device_map (str | dict | None): A map that specifies where each
            submodule should go. It doesn't need to be refined to each
            parameter/buffer name, once a given module name is inside, every
            submodule of it will be sent to the same device. You can use
            `device_map="auto"` to automatically generate the device map.
            Defaults to None.
        offload_folder (str | None): If the `device_map` contains any value
            `"disk"`, the folder where we will offload weights.
        **kwargs: Other keyword arguments to initialize the model (only work if
            the ``model`` is a model name).
    """

    preprocess_kwargs: set = set()
    forward_kwargs: set = set()
    visualize_kwargs: set = set()
    postprocess_kwargs: set = set()

    def __init__(self,
                 model_cfg: ModelType,
                 image_encoder,
                 prototype,
                 # pretrained: Union[bool, str] = True,
                 # device: Union[str, torch.device, None] = None,
                 # device_map=None,
                 # offload_folder=None,
                 **kwargs) -> None:

        # if isinstance(model, BaseModel):
        #     if isinstance(pretrained, str):
        #         load_checkpoint(model, pretrained, map_location='cpu')
        #     if device_map is not None:
        #         from ..dependencies.mmpretrain.mmpretrain.apis.utils import dispatch_model
        #         model = dispatch_model(
        #             model,
        #             device_map=device_map,
        #             offload_folder=offload_folder)
        #     elif device is not None:
        #         model.to(device)
        # else:
        #     model = get_model(
        #         model,
        #         pretrained,
        #         device=device,
        #         device_map=device_map,
        #         offload_folder=offload_folder,
        #         **kwargs)   # todo build a model to convert

        model = ImageToImageRetrieverTRT(image_encoder=image_encoder,
                                         prototype=prototype)


        model.eval()

        self.config = Config.fromfile(model_cfg)
        self.model = model
        self.pipeline = self._init_pipeline(self.config)
        self.visualizer = None

    def __call__(
        self,
        inputs,
        return_datasamples: bool = False,
        batch_size: int = 1,
        **kwargs,
    ) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            batch_size (int): Batch size. Defaults to 1.
            **kwargs: Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results.
        """
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        ori_inputs = self._inputs_to_list(inputs)
        inputs = self.preprocess(
            ori_inputs, batch_size=batch_size, **preprocess_kwargs)
        preds = []
        for data in inputs:
            preds.extend(self.forward(data, **forward_kwargs))
        visualization = self.visualize(ori_inputs, preds, **visualize_kwargs)
        results = self.postprocess(preds, visualization, return_datasamples,
                                   **postprocess_kwargs)
        return results

    def _inputs_to_list(self, inputs: InputType) -> list:
        """Preprocess the inputs to a list.

        Cast the input data to a list of data.

        - list or tuple: return inputs
        - str:
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              could be a path to file, a url or other types of string according
              to the task.
        - other: return a list with one item.

        Args:
            inputs (str | array | list): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        if isinstance(inputs, str):
            backend = get_file_backend(inputs)
            if hasattr(backend, 'isdir') and backend.isdir(inputs):
                # Backends like HttpsBackend do not implement `isdir`, so only
                # those backends that implement `isdir` could accept the inputs
                # as a directory
                file_list = backend.list_dir_or_file(inputs, list_dir=False)
                inputs = [
                    backend.join_path(inputs, file) for file in file_list
                ]

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return list(inputs)

    def preprocess(self, inputs: InputType, batch_size: int = 1, **kwargs):
        """Process the inputs into a model-feedable format.

        Customize your preprocess by overriding this method. Preprocess should
        return an iterable object, of which each item will be used as the
        input of ``model.test_step``.

        ``BaseInferencer.preprocess`` will return an iterable chunked data,
        which will be used in __call__ like this:

        .. code-block:: python

            def __call__(self, inputs, batch_size=1, **kwargs):
                chunked_data = self.preprocess(inputs, batch_size, **kwargs)
                for batch in chunked_data:
                    preds = self.forward(batch, **kwargs)

        Args:
            inputs (InputsType): Inputs given by user.
            batch_size (int): batch size. Defaults to 1.

        Yields:
            Any: Data processed by the ``pipeline`` and ``default_collate``.
        """
        chunked_data = self._get_chunk_data(
            map(self.pipeline, inputs), batch_size)
        yield from map(default_collate, chunked_data)

    @torch.no_grad()
    def forward(self, inputs: Union[dict, tuple], **kwargs):
        """Feed the inputs to the model."""
        return self.model.test_step(inputs)

    def visualize(self,
                  inputs: list,
                  preds: List[DataSample],
                  show: bool = False,
                  **kwargs) -> List[np.ndarray]:
        """Visualize predictions.

        Customize your visualization by overriding this method. visualize
        should return visualization results, which could be np.ndarray or any
        other objects.

        Args:
            inputs (list): Inputs preprocessed by :meth:`_inputs_to_list`.
            preds (Any): Predictions of the model.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.

        Returns:
            List[np.ndarray]: Visualization results.
        """
        if show:
            raise NotImplementedError(
                f'The `visualize` method of {self.__class__.__name__} '
                'is not implemented.')

    @abstractmethod
    def postprocess(
        self,
        preds: List[DataSample],
        visualization: List[np.ndarray],
        return_datasample=False,
        **kwargs,
    ) -> dict:
        """Process the predictions and visualization results from ``forward``
        and ``visualize``.

        This method should be responsible for the following tasks:

        1. Convert datasamples into a json-serializable dict if needed.
        2. Pack the predictions and visualization results and return them.
        3. Dump or log the predictions.

        Customize your postprocess by overriding this method. Make sure
        ``postprocess`` will return a dict with visualization results and
        inference results.

        Args:
            preds (List[Dict]): Predictions of the model.
            visualization (np.ndarray): Visualized predictions.
            return_datasample (bool): Whether to return results as datasamples.
                Defaults to False.

        Returns:
            dict: Inference and visualization results with key ``predictions``
            and ``visualization``

            - ``visualization (Any)``: Returned by :meth:`visualize`
            - ``predictions`` (dict or DataSample): Returned by
              :meth:`forward` and processed in :meth:`postprocess`.
              If ``return_datasample=False``, it usually should be a
              json-serializable dict containing only basic data elements such
              as strings and numbers.
        """

    @abstractmethod
    def _init_pipeline(self, cfg: Config) -> Callable:
        """Initialize the test pipeline.

        Return a pipeline to handle various input data, such as ``str``,
        ``np.ndarray``. It is an abstract method in BaseInferencer, and should
        be implemented in subclasses.

        The returned pipeline will be used to process a single data.
        It will be used in :meth:`preprocess` like this:

        .. code-block:: python
            def preprocess(self, inputs, batch_size, **kwargs):
                ...
                dataset = map(self.pipeline, dataset)
                ...
        """

    def _get_chunk_data(self, inputs: Iterable, chunk_size: int):
        """Get batch data from dataset.

        Args:
            inputs (Iterable): An iterable dataset.
            chunk_size (int): Equivalent to batch size.

        Yields:
            list: batch data.
        """
        inputs_iter = iter(inputs)
        while True:
            try:
                chunk_data = []
                for _ in range(chunk_size):
                    processed_data = next(inputs_iter)
                    chunk_data.append(processed_data)
                yield chunk_data
            except StopIteration:
                if chunk_data:
                    yield chunk_data
                break

    def _dispatch_kwargs(self, **kwargs) -> Tuple[dict, dict, dict, dict]:
        """Dispatch kwargs to preprocess(), forward(), visualize() and
        postprocess() according to the actual demands.

        Returns:
            Tuple[Dict, Dict, Dict, Dict]: kwargs passed to preprocess,
            forward, visualize and postprocess respectively.
        """
        # Ensure each argument only matches one function
        method_kwargs = self.preprocess_kwargs | self.forward_kwargs | \
            self.visualize_kwargs | self.postprocess_kwargs

        union_kwargs = method_kwargs | set(kwargs.keys())
        if union_kwargs != method_kwargs:
            unknown_kwargs = union_kwargs - method_kwargs
            raise ValueError(
                f'unknown argument {unknown_kwargs} for `preprocess`, '
                '`forward`, `visualize` and `postprocess`')

        preprocess_kwargs = {}
        forward_kwargs = {}
        visualize_kwargs = {}
        postprocess_kwargs = {}

        for key, value in kwargs.items():
            if key in self.preprocess_kwargs:
                preprocess_kwargs[key] = value
            if key in self.forward_kwargs:
                forward_kwargs[key] = value
            if key in self.visualize_kwargs:
                visualize_kwargs[key] = value
            if key in self.postprocess_kwargs:
                postprocess_kwargs[key] = value

        return (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        )

    @staticmethod
    def list_models(pattern: Optional[str] = None):
        """List models defined in metafile of corresponding packages.

        Args:
            pattern (str | None): A wildcard pattern to match model names.

        Returns:
            List[str]: a list of model names.
        """
        return list_models(pattern=pattern)


class FaceidInferencerTRT(NewInferencer):
    """The inferencer for image to image retrieval.

    Args:
        model (BaseModel | str | Config): A model name or a path to the config
            file, or a :obj:`BaseModel` object. The model name can be found
            by ``ImageRetrievalInferencer.list_models()`` and you can also
            query it in :doc:`/modelzoo_statistics`.
        prototype (str | list | dict | DataLoader, BaseDataset): The images to
            be retrieved. It can be the following types:

            - str: The directory of the the images.
            - list: A list of path of the images.
            - dict: A config dict of the a prototype dataset.
            - BaseDataset: A prototype dataset.
            - DataLoader: A data loader to load the prototype data.

        prototype_cache (str, optional): The path of the generated prototype
            features. If exists, directly load the cache instead of re-generate
            the prototype features. If not exists, save the generated features
            to the path. Defaults to None.
        pretrained (str, optional): Path to the checkpoint. If None, it will
            try to find a pre-defined weight from the model you specified
            (only work if the ``model`` is a model name). Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        **kwargs: Other keyword arguments to initialize the model (only work if
            the ``model`` is a model name).

    Example:
        >>> from mmpretrain import ImageRetrievalInferencer
        >>> inferencer = ImageRetrievalInferencer(
        ...     'resnet50-arcface_inshop',
        ...     prototype='./demo/',
        ...     prototype_cache='img_retri.pth')
        >>> inferencer('demo/cat-dog.png', topk=2)[0][1]
        {'match_score': tensor(0.4088, device='cuda:0'),
         'sample_idx': 3,
         'sample': {'img_path': './demo/dog.jpg'}}
    """  # noqa: E501

    visualize_kwargs: set = {
        'draw_score', 'resize', 'show_dir', 'show', 'wait_time', 'topk'
    }
    postprocess_kwargs: set = {'topk'}

    def __init__(
        self,
        model_cfg: ModelType,
        image_encoder,
        prototype,
        prototype_cache=None,
        prepare_batch_size=1,
        pretrained: Union[bool, str] = True,
        device: Union[str, torch.device, None] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_cfg=model_cfg, image_encoder=image_encoder, prototype=prototype, **kwargs)

        self.prototype_dataset = self._prepare_prototype(
            prototype, prototype_cache, prepare_batch_size)

    def _prepare_prototype(self, prototype, cache=None, batch_size=8):
        from mmengine.dataset import DefaultSampler
        from torch.utils.data import DataLoader

        def build_dataloader(dataset):
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=default_collate,
                sampler=DefaultSampler(dataset, shuffle=False),
                persistent_workers=False,
            )

        if isinstance(prototype, str):
            # A directory path of images
            prototype = dict(
                type='CustomDataset', with_label=False, data_root=prototype)

        if isinstance(prototype, list):
            test_pipeline = [dict(type='LoadImageFromFile'), self.pipeline]
            dataset = BaseDataset(
                lazy_init=True, serialize_data=False, pipeline=test_pipeline)
            dataset.data_list = [{
                'sample_idx': i,
                'img_path': file
            } for i, file in enumerate(prototype)]
            dataset._fully_initialized = True
            dataloader = build_dataloader(dataset)
        elif isinstance(prototype, dict):
            # A config of dataset
            from mmpretrain.registry import DATASETS
            test_pipeline = [dict(type='LoadImageFromFile'), self.pipeline]
            prototype.setdefault('pipeline', test_pipeline)
            dataset = DATASETS.build(prototype)
            dataloader = build_dataloader(dataset)
        elif isinstance(prototype, DataLoader):
            dataset = prototype.dataset
            dataloader = prototype
        elif isinstance(prototype, BaseDataset):
            dataset = prototype
            dataloader = build_dataloader(dataset)
        else:
            raise TypeError(f'Unsupported prototype type {type(prototype)}.')

        if cache is not None and Path(cache).exists():
            self.model.prototype = cache
        else:
            self.model.prototype = dataloader
        self.model.prepare_prototype()

        from mmengine.logging import MMLogger
        logger = MMLogger.get_current_instance()
        if cache is None:
            logger.info('The prototype has been prepared, you can use '
                        '`save_prototype` to dump it into a pickle '
                        'file for the future usage.')
        elif not Path(cache).exists():
            self.save_prototype(cache)
            logger.info(f'The prototype has been saved at {cache}.')

        return dataset

    def save_prototype(self, path):
        self.model.dump_prototype(path)

    def __call__(self,
                 inputs: InputType,
                 return_datasamples: bool = False,
                 batch_size: int = 1,
                 **kwargs) -> dict:
        """Call the inferencer.

        Args:
            inputs (str | array | list): The image path or array, or a list of
                images.
            return_datasamples (bool): Whether to return results as
                :obj:`DataSample`. Defaults to False.
            batch_size (int): Batch size. Defaults to 1.
            resize (int, optional): Resize the long edge of the image to the
                specified length before visualization. Defaults to None.
            draw_score (bool): Whether to draw the match scores.
                Defaults to True.
            show (bool): Whether to display the visualization result in a
                window. Defaults to False.
            wait_time (float): The display time (s). Defaults to 0, which means
                "forever".
            show_dir (str, optional): If not None, save the visualization
                results in the specified directory. Defaults to None.

        Returns:
            list: The inference results.
        """
        return super().__call__(inputs, return_datasamples, batch_size,
                                **kwargs)

    def _init_pipeline(self, cfg: Config) -> Callable:
        test_pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        from mmpretrain.datasets import remove_transform

        # Image loading is finished in `self.preprocess`.
        test_pipeline_cfg = remove_transform(test_pipeline_cfg,
                                             'LoadImageFromFile')
        test_pipeline = Compose(
            [TRANSFORMS.build(t) for t in test_pipeline_cfg])
        return test_pipeline

    def preprocess(self, inputs: List[InputType], batch_size: int = 1):

        def load_image(input_):
            img = imread(input_)
            if img is None:
                raise ValueError(f'Failed to read image {input_}.')
            return dict(
                img=img,
                img_shape=img.shape[:2],
                ori_shape=img.shape[:2],
            )

        pipeline = Compose([load_image, self.pipeline])

        chunked_data = self._get_chunk_data(map(pipeline, inputs), batch_size)
        yield from map(default_collate, chunked_data)

    def visualize(self,
                  ori_inputs: List[InputType],
                  preds: List[DataSample],
                  topk: int = 3,
                  resize: Optional[int] = 224,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_score=True,
                  show_dir=None):
        if not show and show_dir is None:
            return None

        if self.visualizer is None:
            from mmpretrain.visualization import UniversalVisualizer
            self.visualizer = UniversalVisualizer()

        visualization = []
        for i, (input_, data_sample) in enumerate(zip(ori_inputs, preds)):
            image = imread(input_)
            if isinstance(input_, str):
                # The image loaded from path is BGR format.
                image = image[..., ::-1]
                name = Path(input_).stem
            else:
                name = str(i)

            if show_dir is not None:
                show_dir = Path(show_dir)
                show_dir.mkdir(exist_ok=True)
                out_file = str((show_dir / name).with_suffix('.png'))
            else:
                out_file = None

            self.visualizer.visualize_image_retrieval(
                image,
                data_sample,
                self.prototype_dataset,
                topk=topk,
                resize=resize,
                draw_score=draw_score,
                show=show,
                wait_time=wait_time,
                name=name,
                out_file=out_file)
            visualization.append(self.visualizer.get_image())
        if show:
            self.visualizer.close()
        return visualization

    def postprocess(
        self,
        preds: List[DataSample],
        visualization: List[np.ndarray],
        return_datasamples=False,
        topk=1,
    ) -> dict:
        if return_datasamples:
            return preds

        results = []
        for data_sample in preds:
            match_scores, indices = torch.topk(data_sample.pred_score, k=topk)
            matches = []
            for match_score, sample_idx in zip(match_scores, indices):
                sample = self.prototype_dataset.get_data_info(
                    sample_idx.item())
                sample_idx = sample.pop('sample_idx')
                matches.append({
                    'match_score': match_score,
                    'sample_idx': sample_idx,
                    'sample': sample
                })
            results.append(matches)

        return results

    @staticmethod
    def list_models(pattern: Optional[str] = None):
        """List all available model names.

        Args:
            pattern (str | None): A wildcard pattern to match model names.

        Returns:
            List[str]: a list of model names.
        """
        return list_models(pattern=pattern, task='Image Retrieval')








