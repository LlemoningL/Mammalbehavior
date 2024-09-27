import numpy as np
import torch
import torch.nn as nn
import datetime
from overrides import override
from copy import deepcopy
from ultralytics import YOLO
from mmengine.config import Config
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope,  DefaultScope
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from mmaction.apis import inference_skeleton, init_recognizer
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.models.builder import build_pose_estimator
from utils.util import split_xyxy, get_color, is_boxid, FaceidInferencer
from utils.FACEID import FaceidInferencerTRT
from utils.reid_encoder import ReIDEncoder
from pathlib import Path
from typing import List, Optional, Union
from mmengine.dataset import Compose, pseudo_collate
from PIL import Image
from mmpose.structures import PoseDataSample, merge_data_samples
from mmpose.structures.bbox import bbox_xywh2xyxy
from mmdeploy.utils import get_input_shape, load_config
from mmdeploy.apis.utils import build_task_processor
import warnings


class ModelHandler:
    def __init__(self, configs, frame_shape, behavior_label, DataManager, fps, logger):
        self.logger = logger
        self.cfgs = configs
        self.frame_shape = frame_shape
        self.fps = fps
        self.behavior_label = behavior_label
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.init_model()
        self.track_bboxes = None
        self.behavior_cls = ''
        self.behavior_prob = ''
        self.DataManager = DataManager
        self.data_sample = []
        self.frame_coordinates = {}
        self.undetected_counts = {}
        self.pose_results_splited = None
        self.id_bbox_colors = {}
        self.id_bbox_colors = self.color([i for i in range(0, 200)])
        self.text_dict = {}

    def process_frame(self,
                      frame,
                      frame_tag,
                      time_tag,
                      current_frame_id,
                      current_frame_time_stamp):

        if frame_tag:
            _box, _id = self.Track(frame)
            self.track_bboxes = is_boxid(_box, _id, self.frame_coordinates)

        if time_tag:
            self.pose_results_splited = self.DataManager.split_pose_result()

        if self.track_bboxes is not None or self.pose_results_splited:
            obejects_to_process = self.pose_results_splited \
                if self.pose_results_splited else {k[-1]: k for k in self.track_bboxes}

            self.frame_coordinates = {}
            self.data_sample = []
            for track_id, pose_result in obejects_to_process.items():
                if time_tag:
                    self.behavior_cls, self.behavior_prob = self.Behavior(pose_result, self.frame_shape)
                    track_bbox = pose_result[-1]['bboxes']
                    self.pose_results_splited = None
                else:
                    track_bbox = np.reshape(deepcopy(pose_result), (1, -1))
                self.process_single_object(
                    frame,
                    track_id,
                    track_bbox,
                    current_frame_id,
                    current_frame_time_stamp,
                    self.behavior_cls,
                    self.behavior_prob,
                    time_tag)
                # self.track_bboxes = None

        self.update_tracking(self.track_bboxes, frame_tag)
        self.track_bboxes = None

        return frame, self.frame_coordinates, self.data_sample

    def process_single_object(self,
                              frame,
                              track_id,
                              track_bbox,
                              current_frame_id,
                              current_frame_time_stamp,
                              behavior_cls,
                              behavior_prob,
                              time_tag
                              ):
        pose_result, data_sample = self.Pose(frame,
                                             [track_bbox])
        self.data_sample.extend(data_sample)
        id = int(track_id)
        box = track_bbox[:, 0:4][0]
        self.DataManager.update_pose_result(id, pose_result)
        body_x1, body_y1, body_x2, body_y2 = split_xyxy(track_bbox[:, 0:4])
        body_area = frame[body_y1:body_y2, body_x1:body_x2]

        face_name, face_result = self.process_face(body_area, track_bbox, current_frame_id)

        if id not in self.text_dict:
            self.text_dict[id] = {'face_name': face_name, 'track_id': id, 'cls': '', 'prob': '', 'text_extend': None}
        else:
            self.text_dict[id].update({'face_name': face_name, 'track_id': id})

        text = f'{face_name} {id}'
        if time_tag:
            self.DataManager.update_frame_info(
                face_name,
                id,
                current_frame_id,
                current_frame_time_stamp,
                behavior_cls)
            self.text_dict = self.DataManager.update_label_text(
                self.text_dict,
                face_name,
                id,
                behavior_cls,
                behavior_prob)
        if self.text_dict[id]['text_extend'] is not None:
            text += self.text_dict[id]['text_extend']

        if face_result is not None or box.shape[0] != 0:
            self.frame_coordinates = self.update_frame_coordinates(
                id,
                box,
                face_result,
                text)
        self.logger.info(f'Frame [{current_frame_id}], '
                         f'Time stamp [{current_frame_time_stamp}], '
                         f'Info "{text}"')

    def process_face(self, body_area, track_bbox, frame_id):
        face_result = self.Face(body_area)
        if face_result[0].boxes.shape[0] == 0:
            face_name = None
            face_result = None
        else:
            face_xyxy = face_result[0].boxes.xyxy.cpu().numpy()[0]
            face_x1, face_y1, face_x2, face_y2 = split_xyxy(face_xyxy)
            face_area = body_area[face_y1:face_y2, face_x1:face_x2]
            face_name, face_score = self.FaceID(face_area)
            if face_score < 0.7:
                face_name = None
        face_name = self.DataManager.update_faceid_trackid(face_name,
                                                           int(track_bbox[:, -1]),
                                                           frame_id)

        return face_name, face_result

    def color(self, track_ids):
        bbox_colors = get_color(track_ids)
        for t_id, b_colors in zip(track_ids, bbox_colors):
            if t_id not in self.id_bbox_colors:
                self.id_bbox_colors[int(t_id)] = tuple(b_colors)
        return self.id_bbox_colors

    def update_frame_coordinates(self, track_id, body_result, face_result, label_text):
        self.frame_coordinates[int(track_id)] = [body_result, face_result, label_text]

        return self.frame_coordinates

    def update_tracking(self, new_track_bboxes, frame_tag=False):
        new_tracks_dict = {}

        if new_track_bboxes is not None:
            # 将新的跟踪结果转换为字典格式
            new_tracks_dict = {int(bbox[4]): bbox[:4] for bbox in new_track_bboxes}

        if new_tracks_dict and frame_tag:
            # 增加未检测到的目标的计数
            for track_id in list(self.frame_coordinates.keys()):
                if track_id not in new_tracks_dict:
                    self.undetected_counts[track_id] = self.undetected_counts.get(track_id, 0) + 1
                else:
                    self.undetected_counts[track_id] = 0
        else:
            # 在非检测帧中，增加所有目标的计数
            for track_id in self.frame_coordinates.keys():
                self.undetected_counts[track_id] = self.undetected_counts.get(track_id, 0) + 1

        # 清除长时间未检测到的目标
        for track_id in list(self.undetected_counts.keys()):
            if self.undetected_counts[track_id] >= 7:
                self.frame_coordinates.pop(track_id, None)
                self.undetected_counts.pop(track_id, None)

    def init_model(self):
        self.face = YOLO(self.cfgs.MODEL.FACE.weight)
        print('loads face detect model')
        self.logger.info('loads face detect model')

        new_instance_name = f'mmpretrain-{datetime.datetime.now()}'
        DefaultScope.get_instance(new_instance_name, scope_name='mmpretrain')
        self.faceid = FaceidInferencer(
            self.cfgs.MODEL.FACEID.cfg,
            prototype=self.cfgs.MODEL.FACEID.prototype,
            prototype_cache=self.cfgs.MODEL.FACEID.prototype_cache,
            pretrained=self.cfgs.MODEL.FACEID.weight,
            device=self.DEVICE)
        print('loads face id model')
        self.logger.info('loads face id model')
        self.tracker = YOLO(self.cfgs.MODEL.BODY.weight)
        print('loads track model')
        self.logger.info('loads track model')
        if self.cfgs.MODEL.BODY.with_reid:
            if self.cfgs.MODEL.BODY.reid_encoder is not None:
                ckpt = torch.load(self.cfgs.MODEL.BODY.reid_encoder)
                input_size = ckpt['input_size']
            else:
                input_size = (224, 224)
            self.reid_encoder = ReIDEncoder(
                input_size=input_size,
                weights_path=self.cfgs.MODEL.BODY.reid_encoder)

            print('loads reid model')
            self.logger.info('loads reid model')
        else:
            self.reid_encoder = None

        self.behavior = init_recognizer(
            self.cfgs.MODEL.BEHAVIOR.cfg,
            self.cfgs.MODEL.BEHAVIOR.weight,
            device=self.DEVICE)
        print('loads behavior model')
        self.logger.info('loads behavior model')
        behavior_cfg = Config.fromfile(self.cfgs.MODEL.BEHAVIOR.cfg)
        test_pipeline_cfg = behavior_cfg.test_pipeline
        self.behavior_test_pipeline = Compose(test_pipeline_cfg)

        self.pose = self.pose_model(
            config=self.cfgs.MODEL.POSE.cfg,
            checkpoint=self.cfgs.MODEL.POSE.weight,
            device=self.DEVICE)
        print('loads pose model')
        self.logger.info('loads pose model')

    def Face(self, body_area):
        face_reasutl = self.face(body_area,
                                 device=self.DEVICE,
                                 verbose=False,
                                 # conf=0.75
                                 )
        self.logger.info('Face detect')

        return face_reasutl

    def FaceID(self, face_area, top=1):
        predict = self.faceid(face_area, topk=top)[0]
        face_score = round(predict[0]['match_score'].cpu().numpy().item(), 2)
        face_name = predict[0]['sample']['img_path'].split('/')[-2] \
            if '/' in predict[0]['sample']['img_path'] \
            else predict[0]['sample']['img_path'].split('\\')[-2]
        self.logger.info(f'Face id recognize [{face_name}] [{face_score}]')
        return face_name, face_score

    def Track(self, img):
        track_result = self.tracker.track(
            source=img,
            device=self.DEVICE,
            verbose=False,
            persist=True,
            stream=True,
            encoder=self.reid_encoder,
            with_reid=self.cfgs.MODEL.BODY.with_reid,
            frame_rate=self.fps)

        self.logger.info('Body track')
        for result in track_result:
            return result.boxes.xyxy, result.boxes.id
    def Pose(self, frame_paths, det_results: List[np.ndarray]):
        new_instance_name = f'mmpose-{datetime.datetime.now()}'
        DefaultScope.get_instance(new_instance_name, scope_name='mmpose')

        results = []
        data_samples = []
        pose_data_samples = self.inference_topdown(
            self.pose,
            frame_paths,
            det_results[0][..., :4],
            bbox_format='xyxy')
        self.logger.info('Pose estimate')
        pose_data_sample = merge_data_samples(pose_data_samples)
        pose_data_sample.dataset_meta = self.pose.dataset_meta
        # make fake pred_instances
        if not hasattr(pose_data_sample, 'pred_instances'):
            num_keypoints = self.pose.dataset_meta['num_keypoints']
            pred_instances_data = dict(
                keypoints=np.empty(shape=(0, num_keypoints, 2)),
                keypoints_scores=np.empty(shape=(0, 17), dtype=np.float32),
                bboxes=np.empty(shape=(0, 4), dtype=np.float32),
                bbox_scores=np.empty(shape=(0), dtype=np.float32))
            pose_data_sample.pred_instances = InstanceData(
                **pred_instances_data)

        poses = pose_data_sample.pred_instances.to_dict()
        poses['track_bboxes'] = det_results[0]
        results.append(poses)
        data_samples.append(pose_data_sample)

        return results, data_samples

    def Behavior(self, pose_results, img_shape):
        new_instance_name = f'mmaction-{datetime.datetime.now()}'
        DefaultScope.get_instance(new_instance_name, scope_name='mmaction')
        outputs = inference_skeleton(self.behavior, pose_results, img_shape,
                                     self.behavior_test_pipeline)
        max_pred_index = outputs.pred_score.argmax().item()
        pose_list = self.behavior_label
        pose_name = pose_list[max_pred_index]
        pose_prob = round(outputs.pred_score[max_pred_index].item(), 2)
        self.logger.info('Behavior recognize')

        return pose_name, pose_prob

    def pose_model(self,
                   config: Union[str, Path, Config],
                   checkpoint: Optional[str] = None,
                   device: Union[str, torch.device] = 'cuda:0',
                   cfg_options: Optional[dict] = None):
        if isinstance(config, (str, Path)):
            config = Config.fromfile(config)
        elif not isinstance(config, Config):
            raise TypeError('config must be a filename or Config object, '
                            f'but got {type(config)}')
        if cfg_options is not None:
            config.merge_from_dict(cfg_options)
        elif 'init_cfg' in config.model.backbone:
            config.model.backbone.init_cfg = None
        config.model.train_cfg = None

        new_instance_name = f'mmpose-{datetime.datetime.now()}'
        DefaultScope.get_instance(new_instance_name, scope_name='mmpose')

        model = build_pose_estimator(config.model)
        model = revert_sync_batchnorm(model)
        # get dataset_meta in this priority: checkpoint > config > default (COCO)
        dataset_meta = None

        if checkpoint is not None:
            ckpt = load_checkpoint(model, checkpoint, map_location='cuda')
            if 'dataset_meta' in ckpt.get('meta', {}):
                # checkpoint from mmpose 1.x
                dataset_meta = ckpt['meta']['dataset_meta']
        if dataset_meta is None:
            dataset_meta = self.dataset_meta_from_config(config, dataset_mode='train')
        if dataset_meta is None:
            warnings.simplefilter('once')
            warnings.warn('Can not load dataset_meta from the checkpoint or the '
                          'model config. Use COCO metainfo by default.')
            dataset_meta = parse_pose_metainfo(
                dict(from_file='configs/_base_/datasets/coco.py'))

        model.dataset_meta = dataset_meta

        model.cfg = config  # save the config in the model for convenience
        model.to(device)
        model.eval()


        self.pose_pipeline = Compose(config.test_dataloader.dataset.pipeline)
        # self.


        return model

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

    def inference_topdown(self, model: nn.Module,
                          img: Union[np.ndarray, str],
                          bboxes: Optional[Union[List, np.ndarray]] = None,
                          bbox_format: str = 'xyxy') -> List[PoseDataSample]:
        """Inference image with a top-down pose estimator.

        Args:
            model (nn.Module): The top-down pose estimator
            img (np.ndarray | str): The loaded image or image file to inference
            bboxes (np.ndarray, optional): The bboxes in shape (N, 4), each row
                represents a bbox. If not given, the entire image will be regarded
                as a single bbox area. Defaults to ``None``
            bbox_format (str): The bbox format indicator. Options are ``'xywh'``
                and ``'xyxy'``. Defaults to ``'xyxy'``

        Returns:
            List[:obj:`PoseDataSample`]: The inference results. Specifically, the
            predicted keypoints and scores are saved at
            ``data_sample.pred_instances.keypoints`` and
            ``data_sample.pred_instances.keypoint_scores``.
        """
        scope = model.cfg.get('default_scope', 'mmpose')
        if scope is not None:
            init_default_scope(scope)
        pipeline = self.pose_pipeline

        if bboxes is None or len(bboxes) == 0:
            # get bbox from the image size
            if isinstance(img, str):
                w, h = Image.open(img).size
            else:
                h, w = img.shape[:2]

            bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
        else:
            if isinstance(bboxes, list):
                bboxes = np.array(bboxes)

            assert bbox_format in {'xyxy', 'xywh'}, \
                f'Invalid bbox_format "{bbox_format}".'

            if bbox_format == 'xywh':
                bboxes = bbox_xywh2xyxy(bboxes)

        # construct batch data samples
        data_list = []
        for bbox in bboxes:
            if isinstance(img, str):
                data_info = dict(img_path=img)
            else:
                data_info = dict(img=img)
            data_info['bbox'] = bbox[None]  # shape (1, 4)
            data_info['bbox_score'] = np.ones(1, dtype=np.float32)  # shape (1,)
            data_info.update(model.dataset_meta)
            data_list.append(pipeline(data_info))

        if data_list:
            # collate data list into a batch, which is a dict with following keys:
            # batch['inputs']: a list of input images
            # batch['data_samples']: a list of :obj:`PoseDataSample`
            batch = pseudo_collate(data_list)
            with torch.no_grad():
                results = model.test_step(batch)
        else:
            results = []

        return results


class ModelTRTHandler(ModelHandler):
    def __init__(self, configs, frame_shape, behavior_label, DataManager, fps, logger):
        self.logger = logger
        self.cfgs = configs
        self.fps = fps
        self.frame_shape = frame_shape
        self.behavior_label = behavior_label
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.init_model()
        self.track_bboxes = None
        self.behavior_cls = ''
        self.behavior_prob = ''
        self.DataManager = DataManager
        self.data_sample = []
        self.frame_coordinates = dict()
        self.undetected_counts = {}
        self.id_bbox_colors = dict()
        self.pose_results_splited = None
        self.id_bbox_colors = self.color([i for i in range(0, 200)])
        self.text_dict = {}

    def process_frame(self,
                      frame,
                      frame_tag,
                      time_tag,
                      current_frame_id,
                      current_frame_time_stamp):

        if frame_tag:
            _box, _id = self.Track(frame)
            self.track_bboxes = is_boxid(_box, _id, self.frame_coordinates)

        if time_tag:
            self.pose_results_splited = self.DataManager.split_pose_result()

        if self.track_bboxes is not None or self.pose_results_splited is not None:
            obejects_to_process = self.pose_results_splited \
                if self.pose_results_splited is not None \
                else {k[-1]: k for k in self.track_bboxes}

            self.frame_coordinates = {}
            self.data_sample = []
            for track_id, pose_result in obejects_to_process.items():
                if time_tag:
                    self.behavior_cls, self.behavior_prob = self.Behavior(pose_result,
                                                                          self.frame_shape)
                    track_bbox = pose_result[-1]['bboxes']
                    self.pose_results_splited = None
                else:
                    track_bbox = np.reshape(deepcopy(pose_result), (1, -1))
                self.process_single_object(
                    frame,
                    track_id,
                    track_bbox,
                    current_frame_id,
                    current_frame_time_stamp,
                    self.behavior_cls,
                    self.behavior_prob,
                    time_tag)

        self.update_tracking(self.track_bboxes, frame_tag)
        self.track_bboxes = None
        return frame, self.frame_coordinates, self.data_sample

    def init_model(self):
        self.face = YOLO(self.cfgs.MODEL.FACE.trt_engine, task='detect')
        print('loads face detect model')
        self.logger.info('loads face detect model')

        new_instance_name = f'mmpretrain-{datetime.datetime.now()}'
        DefaultScope.get_instance(new_instance_name, scope_name='mmpretrain')
        self.faceid = FaceidInferencerTRT(
            model_cfg=self.cfgs.MODEL.FACEID.cfg,
            image_encoder=self.cfgs.MODEL.FACEID.trt_engine,
            prototype=self.cfgs.MODEL.FACEID.prototype,
            prototype_cache=self.cfgs.MODEL.FACEID.prototype_cache)
        print('loads face id model')
        self.logger.info('loads face id model')
        self.tracker = YOLO(self.cfgs.MODEL.BODY.trt_engine, task='detect')
        if self.cfgs.MODEL.BODY.with_reid:
            # input_size = (self.frame_shape[0], self.frame_shape[1])
            if self.cfgs.MODEL.BODY.reid_encoder is not None:
                ckpt = torch.load(self.cfgs.MODEL.BODY.reid_encoder)
                input_size = ckpt['input_size']
            else:
                input_size = (224, 224)
            self.reid_encoder = ReIDEncoder(input_size=input_size,
                                            # weights_path=self.cfgs.MODEL.BODY.r_e_trt_engine)
                                            weights_path=self.cfgs.MODEL.BODY.reid_encoder)
            print('loads reid model')
            self.logger.info('loads reid model')
        else:
            self.reid_encoder = None

        self.behavior = init_recognizer(
            self.cfgs.MODEL.BEHAVIOR.cfg,
            self.cfgs.MODEL.BEHAVIOR.weight,
            device=self.DEVICE)
        print('loads behavior model')
        self.logger.info('loads behavior model')
        behavior_cfg = Config.fromfile(self.cfgs.MODEL.BEHAVIOR.cfg)
        test_pipeline_cfg = behavior_cfg.test_pipeline
        self.behavior_test_pipeline = Compose(test_pipeline_cfg)

        self.pose, self.dataset_meta = self.pose_model(
            model_cfg=self.cfgs.MODEL.POSE.cfg,
            deploy_cfg=self.cfgs.MODEL.POSE.deploy_cfg,
            backend_files=[self.cfgs.MODEL.POSE.trt_engine],
            device=self.DEVICE)
        print('loads pose model')
        self.logger.info('loads pose model')

    def Pose(self, frame_paths, det_results: List[np.ndarray]):

        new_instance_name = f'mmpose-{datetime.datetime.now()}'
        DefaultScope.get_instance(new_instance_name, scope_name='mmpose')

        results = []
        data_samples = []
        # print('Performing Human Pose Estimation for each frame')

        pose_data_samples = self.inference_topdown(
            self.pose,
            frame_paths,
            det_results[0][..., :4],
            bbox_format='xyxy',
            model_cfg=self.cfgs.MODEL.POSE.cfg,)
        self.logger.info('Pose estimate')
        pose_data_sample = merge_data_samples(pose_data_samples)
        pose_data_sample.dataset_meta = self.dataset_meta
        # make fake pred_instances
        if not hasattr(pose_data_sample, 'pred_instances'):
            num_keypoints = self.pose.dataset_meta['num_keypoints']
            pred_instances_data = dict(
                keypoints=np.empty(shape=(0, num_keypoints, 2)),
                keypoints_scores=np.empty(shape=(0, 17), dtype=np.float32),
                bboxes=np.empty(shape=(0, 4), dtype=np.float32),
                bbox_scores=np.empty(shape=(0), dtype=np.float32))
            pose_data_sample.pred_instances = InstanceData(
                **pred_instances_data)

        poses = pose_data_sample.pred_instances.to_dict()
        poses['track_bboxes'] = det_results[0]
        results.append(poses)
        data_samples.append(pose_data_sample)

        return results, data_samples

    @override
    def pose_model(self, *args, **kwargs,):
        # 从 kwargs 中获取需要的参数
        deploy_cfg = kwargs.get('deploy_cfg')
        backend_files = kwargs.get('backend_files')
        device = kwargs.get('device', 'cuda:0')
        model_cfg = kwargs.get('model_cfg')

        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
        self.input_shape = get_input_shape(deploy_cfg)
        self.task_processor = build_task_processor(model_cfg, deploy_cfg, str(device))

        model = self.task_processor.build_backend_model(backend_files,
                                                        self.task_processor.update_data_preprocessor)

        dataset_meta = self.dataset_meta_from_config(model_cfg, dataset_mode='train')

        self.pose_pipeline = Compose(model_cfg.test_dataloader.dataset.pipeline)

        return model, dataset_meta

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

    @override
    def inference_topdown(self,
                          *args, **kwargs) -> List[PoseDataSample]:
        """Inference image with a top-down pose estimator.

        Args:
            model (nn.Module): The top-down pose estimator
            img (np.ndarray | str): The loaded image or image file to inference
            bboxes (np.ndarray, optional): The bboxes in shape (N, 4), each row
                represents a bbox. If not given, the entire image will be regarded
                as a single bbox area. Defaults to ``None``
            bbox_format (str): The bbox format indicator. Options are ``'xywh'``
                and ``'xyxy'``. Defaults to ``'xyxy'``

        Returns:
            List[:obj:`PoseDataSample`]: The inference results. Specifically, the
            predicted keypoints and scores are saved at
            ``data_sample.pred_instances.keypoints`` and
            ``data_sample.pred_instances.keypoint_scores``.
        """
        model = args[0] if args else kwargs.get('model')
        img = args[1] if len(args) > 1 else kwargs.get('img')
        bboxes = args[2] if len(args) > 2 else kwargs.get('bboxes')
        bbox_format = args[3] if len(args) > 3 else kwargs.get('bbox_format', 'xyxy')
        model_cfg = kwargs.get('model_cfg')

        if isinstance(model_cfg, (str, Path)):
            model_cfg = Config.fromfile(model_cfg)
        scope = model_cfg.get('default_scope', 'mmpose')
        if scope is not None:
            init_default_scope(scope)
        pipeline = self.pose_pipeline

        if bboxes is None or len(bboxes) == 0:
            # get bbox from the image size
            if isinstance(img, str):
                w, h = Image.open(img).size
            else:
                h, w = img.shape[:2]

            bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
        else:
            if isinstance(bboxes, list):
                bboxes = np.array(bboxes)

            assert bbox_format in {'xyxy', 'xywh'}, \
                f'Invalid bbox_format "{bbox_format}".'

            if bbox_format == 'xywh':
                bboxes = bbox_xywh2xyxy(bboxes)

        # construct batch data samples
        data_list = []
        for bbox in bboxes:
            if isinstance(img, str):
                data_info = dict(img_path=img)
            else:
                data_info = dict(img=img)
            data_info['bbox'] = bbox[None]  # shape (1, 4)
            data_info['bbox_score'] = np.ones(1, dtype=np.float32)  # shape (1,)
            data_info.update(self.dataset_meta)
            data_list.append(pipeline(data_info))

        if data_list:
            # collate data list into a batch, which is a dict with following keys:
            # batch['inputs']: a list of input images
            # batch['data_samples']: a list of :obj:`PoseDataSample`
            batch = pseudo_collate(data_list)
            with torch.no_grad():
                results = model.test_step(batch)
        else:
            results = []

        return results