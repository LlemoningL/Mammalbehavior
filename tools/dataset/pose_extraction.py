import re
import os.path as osp
import cv2
import json
import torch
import queue
import numpy as np
import mmengine
from ultralytics import YOLO
from mmpose.apis import init_model
from typing import List, Optional, Union
from mmengine.structures import InstanceData
from mmengine.utils import track_iter_progress
from pathlib import Path
from tqdm import tqdm


class PoseExtraction:

    def __init__(self,
                 det_mdoel,
                 pose_config,
                 pose_model,
                 behavior_label,
                 outroot=None,
                 auto_combine=True,
                 device='cuda:0',
                 Queue_size=10000,
                 split_path='extracted_split',
                 combi_path='extracted_combi'):
        self.det_model = det_mdoel
        self.pose_config = pose_config
        self.pose_model = pose_model
        self.target_type = None
        self.auto_combine = auto_combine
        self.combine_list = []
        if outroot is not None:
            self.output = Path(outroot) / split_path
            self.combi_out = Path(outroot) / combi_path
        else:
            self.output = Path(split_path)
            self.combi_out = Path(combi_path)

        self.behavior_label = behavior_label
        self.device = device
        self.Posemodel = self.init_pose_model()
        self.frames = queue.Queue(maxsize=Queue_size)

    def init_pose_model(self):
        pose_model = init_model(self.pose_config, self.pose_model, device=self.device)
        return pose_model

    def init_det_mdoel(self):
        det_model = YOLO(self.det_model)
        return det_model

    def label(self):
        if self.target_type in ["Primates", "Artiodactyla", "Carnivora", "Perissodactyla"]:
            with open(self.behavior_label, 'r') as f:
                self.behavior_label_ = json.load(f)
            return self.behavior_label_[self.target_type.title()]['categories']
        else:
            raise ValueError(f"Unrecognized behavior type '{self.target_type}', must be in "
                             f"'Primates', 'Artiodactyla', 'Carnivora', 'Perissodactyla'\n"
                             f"'Primates' for 'Golden Snub-nosed Monkey', 'Ring-tailed Lemur', 'Hamadryas Baboon'\n"
                             f"'Artiodactyla' for 'Takin', 'Gnu', 'Lechwe'\n"
                             f"'Carnivora' for 'Tiger', 'Black Bear', 'Brown Bear'\n"
                             f"'Perissodactyla' for 'Zebra")

    def frame_extract(self, video_path: str):

        vid = cv2.VideoCapture(video_path)
        size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        tag = 0
        print(f'extracting frames from video: {video_path}')
        while vid.isOpened():
            flag, frame = vid.read()
            if not flag:
                break
            if tag >= 5:
                self.frames.put(frame)
                tag = 0
            tag += 1

        return size

    def pose_inference_with_align(self, pose_model, frame_paths, det_results):
        # pose_results_ecah = {}
        kp_sc_list = [[], []]
        for k, v in det_results.items():
            pose_results, _ = self.pose_inference(pose_model, frame_paths, v)
            num_persons = 1
            num_points = pose_results[0]['keypoints'].shape[1]
            num_frames = len(pose_results)
            keypoints = np.zeros((num_persons, num_frames, num_points, 2),
                                 dtype=np.float32)
            scores = np.zeros((num_persons, num_frames, num_points), dtype=np.float32)

            for f_idx, frm_pose in enumerate(pose_results):
                frm_num_persons = frm_pose['keypoints'].shape[0]
                for p_idx in range(frm_num_persons):
                    keypoints[p_idx, f_idx] = frm_pose['keypoints'][p_idx]
                    scores[p_idx, f_idx] = frm_pose['keypoint_scores'][p_idx]
            kp_sc_list[0].append(keypoints)
            kp_sc_list[1].append(scores)

        return kp_sc_list[0], kp_sc_list[1]

    def pose_inference(self,
                       pose_model,
                       frame_paths: List[str],
                       det_results: List[np.ndarray],
                       ) -> tuple:
        """Perform Top-Down pose estimation.

        Args:
            pose_config (Union[str, :obj:`Path`, :obj:`mmengine.Config`,
                :obj:`torch.nn.Module`]): Pose config file path or
                pose model object. It can be a :obj:`Path`, a config object,
                or a module object.
            pose_checkpoint: Checkpoint path/url.
            frame_paths (List[str]): The paths of frames to do pose inference.
            det_results (List[np.ndarray]): List of detected human boxes.
            device (Union[str, torch.device]): The desired device of returned
                tensor. Defaults to ``'cuda:0'``.

        Returns:
            List[List[Dict[str, np.ndarray]]]: List of pose estimation results.
            List[:obj:`PoseDataSample`]: List of data samples, generally used
                to visualize data.
        """
        try:
            from mmpose.apis import inference_topdown, init_model
            from mmpose.structures import PoseDataSample, merge_data_samples
        except (ImportError, ModuleNotFoundError):
            raise ImportError('Failed to import `inference_topdown` and '
                              '`init_model` from `mmpose.apis`. These apis '
                              'are required in this inference api! ')
        model = pose_model

        results = []
        data_samples = []
        # print('Performing Human Pose Estimation for each frame')
        for f, d in track_iter_progress(list(zip(frame_paths, det_results))):
            pose_data_samples: List[PoseDataSample] \
                = inference_topdown(model, f, [d], bbox_format='xyxy')
            pose_data_sample = merge_data_samples(pose_data_samples)
            pose_data_sample.dataset_meta = model.dataset_meta
            # make fake pred_instances
            if not hasattr(pose_data_sample, 'pred_instances'):
                num_keypoints = model.dataset_meta['num_keypoints']
                pred_instances_data = dict(
                    keypoints=np.empty(shape=(0, num_keypoints, 2)),
                    keypoints_scores=np.empty(shape=(0, 17), dtype=np.float32),
                    bboxes=np.empty(shape=(0, 4), dtype=np.float32),
                    bbox_scores=np.empty(shape=(0), dtype=np.float32))
                pose_data_sample.pred_instances = InstanceData(
                    **pred_instances_data)

            poses = pose_data_sample.pred_instances.to_dict()
            results.append(poses)
            data_samples.append(pose_data_sample)

        return results, data_samples

    def pose_extract(self, det_model, pose_model, vid):
        size = self.frame_extract(vid)
        track_results = {}
        frames = []
        print(f'detecting frames from video: {vid}')
        while not self.frames.empty():
            frame = self.frames.get()
            track_result = det_model.track(
                source=frame,
                device=self.device,
                verbose=True,
                persist=True,
                imgsz=(640, 640)
            )  # show=True,
            for result in track_result:
                trackid = result.boxes.id
                if trackid is not None:
                    track_bboxes = result.boxes.xyxy.cpu().numpy()
                    if track_bboxes.shape[1] < 4:
                        continue
                    for id_, box in zip(trackid, track_bboxes):
                        id_ = int(id_)
                        if id_ not in track_results:
                            track_results[id_] = []
                            track_results[id_].append(box)
                            frames.append(frame)
                        else:
                            track_results[id_].append(box)
                            frames.append(frame)
                # track_results.append(track_bboxes)
        patter = re.compile('[a-z][a-z]+')
        str_ = Path(vid).parent.name.lower()
        temp = patter.findall(str_)[0]
        label = self.label().index(temp.title())
        keypoints, scores = self.pose_inference_with_align(pose_model, frames, track_results)
        if len(keypoints) == 0 or len(scores) == 0:
            annos = []
            return annos
        annos = []
        for k, s in zip(keypoints, scores):
            anno = dict()
            anno['keypoint'] = k
            anno['keypoint_score'] = s
            anno['frame_dir'] = osp.splitext(osp.basename(vid))[0]
            anno['img_shape'] = (size[1], size[0])
            anno['original_shape'] = (size[1], size[0])
            anno['total_frames'] = k.shape[1]
            anno['label'] = int(label)
            annos.append(anno)
        return annos

    def extract(self, video_path: str, specie: str, save_dir=None, target_type=None):
        self.target_type = target_type
        Detmodel = self.init_det_mdoel()
        pkl_name = f'{Path(video_path).parent.stem}_{str(Path(video_path).stem)}'
        if save_dir is not None:
            pkl_outdir = self.output / specie / save_dir
            # self.pkl_conmbi_out = self.combi_out / specie /save_dir
        else:
            pkl_outdir = self.output / specie
            # self.pkl_conmbi_out = self.combi_out / save_dir
        annos = self.pose_extract(Detmodel, self.Posemodel, str(video_path))
        if len(annos) > 0:
            for idx, k in enumerate(annos):
                pkl_name_save = pkl_outdir / f'{pkl_name}_{idx}.pkl'
                mmengine.dump(k, pkl_name_save)
        if self.auto_combine:
            self.combine_list.append(pkl_outdir)
        # self.combin_pkl(pkl_outdir, (pkl_conmbi_out / f'{pkl_outdir.stem}.pkl'))

    def combine_pkl(self, path, pklname):
        result = []
        path = Path(path).rglob('**/*.pkl')
        for d in tqdm(path):
            if d.suffix == '.pkl':
                content = mmengine.load(d)
                result.append(content)
        mmengine.dump(result, pklname)

    def combine(self, path=None, pklname='result.pkl'):
        if self.auto_combine:
            for d in self.combine_list:
                if d.parent.stem not in ['train', 'val', 'test']:
                    pkl_save_name = self.combi_out / d.stem / f'{d.stem}.pkl'
                else:
                    pkl_save_name = self.combi_out / d.parent.stem / d.stem / f'{d.parent.stem}_{d.stem}.pkl'
                self.combine_pkl(d, pkl_save_name,)
        elif path is not None:
            self.combine_pkl(path, pklname)


if __name__ == '__main__':
    det_mdoel = 'path/to/detect_mdoel_weight'  # recommend using your trained YOLO model for body detection
    pose_config = 'path/to/pose_config'  # recommend using your pose estimate config
    pose_weight = 'path/to/pose_model_weight'  # recommend using your trained pose estiamte model
    target_type = 'Primates'  # choose ont in ["Primates", "Artiodactyla", "Carnivora", "Perissodactyla"]
    behavior_label = '../behavior_label.json'  # default behavior label file path
    output = None  # optional, can be None or specific directory path
    pose_extr = PoseExtraction(det_mdoel,
                               pose_config,
                               pose_weight,
                               target_type,
                               behavior_label,
                               output)
    # for single video
    video = 'path/to/video'
    specie = 'gsm'  # Golden Snub-nosed Monkey
    pose_extr.extract(video, specie)
    pose_extr.combine()

'''    
    # for multiple videos with same type of behavior in one directory
    # you should make sure that the directory name like 'Feeding' or 'feeding', the type of behavior, 
    # and the video file name like 'Feeding000001.mp4' or 'feeding000001.mp4', the label of video file.  
    video_dir = Path('path/to/video/dir')
    specie = 'gsm'
    for i in video_dir.iterdir():
        pose_extr.extract(str(i), specie)
    pose_extr.combine()
'''