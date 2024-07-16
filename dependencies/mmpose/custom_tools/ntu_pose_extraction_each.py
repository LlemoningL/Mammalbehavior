# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import mmcv
import abc
import argparse
from tempfile import TemporaryDirectory
from ultralytics import YOLO
import mmengine
from mmpose.apis import inference_topdown, init_model
import os.path as osp
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from mmengine.structures import InstanceData
from mmengine.utils import track_iter_progress
from custom_example.mm_groundtruth import mkgdjson
from pathlib import Path as p
import re

def frame_extract(video_path: str,
                  short_side: Optional[int] = None,
                  out_dir: str = './tmp'):
    """Extract frames given video_path.

    Args:
        video_path (str): The video path.
        short_side (int): Target short-side of the output image.
            Defaults to None, means keeping original shape.
        out_dir (str): The output directory. Defaults to ``'./tmp'``.
    """
    # Load the video, extract frames into OUT_DIR/video_name
    target_dir = osp.join(out_dir, osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    assert osp.exists(video_path), f'file not exit {video_path}'
    vid = cv2.VideoCapture(video_path)
    size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        if short_side is not None:
            if new_h is None:
                h, w, _ = frame.shape
                new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))
            frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames, size



def split_pose_result_to_each(pose_results_list):
    num_person = max([len(x['keypoints']) for x in pose_results_list])
    pose_results_list_each = [[] for j in range(num_person)]
    for idx, d in enumerate(pose_results_list):
        temp_ecah_list = [[] for j in range(num_person)]
        if len(d['keypoints']) < num_person:
            frame_person = len(d['keypoints'])
        else:
            frame_person = num_person
        for i in range(frame_person):
            temp_dict = dict(
                bboxes=np.zeros((1, 4), dtype=np.float16),
                keypoints_visible=np.zeros((1, 17), dtype=np.float16),
                keypoints=np.zeros((1, 17, 2), dtype=np.float16),
                bbox_scores=np.zeros((1), dtype=np.float16),
                keypoint_scores=np.zeros((1, 17), dtype=np.float16))
            temp_dict['bboxes'][0] = d['bboxes'][i]
            temp_dict['keypoints_visible'][0] = d['keypoints_visible'][i]
            temp_dict['keypoints'][0] = d['keypoints'][i]
            temp_dict['bbox_scores'] = d['bbox_scores'][i]
            temp_dict['keypoint_scores'][0] = d['keypoint_scores'][i]
            temp_ecah_list[i].append(temp_dict)
            pose_results_list_each[i].extend(temp_ecah_list[i])
    pose_results_list_each = [[x, pose_results_list_each[x]] for x in range(len(pose_results_list_each))]
    return pose_results_list_each


def pose_inference(pose_model,
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
    print('Performing Target Pose Estimation for each frame')
    for f, d in track_iter_progress(list(zip(frame_paths, det_results))):
        pose_data_samples: List[PoseDataSample] \
            = inference_topdown(model, f, d[..., :4], bbox_format='xyxy')
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



def pose_inference_with_align(pose_model, frame_paths, det_results):
    # filter frame without det bbox
    kp_sc_list = [[], []]
    det_results = [
        frm_dets for frm_dets in det_results if frm_dets.shape[0] > 0
    ]
    pose_results, _ = pose_inference(pose_model, frame_paths, det_results)
    num_person = [len(x['keypoints']) for x in pose_results]
    if not len(num_person) > 0:
        kp_sc_list[0], kp_sc_list[1] = 0, 0
        return kp_sc_list[0], kp_sc_list[1]
    pose_results_ecah = split_pose_result_to_each(pose_results)
    for i in pose_results_ecah:
        # align the num_person among frames
        num_persons = max([pose['keypoints'].shape[0] for pose in i[1]])
        num_points = i[1][0]['keypoints'].shape[1]
        num_frames = len(i[1])
        keypoints = np.zeros((num_persons, num_frames, num_points, 2),
                             dtype=np.float32)
        scores = np.zeros((num_persons, num_frames, num_points), dtype=np.float32)

        for f_idx, frm_pose in enumerate(i[1]):
            frm_num_persons = frm_pose['keypoints'].shape[0]
            for p_idx in range(frm_num_persons):
                keypoints[p_idx, f_idx] = frm_pose['keypoints'][p_idx]
                scores[p_idx, f_idx] = frm_pose['keypoint_scores'][p_idx]
        kp_sc_list[0].append(keypoints)
        kp_sc_list[1].append(scores)

    return kp_sc_list[0], kp_sc_list[1]


def ntu_pose_extraction(det_model, pose_model, vid, json_file):
    tmp_dir = TemporaryDirectory()
    frame_paths, _, size = frame_extract(vid, out_dir=tmp_dir.name)
    track_results = []
    for i in frame_paths:
        track_result = det_model.track(i, conf=0.5, device='cuda:0', persist=True)  # , imgsz=(1920, 1920)
        for result in track_result:
            track_bboxes = result.boxes.xyxy.cpu().numpy()
            track_results.append(track_bboxes)
    patter = re.compile('[a-z][a-z]+')
    str_ = p(vid).name.lower()
    temp = patter.findall(str_)[0]
    label = json_file['categories'].index(temp.title())
    keypoints, scores = pose_inference_with_align(pose_model, frame_paths, track_results)
    if keypoints == 0 or scores == 0:
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
    tmp_dir.cleanup()
    return annos


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Pose Annotation for a single NTURGB-D video')

    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    def extract(path, cls):
        args = abc.abstractproperty()
        args.det_checkpoint = '/home/ztx/lj/Animalbehavior/ultralytics/runs/train/qlzoo_10sp_bodydetect20231117142/weights/best.pt'  # noqa: E501

        # args.pose_config = '/home/ztx/lj/Animalbehavior/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/primate_td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'  # noqa: E501
        args.pose_config ='/home/ztx/lj/Animalbehavior/mmpose/configs/animal_2d_keypoint/topdown_heatmap/ap10k/fourleg_td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.py'
        # args.pose_checkpoint = '/home/ztx/lj/Animalbehavior/mmpose/custom_tools/best_coco_AP_epoch_110_primatepose.pth'  # noqa: E501
        args.pose_checkpoint ='/home/ztx/lj/Animalbehavior/mmpose/custom_tools/official_checkpoint/4legposebest_coco_AP_epoch_160.pth'
        global_args = parse_args()
        args.device = global_args.device


        path = p(path)
        json_file = mkgdjson(p(path), cls)
        pose_model = init_model(args.pose_config, args.pose_checkpoint, args.device)

        for i in path.iterdir():
            for j in i.iterdir():
                det_model = YOLO(args.det_checkpoint)
                pklname = str(j.stem)
                pklouput = os.getcwd() + f'/extract_split2/pkl_{cls}_{path.parent.stem}_A800_{path.stem}'
                annos = ntu_pose_extraction(det_model, pose_model, str(j), json_file)
                if len(annos) > 0:
                    for idx, k in enumerate(annos):
                        if k['keypoint'].shape[1] >= 60:
                            pkl_name = f'{pklouput}/{pklname}-{idx}.pkl'
                            mmengine.dump(k, pkl_name)

# path1 = '/media/ztx/Elements/2023.10 Article data supplements processed data/behavior/dataset_collection/primate'
path2 = '/home/ztx/lj/4tdisk/animalbehavior/behavior/dataset_collection/TAKIN'
# cls1 = 'primate'
cls2 = '4leg'

# d = dict(cls1=path1, cls2=path2)
d = dict(cls2=path2)
for k, v in d.items():
    cls = k
    path = v
    for i in p(path).iterdir():
        for j in i.iterdir():
            extract(j, cls2)




