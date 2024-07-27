import random
import pandas as pd
import seaborn as sns
import cv2
import matplotlib.font_manager as fm
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont

plt.rcParams['font.sans-serif'] = ['AR PL UMing CN']

from mmpretrain import ImageRetrievalInferencer
from mmengine.model import BaseModel
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from mmcv.image import imread
from mmengine.config import Config
from mmengine.dataset import BaseDataset, Compose, default_collate

from mmpretrain.registry import TRANSFORMS
from mmpretrain.structures import DataSample
# from .base import BaseInferencer, InputType, ModelType
from abc import abstractmethod
from math import ceil
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import default_collate
from mmengine.fileio import get_file_backend
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint

from mmpretrain.structures import DataSample
from mmpretrain.utils import track
# from ..dependencies.mmpretrain.mmpretrain.apis.model import get_model, list_models
# from ..dependencies.mmpretrain.mmpretrain.apis.model import get_model, list_models


ModelType = Union[BaseModel, str, Config]
InputType = Union[str, np.ndarray, list]
class FaceidInferencer(ImageRetrievalInferencer):
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
            model: ModelType,
            prototype,
            prototype_cache=None,
            prepare_batch_size=8,
            pretrained: Union[bool, str] = True,
            device: Union[str, torch.device, None] = None,
            **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            prototype=prototype,
            prototype_cache=prototype_cache,
            pretrained=pretrained,
            device=device,
            **kwargs)

        self.prototype_dataset = self._prepare_prototype(
            prototype, prototype_cache, prepare_batch_size)

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
        # The terminal output progress bar code has been rewritten.
        for data in inputs:
            preds.extend(self.forward(data, **forward_kwargs))
        visualization = self.visualize(ori_inputs, preds, **visualize_kwargs)
        results = self.postprocess(preds, visualization, return_datasamples,
                                   **postprocess_kwargs)
        return results


def plot_one_box(x, img, color=None, label=None, line_thickness=3, padding=8):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 2)  # font thickness tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 1.1, thickness=tf)[0]  # fontScale=tl / 3
        font_size = t_size[1]
        font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='AR PL UMing CN')), font_size) #支持中英文字体 ‘AR PL UMing CN‘
        bbox = font.getbbox(label)
        t_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        c2 = c1[0] + t_size[0] + padding, c1[1] - t_size[1] - padding
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_PIL)
        draw.text((c1[0], c2[1] - 2), label, fill=(255, 255, 255), font=font)

        return cv2.cvtColor(np.array(img_PIL), cv2.COLOR_RGB2BGR)


def rythym_budget(info, seg=None, poselist=None,
                  savepath=None, videoname=None,
                  videofps=60):
    for j, k in info.items():
        if int(j) == 0:
            continue
        elif len(k) < round(seg) / 2:
            print(f'id [{j + 1}]: insufficient data, ignore')
        else:

            df_ = pd.DataFrame()
            budget_list = []
            p_list = []
            for pl in poselist:
                pl_num = 0
                frame_list = []
                pl_num_list = []
                t_stamp_list = []
                init_frame_ = 0
                for frame, t_stemp, pose_num in zip(k[1:][1], k[1:][2], k[1:][poselist.index(pl) + 3]):
                    frame = int(frame)
                    pose_num = int(pose_num)
                    pl_num = pl_num + pose_num
                    if t_stemp % seg == 0:
                        if frame == init_frame_:
                            init_frame_ = frame
                        else:
                            frame_list.append(frame)
                            pl_num_list.append(pl_num)
                            t_stamp_list.append(t_stemp)
                            pl_num = 0

                title_df = pd.DataFrame([pl])
                pl_num_list = pd.DataFrame(pl_num_list)
                pl_num_list = pd.concat([title_df, pl_num_list], axis=0)
                df_ = pd.concat([pl_num_list, df_], axis=1)

            title_df = pd.DataFrame(['t_stamp'])
            frame_list = pd.DataFrame(t_stamp_list)
            frame_list = pd.concat([title_df, frame_list], axis=0)
            df_ = pd.concat([frame_list, df_], axis=1)

            df_ = df_.reset_index(drop=True)
            title_list = df_.iloc[0]
            df_.columns = title_list
            df_.drop([0], inplace=True)
            random.seed(30)  # 18 30
            marker_style = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p',
                            'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd']
            line_style = ['-', ':', '--', '-.']
            for i in range(len(df_.columns) - 1):
                try:
                    plt.plot(df_.iloc[:, 0], df_.iloc[:, i + 1], linewidth='1.5',
                             linestyle=line_style[i if i < 4 else 3],
                             marker=marker_style[i if i < 19 else 19], label=df_.columns[i + 1])
                    plt.title(f'{videoname}_rythym_id-{j}')
                    plt.xlabel(f'Time period(s) ')
                    plt.ylabel('Frequency of behavior')
                    plt.legend()
                    plt.tight_layout()
                # plt.show()
                except:
                    print('Failure to plot the time rythym chart, missing values, '
                          'or Nah in the data could be the cause ')

                tmp_sum = df_[[df_.columns[i + 1]]].sum()
                budget_list.append(tmp_sum.values[0])
                p_list.append(tmp_sum.index[0])
            # plt.close()
            rythym_savepath = savepath / f'{videoname}_rythym_id-{j}.png'
            plt.savefig(rythym_savepath)
            # plt.show()
            plt.close()
            sum_ = sum(budget_list)
            pie_ = []
            for num in budget_list:
                pie_.append(num / sum_)
            explode_ = []
            for i in range(len(pie_)):
                explode_.append(0.01)
            max_exp = max(pie_)
            max_id = pie_.index(max_exp)
            explode_[max_id] = 0.03
            try:
                plt.pie(pie_, explode=explode_, labels=p_list, autopct='%1.1f%%')
                plt.title(f'{videoname}_Time budget_id-{j}')
                plt.tight_layout()
                plt.axis('equal')
                # plt.close()
                budget_savepath = savepath / f'{videoname}_Time_budget_id-{j}.png'
                plt.savefig(budget_savepath)
                plt.close()
            except:
                print('Failure to plot the time budget pie chart, missing values, '
                      'or Nah in the data could be the cause ')


def get_color(seeds):
    palette = sns.color_palette('hls', 200)
    bbox_colors = []
    for seed in seeds:
        random.seed(seed)
        bbox_color = random.choice(palette)
        bbox_color = [int(255 * c) for c in bbox_color][::-1]
        bbox_colors.append(tuple(bbox_color))

    return bbox_colors


def visualize_frame(visualizer, frames, data_samples):
    for d in data_samples:
        visualizer.add_datasample(
            'result',
            frames,
            data_sample=d,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=False,
            show=False,
            wait_time=0,
            out_file=None,
            kpt_thr=0.3)
        frames = visualizer.get_image()

    return frames


def show_img(vis_img, waitkey=1, name='Image'):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, vis_img)
    cv2.waitKey(waitkey)


def split_xyxy(cord_xyxy):

    if isinstance(cord_xyxy, np.ndarray):
        if cord_xyxy.shape[0] == 1:
            x1, y1, x2, y2 = int(cord_xyxy[:, 0]), \
                int(cord_xyxy[:, 1]), \
                int(cord_xyxy[:, 2]), \
                int(cord_xyxy[:, 3])
        else:
            x1, y1, x2, y2 = int(cord_xyxy[0]), \
                int(cord_xyxy[1]), \
                int(cord_xyxy[2]), \
                int(cord_xyxy[3])

    elif isinstance(cord_xyxy, torch.Tensor):
        x1, y1, x2, y2 = int(cord_xyxy[0].item()), \
            int(cord_xyxy[1].item()), \
            int(cord_xyxy[2].item()), \
            int(cord_xyxy[3].item())
    else:
        raise TypeError(f'Unsupported type{type(cord_xyxy)}')

    return x1, y1, x2, y2


# def calculate_box_dis(big_box, small_box):
#     # Calculate the top center coordinates of the large box
#     if isinstance(big_box, torch.Tensor):
#         big_box = big_box.cpu().numpy()
#     if isinstance(small_box, torch.Tensor):
#         small_box = small_box.cpu().numpy()
#     if isinstance(big_box, np.ndarray):
#         top_center = big_box[:2] + big_box[2:] / 2
#         # Calculate the center point coordinates of each small box
#         center1 = (small_box[0] + small_box[2]) / 2, (small_box[1] + small_box[3]) / 2
#         # Calculate the distance from the center of the small box to the top center of the large box
#         distance = np.sqrt((center1[0] - top_center[0]) ** 2 + (center1[1] - top_center[1]) ** 2)
#     else:
#         raise ValueError(f'{big_box} must be numpy.ndarray, but got {type(big_box)}')
#
#     return distance


# def select_box(box_list):
#     min_dis = 6000
#     for i in box_list:
#         min_d = i[0]
#         if min_d < min_dis:
#             min_dis = min_d
#             box_coord = i[1]
#
#     return box_coord
# def bind_faceid_trackid(face_name: str, track_id: int, faceid_trackid: dict):
#     """
#     Args:
#         face_name: face id
#         track_id: track id
#         faceid_trackid: dictionary of bound face id and track id list
#
#     Returns: face id and dictionary of bound face id and track id
#     """
#     track_idlist = list()
#     # check first time when get face id and faceid_trackid is empty
#     if not len(faceid_trackid) > 0 and face_name is not None:
#         faceid_trackid[face_name] = list()
#         faceid_trackid[face_name].append(track_id)
#     # get track id no duplicates
#     if len(faceid_trackid) > 0:
#         for k, v in faceid_trackid.items():
#             track_idlist.extend(v)
#     # face id is not empty(not None)
#     if face_name is not None:
#         # new face id and track id
#         if face_name not in faceid_trackid and track_id not in track_idlist:
#             faceid_trackid[face_name] = list()
#             faceid_trackid[face_name].append(track_id)
#         # exist face id and new track id
#         elif face_name in faceid_trackid and track_id not in track_idlist:
#             faceid_trackid[face_name].append(track_id)
#         # new face id and exist track id, keep nwe face id
#         elif face_name not in faceid_trackid and track_id in track_idlist:
#             faceid_trackid[face_name] = list()
#         # otherwise do this
#         else:
#             # use track id to find bound face id if exist
#             for k, v in faceid_trackid.items():
#                 if track_id in track_idlist and track_id in v:
#                     face_name = k
#     # face id is empty, use track id to find bound face id if exist
#     else:
#         for k, v in faceid_trackid.items():
#             if track_id in track_idlist and track_id in v:
#                 face_name = k
#                 # return face_name, faceid_trackid
#
#     # when face id is empty(is None), set face id as '_'
#     if face_name is None:
#         face_name = '_'
#
#     return face_name, faceid_trackid


def bind_faceid_trackid(face_name: str, track_id: int, faceid_trackid: dict) -> tuple: # renew
    """
    绑定面部ID和追踪ID。

    Args:
    face_name: 面部ID
    track_id: 追踪ID
    faceid_trackid: 面部ID和追踪ID列表的字典

    Returns:
    tuple: (面部ID, 更新后的faceid_trackid字典)
    """

    # 如果face_name为None，尝试通过track_id找到对应的face_name
    if face_name is None:
        for k, v in faceid_trackid.items():
            if track_id in v:
                return k, faceid_trackid
        return '_', faceid_trackid  # 如果没找到对应的face_name，返回'_'

    # 获取所有已存在的track_id
    all_track_ids = [tid for ids in faceid_trackid.values() for tid in ids]

    # 处理face_name不为None的情况
    if face_name not in faceid_trackid:
        faceid_trackid[face_name] = []

    if track_id not in all_track_ids:
        faceid_trackid[face_name].append(track_id)
    elif track_id not in faceid_trackid[face_name]:
        # 如果track_id已存在于其他face_name下，需要移除
        for k, v in faceid_trackid.items():
            if track_id in v:
                v.remove(track_id)
        faceid_trackid[face_name].append(track_id)

    return face_name, faceid_trackid


def line_info(face_name,
              track_id,
              current_frame_id,
              current_frame_time_stamp,
              behavior_cls,
              behavior_label):
    if behavior_cls != '':
        temp_array = [face_name,
                      int(track_id),
                      int(current_frame_id),
                      int(current_frame_time_stamp)]
        temp_array.extend([0 for i in range(len(behavior_label))])
        temp_array[behavior_label.index(behavior_cls.title()) + 3] = 1
    else:
        temp_array = None

    return temp_array


def vis_box(img, coordinate_dict, id_bbox_colors, line_thickness, padding):


    for k, v in coordinate_dict.items():
        track_id = k
        body_coord = v[0]
        face_result = v[1]
        label_text = v[2]
        color = tuple(id_bbox_colors[track_id]) if track_id in id_bbox_colors else None
        body_x1, body_y1, body_x2, body_y2 = split_xyxy(body_coord)
        body_area = img[body_y1:body_y2, body_x1:body_x2]

        if face_result[0].boxes.shape[0] != 0:
            face_xyxy = face_result[0].boxes.xyxy[0]
            plot_one_box(face_xyxy,
                         body_area,
                         color=color,
                         line_thickness=line_thickness,
                         padding=padding)
            # img[body_y1:body_y2, body_x1:body_x2] = body_area

        img = plot_one_box(body_coord,
                           img,
                           label=label_text,
                           color=color,
                           line_thickness=3)
    return img