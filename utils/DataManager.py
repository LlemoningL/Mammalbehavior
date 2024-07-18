import json
import pandas as pd
from utils.util import bind_faceid_trackid, line_info
from pathlib import Path


class DataManager:
    def __init__(self, output_path, traget_type, behavior_label):
        self.target_type = traget_type
        self.behavior_label = behavior_label
        self.pose_results = []
        self.faceid_trackid = {}
        self.Frameinfo = self.make_data_dict()
        self.label_text = {}
        self.frame_coordinates = {}
        self.csv_path = Path(output_path) / 'frame_info.csv'
        self.is_first_save = True

    def update_pose_result(self, pose_result):
        if pose_result[0]['track_bboxes'].shape[1] > 4:
            self.pose_results.extend(pose_result)

    def update_faceid_trackid(self, face_name, track_id):
        face_name, self.faceid_trackid = bind_faceid_trackid(face_name,
                                                             track_id,
                                                             self.faceid_trackid)

        return face_name

    def update_frame_info(self,
                          face_name,
                          track_id,
                          current_frame_id,
                          current_frame_time_stamp,
                          behavior_cls):

        if behavior_cls != '':
            self.Frameinfo['Face_id'].append(face_name)
            self.Frameinfo['Track_id'].append(track_id)
            self.Frameinfo['Frame_id'].append(current_frame_id)
            self.Frameinfo['Time_stamp'].append(current_frame_time_stamp)
            for i in self.labels:
                if i == behavior_cls.title():
                    self.Frameinfo[i].append(1)
                else:
                    self.Frameinfo[i].append(0)

        if len(self.Frameinfo['Frame_id']) % 1000 == 0:
            self.save_generated_data()

    def save_generated_data(self):
        self.Frameinfo = pd.DataFrame(self.Frameinfo)
        if self.is_first_save:
            self.Frameinfo.to_csv(self.csv_path,
                                  lineterminator="\n",
                                  header=True,
                                  index=False,
                                  mode='w',
                                  encoding='utf_8_sig')
            self.Frameinfo = self.make_data_dict()
        else:
            self.Frameinfo.to_csv(self.csv_path,
                                  lineterminator="\n",
                                  header=False,
                                  index=False,
                                  mode='a',
                                  encoding='utf_8_sig')
            self.Frameinfo = self.make_data_dict()
        self.is_first_save = False
    def update_label_text(self, text_dict, face_name, track_id, behavior_cls=None, behavior_prob=None):

        # if behavior_cls != '' and behavior_prob != '':
        #     self.label_text = dict()
        #     self.label_text[track_id] = (behavior_cls.title(), behavior_prob)
        # if self.label_text:
        #     if track_id in self.label_text:
        #         label_text = f'{face_name} {track_id} ' \
        #                      f'{self.label_text[track_id][0]} ' \
        #                      f'{self.label_text[track_id][1]}'
        #     else:
        #         label_text = f'{face_name} {track_id} '
        # else:
        #     label_text = f'{face_name} {track_id}'
        #
        # return label_text


    # def split_pose_result(self): # np.ndarray
    #     num_person = max([len(x['keypoints']) for x in self.pose_results])
    #     pose_results_splited = dict()
    #     for idx, d in enumerate(self.pose_results):
    #         if len(d['keypoints']) < num_person:
    #             frame_person = len(d['keypoints'])
    #         else:
    #             frame_person = num_person
    #         for i in range(frame_person):
    #             temp_dict = dict(
    #                 bboxes=np.zeros((1, 4), dtype=np.float16),
    #                 keypoints_visible=np.zeros((1, 17), dtype=np.float16),
    #                 keypoints=np.zeros((1, 17, 2), dtype=np.float16),
    #                 bbox_scores=np.zeros((1), dtype=np.float16),
    #                 keypoint_scores=np.zeros((1, 17), dtype=np.float16))
    #             temp_dict['bboxes'][0] = d['bboxes'][i]
    #             temp_dict['keypoints_visible'][0] = d['keypoints_visible'][i]
    #             temp_dict['keypoints'][0] = d['keypoints'][i]
    #             temp_dict['bbox_scores'] = d['bbox_scores'][i]
    #             temp_dict['keypoint_scores'][0] = d['keypoint_scores'][i]
    #             track_id = int(d['track_bboxes'][:, 4][i])
    #             if track_id not in pose_results_splited:
    #                 pose_results_splited[track_id] = list()
    #             pose_results_splited[track_id].extend([temp_dict])
    #
    #     self.pose_results = list()
    #     return pose_results_splited

        # 只在需要时更新 self.label_text
        # if track_id not in text_dict:
        #     text_dict[track_id] = {'face_name': face_name, 'track_id': track_id}
        # else:
        #     text_dict[track_id].update({id: {'face_name': face_name, 'track_id': track_id}})
        # text = f'{text_dict[track_id]["face_name"]} {text_dict[track_id]["track_id"]} '
        if behavior_cls and behavior_prob:
            text_dict[track_id].update({'cls': behavior_cls,
                                        'prob': behavior_prob})
            text_extend = f' {behavior_cls} {behavior_prob}'
            text_dict[track_id]['text_extend'] = text_extend
        else:
            text_dict[track_id]['text_extend'] = None

        # if 'cls' in text_dict[id] and 'prob' in text_dict[id]:
        #
        #     text += f'{text_dict[id]["cls"]} {text_dict[id]["prob"]}'






        # if behavior_cls and behavior_prob:
        #     self.label_text = self.label_text or {}
        #     self.label_text[track_id] = (behavior_cls.title(), behavior_prob)
        #
        # # 构建基本标签文本
        # text = {track_id: {}}
        #
        # # 如果有行为信息，添加到标签文本
        # if track_id in self.label_text:
        #     cls, prob = self.label_text[track_id]
        #     text[track_id].update({'cls': cls,
        #                            'prob': prob})

        return text_dict
    def split_pose_result(self):
        num_person = max(len(x['keypoints']) for x in self.pose_results)
        pose_results_splited = {}

        for d in self.pose_results:
            frame_person = min(len(d['keypoints']), num_person)

            # 使用NumPy的广播功能一次性创建所有字典
            temp_dicts = [{
                'bboxes': d['bboxes'][i:i + 1],
                'keypoints': d['keypoints'][i:i + 1],
                'bbox_scores': d['bbox_scores'][i:i + 1],
                'keypoint_scores': d['keypoint_scores'][i:i + 1]
            } for i in range(frame_person)]

            # 使用NumPy索引一次性获取所有track_ids
            # if len(d['track_bboxes']) == 5:
            try:
                track_ids = d['track_bboxes'][:frame_person, 4].astype(int)
                for track_id, temp_dict in zip(track_ids, temp_dicts):
                    # 使用字典推导式更新pose_results_splited
                    pose_results_splited.setdefault(track_id, []).append(temp_dict)
            except:
                continue



        self.pose_results = []
        return pose_results_splited

    def label(self):
        if self.target_type in ["Primates", "Artiodactyla", "Carnivora", "Perissodactyla"]:
            with open(self.behavior_label, 'r') as f:
                self.behavior_label_ = json.load(f)
            return self.behavior_label_[self.target_type.title()]['categories']
        else:
            raise ValueError("Unrecognized behavior type")

    def make_data_dict(self):
        FrameInfo = {'Face_id': [],
                     'Track_id': [],
                     'Frame_id': [],
                     'Time_stamp': []}
        self.labels = self.label()
        for i in self.labels:
            FrameInfo[i] = []

        return FrameInfo

    def generate_reports(self):  # todo
        pass