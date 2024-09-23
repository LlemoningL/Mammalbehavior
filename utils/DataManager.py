import json
import pandas as pd
from utils.util import bind_faceid_trackid
from pathlib import Path


class DataManager:
    def __init__(self, output_path, traget_type, behavior_label, interval):
        self.target_type = traget_type
        self.behavior_label = behavior_label
        self.pose_results = {}
        self.faceid_trackid = {}
        self.Frameinfo = pd.DataFrame(self.make_data_dict())
        self.temp_info = self.make_data_dict()
        self.label_text = {}
        self.frame_coordinates = {}
        self.csv_path = Path(output_path) / f'info_{interval}s_{output_path.stem}.csv'
        self.is_first_save = True

    def update_pose_result(self, id, pose_result):
        if pose_result[0]['track_bboxes'].shape[1] <= 4:
            return
        if id not in self.pose_results:
            self.pose_results[id] = []
            self.pose_results[id].extend(pose_result)
        else:
            self.pose_results[id].extend(pose_result)

    def update_faceid_trackid(self, face_name, track_id, frame_id):
        face_name, self.faceid_trackid = bind_faceid_trackid(face_name,
                                                             track_id,
                                                             self.faceid_trackid,
                                                             frame_id)

        return face_name

    def update_frame_info(self,
                          face_name,
                          track_id,
                          current_frame_id,
                          current_frame_time_stamp,
                          behavior_cls):

        if behavior_cls != '':
            self.temp_info['Face_id'].append(face_name)
            self.temp_info['Track_id'].append(track_id)
            self.temp_info['Frame_id'].append(current_frame_id)
            self.temp_info['Time_stamp'].append(current_frame_time_stamp)
            for i in self.labels:
                if i == behavior_cls.title():
                    self.temp_info[i].append(1)
                else:
                    self.temp_info[i].append(0)
        if len(self.temp_info['Face_id']) > 10:
            self.temp_info = pd.DataFrame(self.temp_info)
            for index, row in self.temp_info.iterrows():
                if row['Face_id'] == '_':
                    for k, v in self.faceid_trackid.items():
                        if row['Track_id'] in v:
                            self.temp_info.loc[index, 'Face_id'] = k

            self.Frameinfo = pd.concat([self.Frameinfo, self.temp_info], axis=0, ignore_index=True)
            self.temp_info = self.make_data_dict()

        # if behavior_cls != '':
        #     self.Frameinfo['Face_id'].append(face_name)
        #     self.Frameinfo['Track_id'].append(track_id)
        #     self.Frameinfo['Frame_id'].append(current_frame_id)
        #     self.Frameinfo['Time_stamp'].append(current_frame_time_stamp)
        #     for i in self.labels:
        #         if i == behavior_cls.title():
        #             self.Frameinfo[i].append(1)
        #         else:
        #             self.Frameinfo[i].append(0)

        if not self.Frameinfo.empty and self.Frameinfo.shape[0] % 1000 == 0:
            self.save_generated_data()

    def save_generated_data(self):
        if not isinstance(self.temp_info, pd.DataFrame):
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

    def save_data_split(self):
        """
            Args:
                file: dataframe including information of  face id, tack id, frame id, time_stamp
                and behavior types.
                face_tarckid: dictionary of bound face id and track id list
            """
        csv_dict = dict()
        csvframe = pd.read_csv(self.csv_path)
        print(csvframe.head())
        # filtering data that contains '_', it is no bound data
        tempframe = csvframe.loc[csvframe['Face_id'] == '_'].copy()
        # filtering data according to face id
        for i, l in self.faceid_trackid.items():
            print(f'split csv file for {i}')
            if i not in csv_dict:
                faceidtemp = csvframe.loc[csvframe['Face_id'] == i].copy()
                csv_dict[i] = faceidtemp
            # filtering track id that contained in face id data in tempframe
            for j in l.keys():
                result = csv_dict[i]['Track_id'].isin([j])
                if result.any():
                    mid_ = tempframe.loc[tempframe['Track_id'] == j].copy()
                    # rename '_' to bound face id
                    mid_['Face_id'] = mid_['Face_id'].apply(lambda x: i)
                    # concate face id data and filtered data in tempframe
                    csv_dict[i] = pd.concat([csv_dict[i], mid_])
                    # delet filtered data in tempframe
                    tempframe.drop(index=tempframe[tempframe['Track_id'].isin([j]) == True].index, inplace=True)
            # sorting data with column 'Time_stamp'
            faceidtemp.sort_values('Time_stamp', inplace=True)
            # writing face id data to specific csv
            faceidtemp.to_csv(self.csv_path.parent / f'{self.csv_path.stem}_{i}.csv', lineterminator="\n", header=True,
                              index=False, mode='a',
                              encoding='utf_8_sig')
        print('csv file saved')

    def update_label_text(self, text_dict, face_name, track_id, behavior_cls=None, behavior_prob=None):

        if behavior_cls and behavior_prob:
            text_dict[track_id].update({'cls': behavior_cls,
                                        'prob': behavior_prob})
            text_extend = f' {behavior_cls} {behavior_prob}'
            text_dict[track_id]['text_extend'] = text_extend
        else:
            text_dict[track_id]['text_extend'] = None

        return text_dict
    def split_pose_result(self):
        # num_person = max(len(x['keypoints']) for x in self.pose_results)
        # pose_results_splited = {}
        #
        # for d in self.pose_results:
        #     frame_person = min(len(d['keypoints']), num_person)
        #
        #     # 使用NumPy的广播功能一次性创建所有字典
        #     temp_dicts = [{
        #         'bboxes': d['bboxes'][i:i + 1],
        #         'keypoints': d['keypoints'][i:i + 1],
        #         'bbox_scores': d['bbox_scores'][i:i + 1],
        #         'keypoint_scores': d['keypoint_scores'][i:i + 1]
        #     } for i in range(frame_person)]
        #
        #     # 使用NumPy索引一次性获取所有track_ids
        #     # if len(d['track_bboxes']) == 5:
        #     try:
        #         track_ids = d['track_bboxes'][:frame_person, 4].astype(int)
        #         for track_id, temp_dict in zip(track_ids, temp_dicts):
        #             # 使用字典推导式更新pose_results_splited
        #             pose_results_splited.setdefault(track_id, []).append(temp_dict)
        #     except:
        #         continue

        pose_results_splited = self.pose_results
        self.pose_results = {}
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