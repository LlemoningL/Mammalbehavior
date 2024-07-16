import pandas as pd
from pathlib import Path as p
import configparser
import cv2
import shutil
from sklearn.model_selection import train_test_split
from copy import deepcopy
import sys


# /home/ztx/lj/4tdisk/1a数据总库20221022/识别数据汇总/佛坪行为数据_2022.4_V/reid/train/custom1/csv.csv
class darklabel2mot:
    """convert darklabel output to mot folder structure and format"""

    def __init__(self, dataset_path,  output='custom_mot', test_ratio=0.2):
        self.video_suffix = ['.avi', '.wmv', '.mov', '.rm', '.ram', '.swf', '.flv', '.mp4']
        self.test_ration = test_ratio
        self.dataset_path = dataset_path
        self.output = output
        self.train_path, self.val_path = self.split_train_test(self.dataset_path)
        for i in [self.train_path, self.val_path]:
            # pass
            self.mk_mot_dir(i)
            self.write_ini_file(i)

    def split_train_test(self, path):
        dir_list = list(p(path).iterdir())
        train_path, test_path = train_test_split(dir_list, test_size=self.test_ration)

        return {'train': train_path}, {'test': test_path}


    def csv2txt(self):

        print('convert csv to txt')

        self.output = p(self.output)
        for i in [self.train_path, self.val_path]:
            for p_key, p_vlaue in i.items():
                for p_ in p_vlaue:
                    p_ = p(p_)
                    # for d_p in list(p_.iterdir()):
                    # if p_key == 'train':
                    csv_path = p(p_) / 'gt.csv'
                    csv_file = pd.read_csv(csv_path, header=None)
                    csv_file_gt = deepcopy(csv_file)
                    csv_file_det = deepcopy(csv_file)
                    for line_gt in csv_file_gt.values:
                        line_gt[0] = int(line_gt[0]) + 1
                        new_gt = pd.DataFrame([line_gt])
                        get_save_path = self.output / p_key / p_.stem / 'gt' / 'gt.txt'
                        new_gt.to_csv(get_save_path, line_terminator="\n", header=False, index=False, mode='a')

                    for line in csv_file_det.values:
                        line[0] = int(line[0]) + 1
                        line = line[:7]
                        line[1] = -1
                        new = pd.DataFrame([line])
                        det_save_path = self.output / p_key / p_.stem / 'det' / 'det.txt'
                        new.to_csv(det_save_path, line_terminator="\n", header=False, index=False, mode='a')
        print('done')
        self.copy_file()
        # self.split_t_v()


    def write_ini_file(self, path):
        '''
        [Sequence]
        name = MOT17 - 02 - FRCNN
        imDir = img1
        frameRate = 30
        seqLength = 600
        imWidth = 1920
        imHeight = 1080
        imExt =.jpg
        '''

        for p_key, p_vlaue in path.items():
            print(f'make seqinfo.ini in {p_key}')
            for p_ in p_vlaue:
                p_ = p(p_)
                # for d_p in list(p_.iterdir()):
                path = p(p_)
                img_path = path / 'img'
                im_list = list(img_path.iterdir())
                seqLength = len(im_list)
                img = cv2.imread(str(im_list[0]))
                imWidth = img.shape[1]
                imHeight = img.shape[0]
                imExt = p(im_list[0]).suffix


                for i in p_.iterdir():
                    if p(i).suffix.lower() in self.video_suffix:
                        cap = cv2.VideoCapture(str(i))
                        fps = cap.get(cv2.CAP_PROP_FPS)

                # if int(fps) <= 0:
                #     print(f'Need video file in {path} to get "fps"')
                #     sys.exit()


                config = configparser.ConfigParser()
                config.add_section('Sequence')
                config.set('Sequence', 'name', f'{p_.stem}')
                config.set('Sequence', 'imDir', 'img')
                config.set('Sequence', 'frameRate', str(round(fps)))
                config.set('Sequence', 'seqLength', str(seqLength))
                config.set('Sequence', 'imWidth', str(imWidth))
                config.set('Sequence', 'imHeight', str(imHeight))
                config.set('Sequence', 'imExt', str(imExt))
                # info = dict()
                # info['name'] = f'custom_dataset_{i}'
                # info['frameRate'] = fps
                # info['seqLength'] = seqLength
                # info['imWidth'] = imWidth
                # info['imHeight'] = imHeight
                # info['imExt'] = imExt
                # config['Sequence'] = info
                if p_key == 'test':
                    save_path1 = p(self.output) / 'test' / path.stem / 'seqinfo.ini'
                    with open(save_path1, 'w', encoding='utf-8') as f:
                        config.write(f)
                elif p_key == 'train':
                    save_path2 = p(self.output) / 'train' / path.stem / 'seqinfo.ini'
                    with open(save_path2, 'w', encoding='utf-8') as f:
                        config.write(f)
                # with open(save_path) as f:
                #     f.read()


        print('done')


    def mk_mot_dir(self, path):
        '''make mot structure like folder'''


        self.output = p(self.output)
        if not self.output.exists():
            self.output.mkdir()
        for p_key, p_vlaue in path.items():
            for p_ in p_vlaue:
                p_ = p(p_)
                # for d_p in list(p_.iterdir()):
                    # if not (self.output / 'train').exists():
                    #     (self.output / 'train').mkdir()
                if p_key == 'test':
                    if not (self.output / 'test' / p_.stem / 'det').exists():
                        (self.output / 'test' / p_.stem / 'det').mkdir(parents=True)
                    if not (self.output / 'test' / p_.stem / 'gt').exists():
                        (self.output / 'test' / p_.stem / 'gt').mkdir(parents=True)
                    print(f'make directory in {p_key}')
                if p_key == 'train':
                    if not (self.output / 'train' / p_.stem / 'gt').exists():
                        (self.output / 'train' / p_.stem / 'gt').mkdir(parents=True)
                    if not (self.output / 'train' / p_.stem / 'det').exists():
                        (self.output / 'train' / p_.stem / 'det').mkdir(parents=True)
                    print(f'make directory in {p_key}')
        print('done')


    def copy_file(self):

        for i in [self.train_path, self.val_path]:
            for p_key, p_vlaue in i.items():
                for p_ in p_vlaue:
                    p_ = p(p_)
                    # for d_p in list(p_.iterdir()):
                    img_path = p(p_) / 'img'
                    new_img_path = p(self.output) / p_key / f'{p_.stem}/img'
                    print(f'from \n{str(img_path)} \ncopy images to \n{str(new_img_path)}, \nplease wait...')
                    shutil.copytree(img_path, new_img_path)
                    print(f'from \n{str(img_path)} \ncopy images to \n{str(new_img_path)}, \ndone.')


    # def split_t_v(self):
    #     """split gt.txt half to train and half to val"""
    #
    #     self.output = p(self.output)
    #     t_path = self.output / 'train'
    #
    #     print('split gt.txt')
    #     for j in list(t_path.iterdir()):
    #         j = p(j)
    #         total_lines = len(list((t_path / j / 'img').iterdir()))
    #         train_lines = int(total_lines / 2 - 10)
    #         val_lines = int(total_lines / 2 + 10)
    #         for i in list(j.iterdir()):
    #             i = p(i)
    #             if i.stem == 'gt':
    #                 csv_path = i / 'gt.txt'
    #                 csv_file = pd.read_csv(csv_path, header=None)
    #                 for line in csv_file.values:
    #                     if int(line[0]) <= train_lines:
    #                         new = pd.DataFrame([line])
    #                         det_save_path = self.output / 'train' / j / 'gt' / 'gt_half-train.txt'
    #                         new.to_csv(det_save_path, line_terminator="\n", header=False, index=False, mode='a')
    #                     elif int(line[0]) >= val_lines:
    #                         new = pd.DataFrame([line])
    #                         det_save_path = self.output / 'train' / j / 'gt' / 'gt_half-val.txt'
    #                         new.to_csv(det_save_path, line_terminator="\n", header=False, index=False, mode='a')
    #     print('done')




if __name__ == '__main__':
    out_path = r'/home/ztx/lj/4tdisk/1a数据总库20221022/识别数据汇总/takin_behavior/track_output'
    in_path = r'/home/ztx/lj/4tdisk/1a数据总库20221022/识别数据汇总/takin_behavior/track'
    d = darklabel2mot(in_path, output=out_path)
    d.csv2txt()

