import argparse
import shutil
import sys
import time
import random
import re
import os
import torch
import torchvision
from PIL import Image
import cv2
import numpy as np
from pathlib import Path as p
from tqdm import tqdm
import yaml
import imageio


class AutoDataset:
    """
    inputpath: data source path.
    outputpath: output path of processed data.
    config_file: config file path.
    train_size: split data to dataset, if train size is 0.8, then val size and test size
                will be 0.1 and 0.1 each.
    max_size: max number of files in dataset, default is 500000
    datasetname: new dataset name after process
    """

    def __init__(self,
                 inputpath,
                 outputpath,
                 train_size=0.8,
                 val_size=0.1,
                 test_size=0.1,
                 max_size=100000,
                 datasetname='AutoDataset'):
        self.inputpath = inputpath
        if outputpath is not None:
            self.outputpath = p(outputpath) / 'train'
        else:
            self.outputpath = p(inputpath).parent / f'{p(inputpath).stem}_EnhanceImages' / 'train'
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.max_size = max_size
        self.datasetname = datasetname
        for i in [self.train_size, self.val_size, self.test_size]:
            if not isinstance(i, (int, float)):
                raise ValueError(f"train_size, val_size, test_size must be float,"
                       f"but got {i} {type(i)}")
        if sum([self.train_size, self.val_size, self.test_size]) != 1:
            raise ValueError(f"The sum of train_size, val_size, test_size must be 1")

    def makedataset(self, datasettype='with_label', enhance=True):
        """
        datasettype: choice in ['with_label', 'no_label'], 'with_label' for making dataset
                    with yolo format like and 'no_label' for resnet format like.
        convert: whether convert data to standard name format, only make resnet format dataset works.
                for example: '多多' convert to 'Chimpanzee_000000000000_多多'
        """
        type = ['with_label', 'no_label']

        if datasettype not in type:
            raise NameError(f"Only 'with_label' or 'no_label' type supported, please check datasettype")
        elif datasettype == type[0]:
            train_path, val_path, test_path = self.dataset_maker_yolo(self.inputpath,
                                                                      self.train_size,
                                                                      self.val_size,
                                                                      self.test_size,
                                                                      self.max_size,
                                                                      self.datasetname)
            if enhance:

                self.imgenhance_yolo(train_path, self.outputpath)
        elif datasettype == type[1]:

            train_path, val_path, test_path = self.dataset_maker(self.inputpath,
                                                                 self.train_size,
                                                                 self.val_size,
                                                                 self.test_size,
                                                                 self.max_size,
                                                                 self.datasetname)
            if enhance:
                self.imgenhance_resnet(train_path, self.outputpath)

    def imgenhance_resnet(self, imgpath, output, seed=17):
        print('enhancing images\n')
        torch.manual_seed(seed)
        imgpath_p = p(imgpath)
        output_p = p(output)

        a = [torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
             # torchvision.transforms.Grayscale(num_output_channels=1),
             torchvision.transforms.GaussianBlur(kernel_size=23, sigma=(0.5, 2.0)),
             # torchvision.transforms.RandomInvert(p=0.5),
             torchvision.transforms.RandomPosterize(4, p=1),
             # torchvision.transforms.RandomSolarize(400, p=0.5),
             torchvision.transforms.RandomAdjustSharpness(3, p=1),
             torchvision.transforms.RandomHorizontalFlip(p=1),
             torchvision.transforms.RandomVerticalFlip(p=1),
             torchvision.transforms.RandomRotation(degrees=90),
             torchvision.transforms.RandomRotation(degrees=120),
             torchvision.transforms.RandomRotation(degrees=150),
             # torchvision.transforms.RandomAutocontrast(p=0.5),
             torchvision.transforms.RandomEqualize(p=1),]
             # torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),]
        temp = list(imgpath_p.iterdir())

        total_dir_num = len(list(imgpath_p.iterdir()))
        for dir in imgpath_p.iterdir():
            if not (output_p / dir.stem).exists():
                (output_p / dir.stem).mkdir(parents=True)
            dir_num = 1

            init_num = 0
            for bhv in tqdm(list(dir.iterdir()), postfix=f'processing number {dir_num }  total {total_dir_num}  name {dir.stem}'):
                if bhv.suffix == '.jpg' or '.JPG' or '.png' or '.PNG'\
                                 '.jpeg' or '.JPEG':
                    img = imageio.imread_v2(str(bhv))
                    if len(img.shape) > 2 and img.shape[2] == 4:
                        # slice off the alpha channel
                        img = img[:, :, :3]
                    # temp = f'{output}/{dir.stem}/{init_num:012d}.jpg'
                    origin_img_name = str(bhv)
                    new_img_name = f'{output}/{dir.stem}/{init_num:012d}.jpg'
                    shutil.copy(origin_img_name, new_img_name)
                    init_num2 = init_num + 1
                    for i, trans in enumerate(a):
                        try:
                            trans_im = trans(Image.fromarray(img))
                        except:
                            continue
                        trans_im = np.array(trans_im)
                        imageio.imwrite(f'{output}/{dir.stem}/{init_num2:012d}.jpg', trans_im)
                        init_num2 = init_num2 + 1


                    init_num = init_num2 + 1


    def imgenhance_yolo(self, imgpath, output, seed=17):
        print('enhancing images\n')
        torch.manual_seed(seed)
        imgpath_p = p(imgpath)
        output_p = p(output)
        yolo_like_dir = ['images', 'labels']

        a = [torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
             torchvision.transforms.Grayscale(num_output_channels=1),
             torchvision.transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
             # torchvision.transforms.RandomInvert(p=0.5),
             torchvision.transforms.RandomPosterize(4, p=1),
             # torchvision.transforms.RandomSolarize(400, p=0.5),
             torchvision.transforms.RandomAdjustSharpness(3, p=1),
             # torchvision.transforms.RandomAutocontrast(p=0.5),
             torchvision.transforms.RandomEqualize(p=1)]

        init_num = 0

        for dir in yolo_like_dir:
            if not (output_p / dir).exists():
                (output_p / dir).mkdir(parents=True)
        for file in tqdm(list(imgpath_p.iterdir()), postfix=f'Augmentation'):
            img = imageio.imread_v2(str(file))
            if len(img.shape) > 2 and img.shape[2] == 4:
                # slice off the alpha channel
                img = img[:, :, :3]
            # temp = f'{output}/{dir.stem}/{init_num:012d}.jpg'
            if img is None:
                break
            else:
                origin_img_name = str(file)
                # new_img_name = f'{output}/{dir.stem}/{k.stem}/{init_num:012d}.jpg'
                new_img_name = output_p / 'images' / f'{init_num:012d}.jpg'
                shutil.copy(origin_img_name, new_img_name)
                origin_label_name = file.parent.parent.parent / 'labels' / 'train' / file.with_suffix('.txt').name
                new_label_name = output_p / 'labels' / f'{init_num:012d}.txt'
                shutil.copy(origin_label_name, new_label_name)
                init_num2 = init_num + 1
                for i, trans in enumerate(a):
                    # if i == len(a):
                    #     trans_im = Image.fromarray(img)
                    #     trans_im = torchvision.transforms.ToTensor()(trans_im)
                    #     trans_im = trans(trans_im)
                    #     trans_im = np.array(trans_im)
                    # else:
                    #     trans_im = trans(Image.fromarray(img))
                    #     trans_im = np.array(trans_im)
                    try:
                        trans_im = trans(Image.fromarray(img))
                    except:
                        continue
                    trans_im = np.array(trans_im)
                    # imageio.imwrite(f'{output}/{p(dir).stem}/{init_num2:012d}.jpg', trans_im)
                    imageio.imwrite(output_p / 'images' / f'{init_num2:012d}.jpg', trans_im)
                    trans_new_label_name = output_p / 'labels' / f'{init_num2:012d}.txt'
                    shutil.copy(origin_label_name, trans_new_label_name)
                    init_num2 = init_num2 + 1

                init_num = init_num2 + 1

    def dataset_maker(self, inputpath, train_size, val_size, test_size, max_size, datasetname):
        '''
        制作数据集，按照训练集：验证集：测试集=8:1:1的比例自动制作数据集
        采用复制方式，源文件保留
        :return:
        '''

        max_num = max_size
        sor_p = p(inputpath)  # 数据路径，包含各目标文件夹
        parent_path = sor_p.parent   # 数据目录父级路径，用于生成datasets路径
        da_p = f'{sor_p.stem}_{datasetname}'
        train_p = parent_path / da_p / 'train'
        valid_p = parent_path / da_p / 'val'
        test_p = parent_path / da_p / 'test'

        if not train_p.is_dir():
            p(train_p).mkdir(parents=True, exist_ok=True)   # 建立训练集文件夹
            print(f'folder {train_p.stem} ready')
        if not valid_p.is_dir():
            p(valid_p).mkdir(parents=True, exist_ok=True)   # 建立验证集文件夹
            print(f'folder {valid_p.stem} ready')
        if not test_p.is_dir():
            p(test_p).mkdir(parents=True, exist_ok=True)   # 建立测试集文件夹
            print(f'folder {test_p.stem} ready\n')
        print('dataset folder ready\n')


        for f_1 in sor_p.iterdir():   # 遍历源文件夹
            if not (train_p / f_1.name).is_dir():
                p(train_p / f_1.name).mkdir(parents=True, exist_ok=True)   # 在训练集文件夹中建立对应源文件夹的目标文件夹
            if not (valid_p / f_1.name).is_dir():
                p(valid_p / f_1.name).mkdir(parents=True, exist_ok=True)   # 在验证文件夹中建立对应源文件夹的目标文件夹
            if not (test_p / f_1.name).is_dir():
                p(test_p / f_1.name).mkdir(parents=True, exist_ok=True)   # 在测试集文件夹中建立对应源文件夹的目标文件夹
        print('target folder ready\n')



        for targ_folder in (parent_path / da_p).iterdir():
            t0 = time.time()
              # print(targ_folder)
            if targ_folder.stem == 'train':   # 判断条件，防止重复复制，优先写入训练集
                j = 1
                for file_folder in sor_p.iterdir():   # 遍历训练目标文件夹
                    print(f'{"="*20} {file_folder.name}({j}/{len(list(sor_p.iterdir()))}) 整理中{"="*20}')
                      #  j = j + 1
                    list_img = list(file_folder.iterdir())
                    total_lenth = len(list_img)
                    if len(list_img) >= int(max_num):
                        list_img = random.sample(list_img, max_num)
                        total_lenth = max_num


                    # list_img = []
                    # for img in file_folder.iterdir():   # 遍历训练目标图片
                    #     seed = random.randint(1, 8)
                    #     if len(list(file_folder.iterdir())) >= int(max_num):
                    #         if seed <= 4:
                    #             list_img.append(img)   # 将图片地址写入空列表，为后续数据切分做准备
                    #     else:
                    #         list_img.append(img)
                    if file_folder.stem == (train_p / file_folder.name).stem:   # 判断源文件夹中和训练集文件夹中名称是否一致
                          # print(file_folder, (test_p / file_folder.name))
                        random.shuffle(list_img)   # 乱序列表，为随机抽取做准备
                        num1 = int(total_lenth * train_size)   # 设置切割占比
                        random_sample_list = random.sample(list_img, num1)  # 随机取样num1个元素
                        # lenth = len(list_img[:num1])
                        for x in tqdm(random_sample_list, desc='train_dir 进度'):
                            shutil.copy(x, (train_p / file_folder.name))   # 复制列表random_sample_list中的元素
                            # lenth = len(list_img[:num1])
                            # a = "*" * int(((i+1) / lenth) * 35)
                            # b = "." * int(35 - (((i+1) / lenth) * 35))
                            # c = (i / lenth) * 100
                            # print(f'\rtrain_dir 进度：{c:.0f}%[{a}->{b}]', end="")
                            list_img.remove(x)  # 在list_img中删除random_sample_list存在的元素


                    if file_folder.stem == (valid_p / file_folder.name).stem:   # 判断源文件夹中和验证集文件夹中名称是否一致

                        random.shuffle(list_img)    # 乱序列表，为随机抽取做准备
                        num2 = int(total_lenth * val_size)    # 剩余数据50%
                        random_sample_list = random.sample(list_img, num2)
                        # i = 1
                        # lenth = len(list_img[:num2])
                        for x2 in tqdm(random_sample_list, desc='valid_dir 进度'):
                            shutil.copy(x2, valid_p / file_folder.name)    # 按比例复制列表
                            # lenth = len(list_img[:num2])
                            # a = "*" * int((i / lenth) * 35)
                            # b = "." * int(35 - ((i / lenth) * 35))
                            # c = (i / lenth) * 100
                            # i = i + 1
                            # print(f'\rvalid_dir 进度：{c:.0f}%[{a}->{b}]', end="")
                            list_img.remove(x2) # 在list_img中删除random_sample_list存在的元素
                              #  time.sleep(0.1)


                    if file_folder.stem == (test_p / file_folder.name).stem:    # 判断源文件夹中和测试集文件夹中名称是否一致

                        # i = 1
                        # lenth = len(list_img)
                        for x3 in tqdm(list_img, desc='test_dir  进度'):
                            shutil.copy(x3, test_p / file_folder.name)    # 将剩余的图片写入测试集对应文件夹
                            # lenth = len(list_img)
                            # a = "*" * int((i / lenth) * 35)
                            # b = "." * int(35 - ((i / lenth) * 35))
                            # c = (i / lenth) * 100
                            # i = i + 1
                            # print(f'\rtest_dir  进度：{c:.0f}%[{a}->{b}]', end="")

                        print(f'\n{"-"*20} {file_folder.name}({j}/{len(list(sor_p.iterdir()))}) 已完毕{"-"*20}\n\n')
                    j = j + 1

        print('====Dataset has been ready====')

        return train_p, valid_p, test_p

    def dataset_maker_yolo(self, inputpath, train_size, val_size, test_size, max_size, datasetname):
        '''
        制作数据集，按照训练集：验证集：测试集的比例自动制作数据集
        采用复制方式，源文件保留
        :return:
        '''
        max_num = max_size
        sor_p = p(inputpath)  # 数据路径，包含各目标文件夹
        parent_path = sor_p.parent  # 数据目录父级路径，用于生成datasets路径
        da_p = p(f'{sor_p.stem}_{datasetname}')
        yolo_like_dir = ['images', 'labels']
        dataset_dir = ['train', 'val', 'test']

        for yld in yolo_like_dir:
            for dd in dataset_dir:
                target_p = parent_path / da_p / yld / dd
                if not target_p.exists():
                    p(target_p).mkdir(parents=True, exist_ok=True)  # 建立文件夹
                    print(f'folder {target_p.stem} ready')
        print('dataset folder ready\n')
        for origin_folder in (sor_p).iterdir():
            t0 = time.time()
            # print(targ_folder)
            if origin_folder.stem == 'images':
                list_img = list(origin_folder.iterdir())
                total_lenth = len(list_img)
                if len(list_img) >= int(max_num):
                    list_img = random.sample(list_img, max_num)
                    total_lenth = max_num
                random.shuffle(list_img)
                train_set = random.sample(list_img, int(total_lenth * train_size))
                temp_list = list(set(list_img) - set(train_set))
                random.shuffle(temp_list)
                val_set = random.sample(temp_list, int(total_lenth * val_size))
                test_set = list(set(temp_list) - set(val_set))

                for l in [['train', train_set], ['val', val_set], ['test', test_set]]:
                    for file in tqdm(l[1], desc=f'{l[0]}'):
                        imgage_target_name = parent_path / da_p / 'images' / l[0] / file.name
                        label_target_name = parent_path / da_p / 'labels' / l[0] / file.with_suffix('.txt').name
                        origin_label = file.parent.parent / 'labels' / file.with_suffix('.txt').name
                        shutil.copy(file, imgage_target_name)  # 复制列表中的元素
                        shutil.copy(origin_label, label_target_name)
                train_p, valid_p, test_p = parent_path / da_p / 'images' / 'train', \
                                           parent_path / da_p / 'images' / 'val', \
                                           parent_path / da_p / 'images' / 'test'

        print('====Dataset has been ready====')

        return train_p, valid_p, test_p

    def load_config(self, filepath):
        with open(filepath, encoding='UTF-8') as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)

        return data_dict


if __name__ == '__main__':
    input = 'path/to/dataset'
    output = 'path/to/output'
    ad = AutoDataset(input, output, train_size=0.8, val_size=0.2, test_size=0.0, datasetname='Autodataset')
    ad.makedataset(datasettype='with_label')




