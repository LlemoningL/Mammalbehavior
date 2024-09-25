import os
import torch
import torchvision
from PIL import Image
from pathlib import Path as p
import imageio
from pathlib import Path
import glob
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm
# from labelme import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ultralytics.data.converter import convert_coco


class Labelme2coco():
    def __init__(self, args):
        self.classname_to_id = args.class_name
        self.images = []
        self.annotations = []
        self.categories = []
        self.ann_id = 0
        self.train_size = args.train_ratio
        self.val_size = args.val_ratio
        self.test_size = args.test_ratio

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    def _get_keypoints(self, points, keypoints, num_keypoints):
        if points[0] == 0 and points[1] == 0:
            visable = 0
        else:
            visable = 2
            num_keypoints += 1
        keypoints.extend([points[0], points[1], visable])
        return keypoints, num_keypoints

    def _image(self, obj, path):
        image = {}
        image['height'], image['width'] = obj['imageHeight'], obj['imageWidth']
        self.img_id = int(os.path.basename(path).split(".json")[0])
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")

        return image

    def _annotation(self, bboxes_list, keypoints_list, polygon_list, json_path):
        # if len(keypoints_list) != args.join_num * len(bboxes_list):
        #     print('you loss {} keypoint(s) with file {}'.format(args.join_num * len(bboxes_list) - len(keypoints_list), json_path))
        #     print('Please check ！！！')
        #     # sys.exit()
        #     json_path = Path(json_path)
        #     filename = json_path.stem
        #     parent_json_path = json_path.parent.parent
        #     check_dir = parent_json_path / 'checkfile'
        #     if not Path(check_dir).exists():
        #         check_dir.mkdir()
        #     shutil.move(json_path, check_dir / str(filename + '.json'))
        #     shutil.move((json_path.parent / str(filename + '.jpg')), (check_dir / str(filename + '.jpg')))

        i = 0
        for object in bboxes_list:
            annotation = {}
            keypoints = []
            num_keypoints = 0

            label = object['label']
            bbox = object['points']
            box_xywh = self._get_box(bbox)  # box_xywh = [x, y, w, h]
            annotation['id'] = self.ann_id
            annotation['image_id'] = self.img_id
            annotation['category_id'] = int(self.classname_to_id[label])
            annotation['iscrowd'] = 0
            annotation['area'] = box_xywh[2] * box_xywh[3]
            annotation['segmentation'] = [np.asarray(polygon_list).flatten().tolist()]
            annotation['bbox'] = box_xywh

            for keypoint in keypoints_list[i * args.join_num: (i + 1) * args.join_num]:
                point = keypoint['points']
                annotation['keypoints'], num_keypoints = self._get_keypoints(point[0], keypoints, num_keypoints)
            annotation['num_keypoints'] = num_keypoints

            i += 1
            self.ann_id += 1
            self.annotations.append(annotation)

    def _init_categories(self):
        for name, id in self.classname_to_id.items():
            category = {}

            category['supercategory'] = name
            category['id'] = id
            category['name'] = name

            # category['keypoint'] = [str(i + 1) for i in range(args.join_num)]

            self.categories.append(category)

    def to_coco(self, json_path_list):
        self._init_categories()

        for json_path in tqdm(json_path_list):
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']

            bboxes_list, keypoints_list, polygon_list = [], [], []
            for shape in shapes:
                if shape['shape_type'] == 'rectangle':
                    bboxes_list.append(shape)
                elif shape['shape_type'] == 'point':
                    keypoints_list.append(shape)
                elif shape['shape_type'] == "polygon":
                    polygon_list.append(shape['points'])

            self._annotation(bboxes_list, keypoints_list, polygon_list, json_path)

        keypoints = {}
        keypoints['info'] = {'description': '', 'version': 1.0, 'year': 2022}
        keypoints['license'] = ['']
        keypoints['images'] = self.images
        keypoints['annotations'] = self.annotations
        keypoints['categories'] = self.categories
        return keypoints

    def check_keypoints_num(self, json_path_list):

        print('Check keypoints num in json file')
        checkfilejsn = []

        for json_path in tqdm(json_path_list):
            obj = self.read_jsonfile(json_path)
            # self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']

            bboxes_list, keypoints_list = [], []
            for shape in shapes:
                if shape['shape_type'] == 'rectangle':
                    bboxes_list.append(shape)
                elif shape['shape_type'] == 'point':
                    keypoints_list.append(shape)

            if len(keypoints_list) != args.join_num * len(bboxes_list):

                log_info = '\nyou loss 【{}】 keypoint(s) with file:"{}"'.format(
                    args.join_num * len(bboxes_list) - len(keypoints_list), json_path)
                # sys.exit()
                json_path = Path(json_path)
                filename = json_path.stem
                parent_json_path = json_path.parent.parent
                check_dir = parent_json_path / 'checkfile'
                global check_log
                check_log = check_dir / f'Check_log.txt'
                if not Path(check_dir).exists():
                    check_dir.mkdir()
                with open(check_log, 'a', encoding='utf-8') as f:
                    f.write(log_info)
                shutil.move(json_path, check_dir / str(filename + '.json'))
                shutil.move((json_path.parent / str(filename + '.jpg')), (check_dir / str(filename + '.jpg')))
                checkfilejsn.append(json_path)

        if len(checkfilejsn) > 0:
            for i in checkfilejsn:
                json_path_list.remove(str(i))
            print(f'Done, please see {check_log} for more details')
        else:
            print(f'Done, every thing is OK')

        return json_path_list

    def enhance_img_json(self, data, inpath, outpath='Enhance', seed=17):
        torch.manual_seed(seed)
        imgpath_p = p(inpath)
        output_p = imgpath_p.parent / f'{imgpath_p.stem}_{outpath}'
        if not output_p.exists():
            output_p.mkdir(parents=True)

        init_num = 0
        for bhv in tqdm(data, postfix=f'processing images'):
            bhv = p(bhv)
            bhv_img = bhv.with_suffix('.jpg')
            img = imageio.imread_v2(str(bhv_img))
            a = [torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
                 torchvision.transforms.Grayscale(num_output_channels=1),
                 torchvision.transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
                 # torchvision.transforms.RandomInvert(p=0.5),
                 torchvision.transforms.RandomPosterize(4, p=1),
                 # torchvision.transforms.RandomSolarize(400, p=0.5),
                 torchvision.transforms.RandomAdjustSharpness(3, p=1),
                 # torchvision.transforms.RandomAutocontrast(p=0.5),
                 torchvision.transforms.RandomEqualize(p=1)]

            # temp = output_p / f'{init_num:012d}.jpg'

            origin_img_name = str(bhv_img)
            new_img_name = output_p / f'{init_num:012d}.jpg'
            if not new_img_name.exists():
                shutil.copy(origin_img_name, new_img_name)
            origin_jsn_name = bhv
            new_jsn_name = output_p / f'{init_num:012d}.json'
            if not new_jsn_name.exists():
                shutil.copy(origin_jsn_name, new_jsn_name)

            init_num2 = init_num + 1

            for i, trans in enumerate(a):
                trans_im = trans(Image.fromarray(img))
                trans_im = np.array(trans_im)
                if trans_im.ndim == 3:
                    trans_im = trans_im[..., ::-1]
                trans_new_jsn_name = output_p / f'{init_num2:012d}.json'
                if not (output_p / f'{init_num2:012d}.jpg').exists():
                    imageio.imwrite(output_p / f'{init_num2:012d}.jpg', trans_im)
                    with open(origin_jsn_name, 'r') as f:
                        content = json.load(f)
                    content['imageData'] = None
                    if not trans_new_jsn_name.exists():
                        with open(trans_new_jsn_name, 'w') as f:
                            json.dump(content, f)
                    init_num2 = init_num2 + 1

            init_num = init_num2 + 1
        output_json_list = glob.glob(os.path.join(output_p, "**/**.json"), recursive=True)
        return output_json_list

    def split_dataset(self, data, labels, train_size, val_size, test_size, seed=42):
        total_num = len(data)
        if train_size == 0:
            train_path = []
            # train_num = round(train_size * total_num)
            val_num = round(val_size * total_num)
            test_num = round(test_size * total_num)
            test_num = min(test_num, total_num - val_num)
            val_path, test_path, _, _ = train_test_split(data,
                                                   labels,
                                                   train_size=val_num,
                                                   test_size=test_num,
                                                   random_state=seed)
        elif val_size == 0:
            val_path =[]
            train_num = round(train_size * total_num)
            # val_num = round(val_size * total_num)
            test_num = round(test_size * total_num)
            test_num = min(test_num, total_num - train_num)
            train_path, test_path, _, _ = train_test_split(data,
                                                     labels,
                                                     train_size=train_num,
                                                     test_size=test_num,
                                                     random_state=seed)
        elif test_size == 0:
            test_path = []
            train_num = round(train_size * total_num)
            val_num = round(val_size * total_num)
            # test_num = round(test_size * total_num)
            val_num = min(val_num, total_num - train_num)
            train_path, val_path, _, _ = train_test_split(data,
                                                    labels,
                                                    train_size=train_num,
                                                    test_size=val_num,
                                                    random_state=seed)
        else:
            train_num = round(train_size * total_num)
            val_num = round(val_size * total_num)
            test_num = round(test_size * total_num)
            test_num = min(test_num, total_num - train_num - val_num)
            train_path, val_path1, labels_train, labels_val = (
                train_test_split(data,
                                 labels,
                                 train_size=train_num,
                                 test_size=total_num - train_num,
                                 random_state=seed))
            val_path, test_path, y_val, y_test = (
                train_test_split(val_path1,
                                 labels_val,
                                 train_size=val_num,
                                 test_size=test_num,
                                 random_state=seed))

        return train_path, val_path, test_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #  bounding class and id with dict

    # switch dict to your own label, coco classes start from index 1
    # class_name_dict = {'Obj': 1, 'Obj': 2, , 'Obj': 2, ...}

    class_name_dict = {'Black Bear': 1, 'Ring-tailed Lemur': 2, 'Golden Monkey': 3, 'Tiger': 4,
                       'Takin': 5,'Lechwe': 6, 'Zebra': 7, 'Gnu': 8, 'Brown Bear': 9, 'Hamadryas Baboon': 10}

    parser.add_argument("--class_name", "--n", help="class name", type=str, default=class_name_dict)
    parser.add_argument("--input", "--i", help="json file path (labelme)", type=str,
                        default=r'path/to/labeled_dir')
    parser.add_argument("--output", "--o", help="output file path (coco format)", type=str,
                        default=r'path/to/dataset/output')
    parser.add_argument("--coco2yolo", "--c2y", help="convert coco to yolo format", default=True)
    parser.add_argument("--join_num", "--j", help="number of join", type=int, default=17)
    parser.add_argument("--train_ratio", "--trr", help="train split ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", "--vr", help="val split ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", "--ter", help="test split ratio", type=float, default=0.2)
    parser.add_argument("--enhance", "--e", help="dataset enhance", default=True)

    parser.add_argument("--check_kp", help="check keypints number", default=False)
    args = parser.parse_args()

    labelme_path = args.input
    saved_coco_path = args.output
    enhance_ = args.enhance

    if not os.path.exists("%s/coco/annotations/" % saved_coco_path):
        os.makedirs("%s/coco/annotations/" % saved_coco_path)
    if not os.path.exists("%s/coco/train/" % saved_coco_path):
        os.makedirs("%s/coco/train" % saved_coco_path)
    if not os.path.exists("%s/coco/val/" % saved_coco_path):
        os.makedirs("%s/coco/val" % saved_coco_path)
    if not os.path.exists("%s/coco/test/" % saved_coco_path):
        os.makedirs("%s/coco/test" % saved_coco_path)

    l2c_train = Labelme2coco(args)
    json_list_path = glob.glob(labelme_path + "/**/**.json", recursive=True)
    fake_cls_names = sorted({p(path).parent.stem for path in json_list_path})
    label_encoder = LabelEncoder()
    fake_cls_id = label_encoder.fit_transform(fake_cls_names)
    class_dict = dict(zip(fake_cls_names, fake_cls_id))
    fake_labels = [class_dict[p(path).parent.stem] for path in json_list_path]

    if args.check_kp:
        json_list_path = l2c_train.check_keypoints_num(json_list_path)
    train_path, val_path, test_path = l2c_train.split_dataset(json_list_path,
                                                              fake_labels,
                                                              args.train_ratio,
                                                              args.val_ratio,
                                                              args.test_ratio)

    if enhance_:
        print('Start transform please wait ...')
        train_path = l2c_train.enhance_img_json(train_path, args.input)
    print('{} for train'.format(len(train_path)),
          '\n{} for val'.format(len(val_path)),
          '\n{} for test'.format(len(test_path)))

    train_keypoints = l2c_train.to_coco(train_path)

    l2c_train.save_coco_json(train_keypoints, '%s/coco/annotations/train.json' % saved_coco_path)

    for file in train_path:
        try:
            shutil.copy(file.replace("json", "jpg"), "%s/coco/train/" % saved_coco_path)
        except:
            print(f'{file} No such file or directory!')
    for file in val_path:
        try:
            shutil.copy(file.replace("json", "jpg"), "%s/coco/val/" % saved_coco_path)
        except:
            print(f'{file} No such file or directory!')
    for file in test_path:
        try:
            shutil.copy(file.replace("json", "jpg"), "%scoco/test/" % saved_coco_path)
        except:
            print(f'{file} No such file or directory!')

    # SGMonkeyDataset
    l2c_val = Labelme2coco(args)
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '%scoco/annotations/val.json' % saved_coco_path)

    l2c_test = Labelme2coco(args)
    test_instance = l2c_test.to_coco(test_path)
    l2c_test.save_coco_json(test_instance, '%s/coco/annotations/test.json' % saved_coco_path)
    if args.coco2yolo:
        convert_coco(labels_dir=str(p(saved_coco_path) / 'coco' / 'annotations'),
                     save_dir=str(p(saved_coco_path).parent / f'{p(saved_coco_path).stem}2yolo'),
                     use_segments=False,
                     use_keypoints=False,
                     cls91to80=False)
        print('copy images to yolo dir')
        for dir in (p(saved_coco_path) / 'coco').iterdir():
            if dir.stem == 'train':
                shutil.copytree(str(p(saved_coco_path) / 'coco' / 'train'),
                                str(p(saved_coco_path).parent / f'{p(saved_coco_path).stem}2yolo' / 'images' / 'train'))
            elif dir.stem == 'val':
                shutil.copytree(str(p(saved_coco_path) / 'coco' / 'val'),
                                str(p(saved_coco_path).parent / f'{p(saved_coco_path).stem}2yolo' / 'images' / 'val'))
            elif dir.stem == 'test':
                shutil.copytree(str(p(saved_coco_path) / 'coco' / 'test'),
                                str(p(saved_coco_path).parent / f'{p(saved_coco_path).stem}2yolo' / 'images' / 'test'))
        print('done')