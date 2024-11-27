<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">An Automated AI Framework for Quantitative Measurement of Mammalian Behavior</h3>

  <p align="center">
    <br />
    An easy-to-use framework for animal behavior recognition and quantitative measurement!
    </p>

English | [简体中文](Tutorials_zh-CN.md)

</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>     
    <li>
      <a href="#Make-your-own-dataset">Make your own dataset</a>
      <ul>        
        <li><a href="#Object-Detection">Object Detection</a></li>
      </ul>
      <ul>        
        <li><a href="#Pose-estimation">Pose estimation</a></li>
      </ul>
      <ul>        
        <li><a href="#Facial-recognition">Facial recognition</a></li>
      </ul>
      <ul>        
        <li><a href="#Behavior-recognition">Behavior recognition</a></li>
      </ul>
    </li>
    <li><a href="#Train">Train</a></li>
    <li><a href="#Inference">Inference</a></li>

  </ol>
</details>


## Make your own dataset
The models needed to implement the framework for behavioural recognition and measurement are object detection, 
object tracking, pose estimation, facial recognition and behavior recognition, and the detector for object tracking is 
the same as the object detection, so it will not be described separately. The following is a description of the dataset 
making method used for each model one by one:：
<br />
### Object Detection
Target detection is recommended to be labelled using [labelme](https://github.com/wkentaro/labelme).
Generate a file with a suffix of `.json` corresponding to the image file after labeled.
Use the `tools/dataset/labelme2coco-keypoint.py`
 script to convert the labeled dataset into `coco` format, modifying the `--class_name`, `--input`, 
`--output` parameters to your actual dataset information.
```python
#... 
#tools/dataset/labelme2coco-keypoint.py
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
```
Use `tools/dataset/coco2yolo.py` to convert it finally to the training format available for `YOLOv8`.
```python
#tools/dataset/coco2yolo.py
from ultralytics.data.converter import convert_coco

label_dir = '/path/to/labelme2coco-keypoint_output/coco/annotations'
output_dir = '/path/to/output'

convert_coco(labels_dir=label_dir,
             save_dir=output_dir,
             use_segments=False,
             use_keypoints=False,
             cls91to80=False)
```
<br />

### Pose estimation
Pose estimates were made using either the `coco` format or the `AP10k` format, both with 17 keypoints.
It is recommended to be labelled using [labelme](https://github.com/wkentaro/labelme). 
Use the `tools/dataset/labelme2coco-keypoint.py`
 script to convert the labeled dataset into `coco` format, modifying the `--class_name`, `--input`, 
`--output` parameters to your actual dataset information.
It is recommended to set `--check_kp` to `True` to check that the number of 
keypoints in the labeled file is correct.

```python
#... 
#tools/dataset/labelme2coco-keypoint.py
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
```
<br />

### Facial recognition
For the facial recognition task, only the target head and face images are used and the target head and face images are cropped using any cropping software. And place the images in the following folder format
. E.g. Obj1 is the object name and Obj1_img1.jpg is the picture name.
```text
Specie1
  |-- Obj1
      |-- Obj1_img1.jpg
      |-- Obj1_img2.jpg
      |-- Obj1_img3.jpg
      ...  
  ...
  |-- Objn
      |-- Objn_img1.jpg
      |-- Objn_img2.jpg
      |-- Objn_img3.jpg
      ...  
Specie2
  |-- Obj1
      |-- Obj1_img1.jpg
      |-- Obj1_img2.jpg
      |-- Obj1_img3.jpg
      ...  
  ...
  |-- Objn
      |-- Objn_img1.jpg
      |-- Objn_img2.jpg
      |-- Objn_img3.jpg
      ...  
  ```
`tools/dataset/Autodataset.py` run the following code, it will generate a dataset in the same directory as the input folder, and automatically split the cropped images into training set, validation set and test set according to the ratio of `7:1:2`.
The `datasettype` parameter `no_label` means unlabelled dataset and `with_label` means labelled dataset.

```python
from tools.dataset.Autodataset import AutoDataset

input = 'ptah/to/specie1'
output = None

ad = AutoDataset(input, output, train_size=0.7, val_size=0.1, test_size=0.2, datasetname='Autodataset')
ad.makedataset(datasettype='no_label')
```

`tools/dataset/inshoplike_dataset.py` run the following code, which will convert the above generated dataset to `Inshop` dataset format for facial recognition training.
```python
if __name__ == '__main__':
    folder_path = r"path/to/your/own/dataset"
    remove_spaces(folder_path)
    inshoplike_dataset(folder_path) 
```

<br />

### Behavior recognition
Behaviour recognition dataset building relies on object detection and pose estimation, 
so model training for object detection and pose estimation needs to be completed first.
Behavioural videos are partitioned into training, 
validation and test sets in the ratio of 7:1:2 using the following script.
```python
from tools.dataset.Autodataset import AutoDataset

input = 'ptah/to/specie_behabior_video'
output = None

ad = AutoDataset(input, output, train_size=0.7, val_size=0.1, test_size=0.2, datasetname='Autodataset')
ad.makedataset(datasettype='no_label') 

```
Using a small amount of data follow the tutorial above to train a target detection model and a pose estimation model. 
And use `tools/dataset/pose_extraction.py` to extract the key point information of the target in the behavioural video dataset that has been built above，and finally generate three `.pkl` files corresponding to the training set, validation set and test set.

The extraction of video keypoint information is set according to the following parameters:

```python
#tools/dataset/pose_extraction.py
if __name__ == '__main__':
    det_mdoel = 'path/to/detect_mdoel_weight'  # recommend using your trained YOLO model for body detection
    pose_config = 'path/to/pose_config'  # recommend using your pose estimate config
    pose_weight = 'path/to/pose_model_weight'  # recommend using your trained pose estiamte model
    target_type = 'Primates'  # choose one in ["Primates", "Artiodactyla", "Carnivora", "Perissodactyla"]
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

```
The `tools/behavior_label.json` holds the behavioural categories of the species, e.g. 
`Primates` has `Grooming`, `Feeding`, `Resting`, `Walking`,
which can be modified to suit your own data.
#### `tools/behavior_label.json` will be used throughout the dataset making, training and inference sessions, so please take care to set them up correctly.
```json
// tools/behavior_label.json
{
  "Primates": {"categories": ["Grooming", "Feeding", "Resting", "Walking"]},
  "Artiodactyla": {"categories": ["Standing", "Feeding", "Resting", "Walking"]},
  "Carnivora": {"categories": ["Resting", "Standing", "Walking", "Feeding"]},
  "Perissodactyla": {"categories": ["Feeding", "Standing", "Walking"]}
}

```






## Train
The framework is trained using `tools/train.py`. The main arguments are `cfg` and `-task`, the rest can be found in `tools/train.py`.
```python
# tools/train.py
def parser_args():

    parser = ArgumentParser(description='Process video.')
    parser.add_argument('cfg', help='Path to config file'),
    parser.add_argument('--task',
                        choices=['face', 'track', 'pose', 'behavior', 'faceid'],
                        help='Choose a task to train')
    ...

```


The `cfg` fix receives `configs/train_cfg.yaml` with the following contents:
```yaml
# Framework configs

TRAIN:
  # path to weight of YOLOv8 detect model
  FACE:
    cfg: ../configs/yolo/datasets/10sp_facedetect.yaml

  # path to weight of YOLOv8 track model
  BODY:
    cfg: ../configs/yolo/datasets/10sp_bodydetect.yaml

  # path to config and weight of mmpose model
  POSE:
    cfg: ../configs/pose/animal_2d_keypoint/topdown_heatmap/ap10k/td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.py

  # path to config and weight of mmaction model
  BEHAVIOR:
    cfg: ../configs/behavior/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py

  # path to config and weight of mmpretrain model
  FACEID:
    cfg: ../configs/faceid/arcface/resnet50-arcface_8xb32_inshop_custom.py
```


<br />`--task` The framework automatically selects the corresponding configuration file when receiving the following parameters. <br />

<br />`face` indicates that the face detection task is performed, and the framework automatically matches `TRAIN.FACE`<br />
```yaml
# Framework configs

TRAIN:
  # path to weight of YOLOv8 detect model
  FACE:
    cfg: ../configs/yolo/datasets/10Sp_facedetect.yaml
```
Just replace `path` with your own data path and `names` with your own category.
```yaml
#../configs/yolo/datasets/10Sp_facedetect.yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: path/to/dataset_root  # dataset root dir
train: images/train  # train images (relative to 'path') 4 images
val: images/val  # val images (relative to 'path') 4 images
test: images/test # test images (optional)
verbose: False
# Classes
names:
  0: Black Bear
  1: Ring-tailed Lemur
  2: Golden Monkey
  ...

```

<br />`track` indicates that the body detection task is performed, sharing the tracker with target tracking, and the frame automatically matches `TRAIN.BODY`.<br />
```yaml
# Framework configs

TRAIN:
  # path to weight of YOLOv8 track model
  BODY:
    cfg: ../configs/yolo/datasets/10Sp_bodydetect.yaml
```
ust replace `path` with your own data path and `names` with your own category.
```yaml
#../configs/yolo/datasets/10Sp_bodydetect.yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: path/to/dataset_root  # dataset root dir
train: images/train  # train images (relative to 'path') 4 images
val: images/val  # val images (relative to 'path') 4 images
test: images/test # test images (optional)
verbose: False
# Classes
names:
  0: Black Bear
  1: Ring-tailed Lemur
  2: Golden Monkey
  ···

```

<br />`pose` indicates that the pose estimation task is performed and the framework automatically matches `TRAIN.POSE`.<br />
```yaml
# Framework configs

TRAIN:
# path to config and weight of mmpose model
  POSE:
    cfg: ../configs/pose/animal_2d_keypoint/topdown_heatmap/ap10k/Fourleg_td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.py
```
To open `Fourleg_td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.py`, it is usually sufficient to replace the
`data_root`, to your own data path.

```python
# ../configs/pose/animal_2d_keypoint/topdown_heatmap/ap10k/Fourleg_td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.py
...
# base dataset settings
dataset_type = 'AP10KDataset'
data_mode = 'topdown'
data_root = 'path/to/dataset/coco/'
...
```
In the case of primates that use the `coco` keypoint format for pose estimation use `Primate_td-hm_hrnet-w32_8xb64-210e_coco-256x192.py`, 
and replace `dataset_type = 'CocoDataset'` 

<br />`behavior` indicates that a behaviour recognition task is performed and the framework automatically matches `TRAIN.BEHAVIOR`.<br />

```yaml
# Framework configs

TRAIN:
# path to config and weight of mmaction model
  BEHAVIOR:
    cfg: ../configs/behavior/skeleton/posec3d/Primate_slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py
```
To open `Primate_slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py`, it is usually sufficient to replace the
`train_ann_file`, `val_ann_file`, `test_ann_file` to your own data path, and `num_classes` to your own behavioural classes.

```python
# ../configs/behavior/skeleton/posec3d/Primate_slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py
...
        in_channels=512,
        num_classes=4, #your own classes
        dropout_ratio=0.5,
        average_clips='prob'))

dataset_type = 'PoseDataset'
train_ann_file = f'/path/to/train.pkl'
val_ann_file = f'/path/to/val.pkl'
test_ann_file = f'/path/to/test.pkl'

...
```
`4Leg_slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py` in the case of four leg using `AP10k` for pose estimation, the rest of the settings are consistent.

<br />`faceid` indicates that the facial recognition task is performed and the framework automatically matches `TRAIN.FACEID`.<br />
```yaml
# Framework configs

TRAIN:
# path to config and weight of mmpretrain model
  FACEID:
    cfg: ../configs/faceid/arcface/Mammal_resnet50-arcface_8xb32_inshop.py
```
To open `Mammal_resnet50-arcface_8xb32_inshop.py`, it is usually sufficient to replace
`dataroot` to your own datapath, and `numclass` to your own category.
```python
# ../configs/faceid/arcface/Mammal_resnet50-arcface_8xb32_inshop.py
...
dataroot = 'path/to/dataset'
numclass = 3 # your own classes
batch_size = 32
work_dir = f'../animal_behavior_runs_result/exp1/faceid/'

...
```
After setting up the configuration file, go to the `tools` folder, replace the parameters with your actual path, and run the following code to start training.
```sh
python train.py configs/train_cfg.yaml --task {task} --work_dir {work_dir}
```


## Inference
Framework inference uses `main.py`. It mainly receives the arguments `cfg` inference configuration file, `video` video path, `target_type` target type, selectable among `[‘Primates’, ‘Artiodactyla’, ‘Carnivora’, ‘Perissodactyla’]`.
```text
'Primates' for 'Golden Snub-nosed Monkey', 'Ring-tailed Lemur', 'Hamadryas Baboon'
'Artiodactyla' for 'Takin', 'Gnu', 'Lechwe'
'Carnivora' for 'Tiger', 'Black Bear', 'Brown Bear'
'Perissodactyla' for 'Zebra"
```
```python
def parser_args():

    parser = ArgumentParser(description='Process video.')
    parser.add_argument('cfg', help='Path to config file'),
    parser.add_argument('video', help='Path to the video file')
    parser.add_argument('target_type', help='Type of target')
    parser.add_argument('--interval', default=3, type=int,
                        help='Interval of recognised behaviour, in seconds')
    parser.add_argument('--trt', default=False, action='store_true',
                        help='Whether to use TRT engine')
    parser.add_argument('--show', default=False, action='store_true',
                        help='Whether to show inferenced frame')
    parser.add_argument('--save_vid', default=False, action='store_true',
                        help='Whether to save inferenced video')
    parser.add_argument('--show_fps', default=False, action='store_true',
                        help='Whether to show inference fps')
    parser.add_argument('--behavior_label',
                        default='./tools/behavior_label.json',
                        type=str,
                        help='Path to behavior label file')
    args = parser.parse_args()
    cfgs = default()
    cfgs.merge_from_file(args.cfg)
    cfgs.freeze()

    return cfgs, args
```
`cfg` is fixed to `configs/inference_cfg.yaml`, replacing each configuration or weight path with your own.
The `trt_engine` for each model is optional and can be empty.
```yaml
# configs/inference_cfg.yaml
# Framework configs

MODEL:
  # path to weight of YOLOv8 detect model
  FACE:
    weight: path/to/face/detect/weight.pt
    trt_engine: path/to/face/detect/weight.engine

  # path to weight of YOLOv8 track model
  BODY:
    weight: path/to/body/detect/weight.pt
    trt_engine: path/to/body/detect/weight_trt.engine
    reid_encoder:
    r_e_trt_engine:
    with_reid: False

  # path to config and weight of mmpose model
  POSE:
    cfg: path/to/pose/estimate/td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.py
    weight: path/to/pose/estimate/weight.pth
    trt_engine: path/to/pose/estimate/weight_trt.engine
    deploy_cfg: path/to/pose/estimate/pose-detection_tensorrt_static-256x256.py

  # path to config and weight of mmaction model
  BEHAVIOR:
    cfg:  path/to/behavior/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py
    weight:  path/to/behavior/weight.pth
    trt_engine: # todo convert to TensorRT engine

  # path to config and weight of mmpretrain model
  FACEID:
    cfg: path/to/faceid/resnet50-arcface_8xb32_inshop_custom.py
    weight: path/to/faceid/weight.pth    
    #  [prototype] usually is a directory with one folder for each target, 
    #  folder name being the target's name.
    prototype: path/to/faceid/prototype
    prototype_cache: path/to/faceid/prototype_cache.pth  # a generated prototype *.pth file
    trt_engine: path/to/faceid/weight_trt.engine

#path to save the output
OUTPUT:
  path: 'VideoOutput'

# type of target
DATA:
  label: Perissodactyla  # "Choose one in ["Primates", "Artiodactyla", "Carnivora", "Perissodactyla"]

```
Set up the configuration file, enter the project folder, replace the parameters with your actual path or target type, and run the following code to start inference.
```bash
python main.py configs/inference_cfg.yaml {video} --target_type {target_type} --show_fps
```



<p align="right">(<a href="#readme-top">back to top</a>)</p>



