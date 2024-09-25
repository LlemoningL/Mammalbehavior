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
    Jia Liu†, Tao Liu†, Zhengfeng Hu, Fan Wu, Zi Wang, Qi Gao, Wenjie Guo, Paul A. Garber, Derek Dunn, Colin A. Chapman, Ruliang Pan4, Tongzuo Zhang, Yang Zhao, Felix Guo, Shuang Yin, Gang He, Pengfei Xu, Baoguo Li, Songtao Guo
    <br />
    动物行为识别与定量测量的便捷框架! 
    </p>

[English](Tutorials.md) | 简体中文

</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>目录</summary>
  <ol>     
    <li>
      <a href="#制作自己的数据集">制作自己的数据集</a>
      <ul>        
        <li><a href="#目标检测">目标检测</a></li>
      </ul>
      <ul>        
        <li><a href="#姿态估计">姿态估计</a></li>
      </ul>
      <ul>        
        <li><a href="#面部识别">面部识别</a></li>
      </ul>
      <ul>        
        <li><a href="#行为识别">行为识别</a></li>
      </ul>
    </li>
    <li><a href="#训练">训练</a></li>
    <li><a href="#推理">推理</a></li>

  </ol>
</details>


## 制作自己的数据集
框架实现行为识别与测量需用到的模型有目标检测、目标追踪、姿态估计、面部识别和行为识别，
目标追踪的检测器与目标检测为同一个，不再单独叙述。
下面就各个模型需要用到的数据集制作方法逐一介绍：
<br />

### 目标检测
目标检测建议使用[labelme](https://github.com/wkentaro/labelme)进行标注。
标注后生成与图片文件对应且后缀为 `.json` 的文件。使用 `tools/dataset/labelme2coco-keypoint.py`
 脚本将标注完成的数据集转化为 `coco` 格式，修改`--class_name`, `--input`, `--output`参数为自己实际数据集信息。

```python
#... 
#tools/dataset/labelme2coco-keypoint.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #  bounding class and id with dict
    

    class_name_dict = {'Black Bear': 1, 'Ring-tailed Lemur': 2, 'Golden Monkey': 3, 'Tiger': 4,
                       'Takin': 5,'Lechwe': 6, 'Zebra': 7, 'Gnu': 8, 'Brown Bear': 9, 'Hamadryas Baboon': 10}
    parser.add_argument("--class_name", "--n", help="class name", type=str, default=class_name_dict)
    parser.add_argument("--input", "--i", help="json file path (labelme)", type=str,
                        default=r'path/to/labeled_Jsonandimg_folder')
    parser.add_argument("--output", "--o", help="output file path (coco format)", type=str,
                        default=r'path/to/output_folder')
    parser.add_argument("--join_num", "--j", help="number of join", type=int, default=17)
    parser.add_argument("--train_ratio", "--trr", help="train split ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", "--vr", help="val split ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", "--ter", help="test split ratio", type=float, default=0.2)
    parser.add_argument("--enhance", "--e", help="dataset enhance", default=True)

    parser.add_argument("--check_kp", help="check keypints number", default=False)
    args = parser.parse_args()
```
使用 `tools/dataset/coco2yolo.py` 将其最终转为 `YOLOv8` 可用的训练格式。
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

### 姿态估计
姿态估计使用 `coco` 格式或者 `AP10k` 格式，均为17个关键点。
建议使用[labelme](https://github.com/wkentaro/labelme)进行标注。
标注后生成与图片文件对应且后缀为 `.json` 的文件。使用 `tools/dataset/labelme2coco-keypoint.py`
 脚本将标注完成的数据集转化为 `coco` 格式，修改`--class_name`, `--input`, `--output`参数为自己实际数据集信息。
建议设置 `--check_kp` 为 `True`，检查标注文件关键点数量是否正确。

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

### 面部识别
面部识别任务，只使用目标头面部图片，使用任意裁图软件裁剪目标头面部图片。并按照如下文件夹格式放置图片
。如：Obj1为对象名称，Obj1_img1.jpg为图片名称。
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
`tools/dataset/Autodataset.py` 运行如下代码, 会在输入文件夹同级目录生成数据集，自动将裁剪好的图片按照 `7:1:2` 的比例分割为训练集、验证集和测试集。
其中 `datasettype`参数 `no_label`，表示无标签数据集， `with_label` 表示有标签数据集。

```python
from tools.dataset.Autodataset import AutoDataset

input = 'ptah/to/specie1'
output = None

ad = AutoDataset(input, output, train_size=0.7, val_size=0.1, test_size=0.2, datasetname='Autodataset')
ad.makedataset(datasettype='no_label') 

```
<br />

`tools/dataset/inshoplike_dataset.py` 运行如下代码, 会将上面生成的数据集转换为 `Inshop` 数据集格式，用于面部识别训练。

```python
if __name__ == '__main__':
    folder_path = r"path/to/your/own/dataset"
    remove_spaces(folder_path)
    inshoplike_dataset(folder_path) 
```
<br />

### 行为识别
行为识别数据集建立依赖于目标检测和姿态估计，因此需要先完成目标检测和姿态估计的模型训练。
使用如下脚本将行为视频按照7:1:2的比例分割为训练集、验证集和测试集。
```python
from tools.dataset.Autodataset import AutoDataset

input = 'ptah/to/specie_behabior_video'
output = None

ad = AutoDataset(input, output, train_size=0.7, val_size=0.1, test_size=0.2, datasetname='Autodataset')
ad.makedataset(datasettype='no_label') 

```
使用少量数据依照上述教程，训练一个目标检测模型和姿态估计模型。
并使用 `tools/dataset/pose_extraction.py` 
提取行为上述已经建好的行为视频数据集中目标的关键点信息，最终生成三个 `.pkl` 后缀的文件，分别对应训练集、验证集和测试集。
提取视频关键点信息按照如下参数设置：
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
`tools/behavior_label.json` 中保存了物种的行为类别，如 `Primates` 的行为类别有 `Grooming`, `Feeding`, `Resting`, `Walking`。
可以根据自己的数据修改，
#### `tools/behavior_label.json` 文件将会在整个数据集制作、训练和推理过程中使用，请注意正确设置。
```json
// tools/behavior_label.json
{
  "Primates": {"categories": ["Grooming", "Feeding", "Resting", "Walking"]},
  "Artiodactyla": {"categories": ["Standing", "Feeding", "Resting", "Walking"]},
  "Carnivora": {"categories": ["Resting", "Standing", "Walking", "Feeding"]},
  "Perissodactyla": {"categories": ["Feeding", "Standing", "Walking"]}
}

```


## 训练
框架训练使用 `tools/train.py`。主要接收参数为 `cfg` 和 `--task`,其余参数可在 `tools/train.py` 中查看.
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


`cfg` 固定接收 `configs/train_cfg.yaml`，其内容如下：
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


<br />`--task` 接收如下参数时，框架会自动选择对应的配置文件。<br />

<br />`face` 表示执行面部检测任务, 框架自动匹配 `TRAIN.FACE`<br />
```yaml
# Framework configs

TRAIN:
  # path to weight of YOLOv8 detect model
  FACE:
    cfg: ../configs/yolo/datasets/10Sp_facedetect.yaml
```
将 `path` 替换为自己的数据路径，并将`names`替换为自己的类别即可
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


```

<br />`track` 表示执行身体检测任务，与目标追踪共用追踪器，框架自动匹配 `TRAIN.BODY`<br />
```yaml
# Framework configs

TRAIN:
  # path to weight of YOLOv8 track model
  BODY:
    cfg: ../configs/yolo/datasets/10Sp_bodydetect.yaml
```
将 `path` 替换为自己的数据路径，并将`names`替换为自己的类别即可
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


```

<br />`pose` 表示执行姿态估计任务，框架自动匹配 `TRAIN.POSE`<br />
```yaml
# Framework configs

TRAIN:
# path to config and weight of mmpose model
  POSE:
    cfg: ../configs/pose/animal_2d_keypoint/topdown_heatmap/ap10k/Fourleg_td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.py
```
打开`Fourleg_td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.py`, 通常只要将
`data_root`, 替换为自己的数据路径即可。

```python
# ../configs/pose/animal_2d_keypoint/topdown_heatmap/ap10k/Fourleg_td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.py
...
# base dataset settings
dataset_type = 'AP10KDataset'
data_mode = 'topdown'
data_root = 'path/to/dataset/coco/'
...
```
如果是使用`coco`关键点格式进行姿态估计的灵长类类动物则使用`Primate_td-hm_hrnet-w32_8xb64-210e_coco-256x192.py`, 
并替换`dataset_type = 'CocoDataset'` 

<br />`behavior` 表示执行行为识别任务， 框架自动匹配 `TRAIN.BEHAVIOR`<br />

```yaml
# Framework configs

TRAIN:
# path to config and weight of mmaction model
  BEHAVIOR:
    cfg: ../configs/behavior/skeleton/posec3d/Primate_slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py
```
打开`Primate_slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py`, 通常只要将
`train_ann_file`, `val_ann_file`, `test_ann_file` 替换为自己的数据路径, `num_classes` 替换为自己的行为类别数即可。

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
如果是使用`AP10k`进行姿态估计的四足类动物则使用`4Leg_slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py`, 其余设置一致。

<br />`faceid` 表示执行面部识别任务， 框架自动匹配 `TRAIN.FACEID`<br />
```yaml
# Framework configs

TRAIN:
# path to config and weight of mmpretrain model
  FACEID:
    cfg: ../configs/faceid/arcface/Mammal_resnet50-arcface_8xb32_inshop.py
```
打开`Mammal_resnet50-arcface_8xb32_inshop.py`, 通常只要将
`dataroot`,替换为自己的数据路径, `numclass` 替换为自己的类别数即可。
```python
# ../configs/faceid/arcface/Mammal_resnet50-arcface_8xb32_inshop.py
...
dataroot = 'path/to/dataset'
numclass = 3 # your own classes
batch_size = 32
work_dir = f'../animal_behavior_runs_result/exp1/faceid/'

...
```
设置好配置文件后， 进入`tools`文件夹，将参数替换为自己的实际路径，运行如下代码开始训练。
```sh
python train.py configs/train_cfg.yaml --task {task} --work_dir {work_dir}
```


## 推理
框架推理使用 `main.py`。主要接收参数 `cfg`推理配置文件, `video`视频路径, `target_type`目标类型，可在`["Primates", "Artiodactyla", "Carnivora", "Perissodactyla"]`中选择。
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
`cfg` 固定接收 `configs/inference_cfg.yaml`， 将各项配置或者权重路径替换为自己的路径。
各个模型的`trt_engine` 为可选项，可以为空。
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
设置好配置文件,进入项目文件夹，将参数替换为自己的实际路径或者目标类型，运行如下代码开始推理。
```bash
python main.py configs/inference_cfg.yaml {video} --target_type {target_type} --show_fps
```



<p align="right">(<a href="#readme-top">返回顶部</a>)</p>



