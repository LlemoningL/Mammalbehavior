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
 <a>
    <img src="images/pipeline.jpg" alt="Logo" width="auto" height="600">
  </a>
  <h3 align="center">An Automated AI Framework for Quantitative Measurement of Mammalian Behavior</h3>

  <p align="center">
    Jia Liu†, Tao Liu†, Zhengfeng Hu, Fan Wu, Zi Wang, Qi Gao, Wenjie Guo, Paul A. Garber, Derek Dunn, Colin A. Chapman, Ruliang Pan4, Tongzuo Zhang, Yang Zhao, Felix Guo, Shuang Yin, Gang He, Pengfei Xu, Baoguo Li, Songtao Guo
    <br />
    动物行为识别与定量测量的便捷框架! 
    </p>

[English](README.md) | 简体中文

</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>目录</summary>
  <ol>
    <li>
      <a href="#关于项目">关于项目</a>
    </li>
    <li>
      <a href="#开始">开始</a>
      <ul>
        <li><a href="#前置准备">前置准备</a></li>
        <li><a href="#安装方法">安装方法</a></li>
      </ul>
    </li>
    <li><a href="#使用方法">使用方法</a></li>
    <li><a href="#许可">许可</a></li>
    <li><a href="#联系方式">联系方式</a></li>
    <li><a href="#致谢">致谢</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## 关于项目
<table align="center">
  <tr>
    <td>
      <img src="images/demo (1).gif" alt="Golden Monkey" width="640" height="auto">
    </td>
    <td>
      <img src="images/demo (2).gif" alt="Golden Monkey" width="640" height="auto">
    </td>
    <td>
      <img src="images/demo (3).gif" alt="Golden Monkey" width="640" height="auto">
    </td>
  </tr>
<tr>
    <td>
      <img src="images/demo (4).gif" alt="Golden Monkey" width="640" height="auto">
    </td>
    <td>
      <img src="images/demo (5).gif" alt="Golden Monkey" width="640" height="auto">
    </td>
     <td>
      <img src="images/demo (6).gif" alt="Golden Monkey" width="640" height="auto">
    </td>
  </tr>
<tr>
    <td>
      <img src="images/demo (7).gif" alt="Golden Monkey" width="640" height="auto">
    </td>
    <td>
      <img src="images/demo (8).gif" alt="Golden Monkey" width="640" height="auto">
    </td>
    <td>
      <img src="images/demo (8).gif" alt="Golden Monkey" width="640" height="auto">
    </td>


  </tr>
</table>
<div align="center">
哺乳动物行为识别和测量框架推理演示视频
</div>
<br/>


我们提供了一个人工智能框架，可以自动识别和测量大中型哺乳动物的行为。用户可以使用我们提供的工具制作自己的数据集、进行训练和推理。

利用我们的框架，您可以在哺乳动物行为研究中完成以下工作：
* 目标检测和跟踪
* 姿态估计
* 面部识别
* 行为识别和测量


阅读 [Tutorials_zh-CN.md](Tutorials_zh-CN.md) 可以获得有关准备数据集、训练和推理的更多详细信息。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>


## 开始

运行代码需要英伟达显卡（NVIDIA GPU）.

### 前置准备


#### 步骤1. 从官方网站下载 [Anoconda](https://www.anaconda.com/) .
创建conda虚拟环境并设置python=3.8.
  ```sh
  conda create -n behavior python=3.8 -y
  conda activate behavior
  ```
#### 步骤2. 按照[Pytorch](https://pytorch.org/get-started/locally/)官网要求安装。
推荐使用Pytorch=1.8和与其匹配的CUDA版本。
  ```sh
  conda install pytorch torchvision -c pytorch
  ```
#### 步骤3. 克隆仓库.
  ```sh
  git clone https://github.com/LlemoningL/Mammalbehavior.git
  cd Mammalbehavior
  ```
### 安装方法
#### 步骤1.使用 MIM 安装 MMEngine 和 MMCV.
[//]: # (_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._)
  ```sh
  pip install -U openmim
  mim install mmengine
  mim install mmcv "mmcv>=2.0.1"
  ```
#### 步骤2. 安装 MMAction2 的依赖项。
安装
进入 `dependencies/mmaction2` 文件夹
  ```sh
  cd dependencies
  cd mmaction2
  pip install -v -e .
  ```
验证

下载配置文件和权重文件。
  ```sh
  mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .
  ```
推理演示验证。
  ```sh
  # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
  python demo/demo.py tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
    tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth \
    demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
  ```
你的终端会显示top-5标签及其对应的分数。
#### 步骤3. 安装MMPose的依赖项。
安装

退出mmpose文件夹，进入mmaction2文件夹。
  ```sh
  cd ..
  cd mmpose
  pip install -r requirements.txt
  pip install -v -e .
  ```
验证

下载配置文件和权重文件。
  ```sh
  mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192  --dest .
  ```
推理演示验证。
  ```sh
  python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
    --out-file vis_results.jpg \
    --draw-heatmap
  ```
你将在当前文件夹的中 `vis_results.jpg` 看到可视化结果。

#### 步骤4. 安装MMPretrain的依赖项。
安装

退出mmpose文件夹，进入mmpretrain文件夹。
  ```sh
  cd ..
  cd mmpretrain
  mim install -e .  
  ```

推理演示验证。
  ```sh
  python demo/image_demo.py demo/demo.JPEG resnet18_8xb32_in1k --device cpu
  ```
你将在终端中看到包含 `pred_label`、`pred_score` 和 `pred_class` 的结果字典。

#### 步骤5. 安装ultralytics的依赖项。
安装

退出mmpretrain文件夹，进入ultralytics文件夹。
  ```sh
  cd ..
  cd ultralytics
  pip install -e .  
  ```

推理演示验证。
  ```sh
  yolo predict model=yolov8n.pt source=../../images/zidane.jpg
  ```
你将在终端中看到推理结果。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

#### 步骤6 (可选). 安装 MMdepoly 和 torch2trt。
我们的框架支持tensorRT加速。如果您想使用它，请按照以下步骤进行安装。

安装

MMdeploy 可将 mmpose 和 mmpretrain 模型转换为至 tensorRT 文件，
[mmpose转换](https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/04-supported-codebases/mmpose.md) 
和 [mmpretrain 转换](https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/04-supported-codebases/mmpretrain.md). 
更多细节请查看[MMdeploy 官方仓库](https://github.com/open-mmlab/mmdeploy).

torch2trt 将 Pytorch 模型转为 tensorRT 文件。. 安装请查看 
[torch2trt 官方仓库](https://github.com/NVIDIA-AI-IOT/torch2trt).








<p align="right">(<a href="#readme-top">返回顶部</a>)</p>


<!-- USAGE EXAMPLES -->


## 使用方法

#### Demo
点击 [此处](*待添加*) 下载演示配置文件和权重。
```python

```

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 许可

以 MIT 许可发布。更多信息请参见 `LICENSE.txt`。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>



<!-- CONTACT -->
## 联系方式

Liu Jia - ljlemoning@163.com

申请数据集: </p> 
[https://github.com/ljlemoning/Animal-Behavior-Dataset](https://github.com/ljlemoning/Animal-Behavior-Dataset)



<p align="right">(<a href="#readme-top">返回顶部</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## 致谢

本仓库基于以下开源库构建，特别感谢以下开源库作者们的思路和贡献。

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab pre-training toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
- [YOLOv8](https://github.com/ultralytics/ultralytics): Ultralytics for a wide range of object detection and tracking, instance segmentation, image classification and pose estimation tasks.
<p align="right">(<a href="#readme-top">返回顶部</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 