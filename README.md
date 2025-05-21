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
    <br />
    An easy-to-use framework for animal behavior recognition and quantitative measurement! 
    </p>

English | [简体中文](README_zh-CN.md)

</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<table align="center">
  <tr>
    <td>
      <img src="images/demo (1).gif" alt="image1" width="300" height="auto">
    </td>
    <td>
      <img src="images/demo (2).gif" alt="image2" width="300" height="auto">
    </td>
    <td>
      <img src="images/demo (3).gif" alt="image3" width="300" height="auto">
    </td>
  </tr>
<tr>
    <td>
      <img src="images/demo (4).gif" alt="image4" width="300" height="auto">
    </td>
    <td>
      <img src="images/demo (5).gif" alt="image5" width="300" height="auto">
    </td>
     <td>
      <img src="images/demo (6).gif" alt="image6" width="300" height="auto">
    </td>
  </tr>
<tr>
    <td>
      <img src="images/demo (7).gif" alt="image7" width="300" height="auto">
    </td>
    <td>
      <img src="images/demo (9).gif" alt="image8" width="300" height="auto">
    </td>
    <td>
      <img src="images/demo (8).gif" alt="image9" width="300" height="auto">
    </td>


  </tr>
</table>
<div align="center">
Demo inference video of framework for mammalian behavior recognition and measurement
</div>
<br/>

<br/>

We provide an AI framework that enables automatic behavioural recognition and measurement in medium and large mammals. Users can make their own datasets, training, and inference using the tools we provide.

You can do the following things in mammal behavior research with our framework:
* Obeject detection and tracking
* Pose estimation
* Face recognition
* behavior recognition and measurement


Use the [Tutorials.md](Tutorials.md) to get more details for prepare datasets, train and inference.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Getting Started

NVIDIA GPU is required.

### Prerequisites

#### Step1. Download [Anoconda](https://www.anaconda.com/) from the official website.
Create conda enviroment and set python version to 3.8.
  ```sh
  conda create -n behavior python=3.8 -y
  conda activate behavior
  ```
#### Step2. Install Pyroch following [official website](https://pytorch.org/get-started/locally/).
Pytorch=1.8 and it's compatible CUDA version is recommended.
  ```sh
  conda install pytorch torchvision -c pytorch
  ```
#### Step3. Clone the repo.
  ```sh
  git clone https://github.com/LlemoningL/Mammalbehavior.git
  cd Mammalbehavior
  ```
### Installation
#### Step1. Install MMEngine and MMCV using MIM.
[//]: # (_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._)
  ```sh
  pip install -U openmim
  mim install mmengine
  mim install mmcv "mmcv>=2.0.1"
  ```
#### Step2. Install dependencies of MMAction2.
Install

Enter  `dependencies/mmaction2` 
  ```sh
  cd dependencies
  cd mmaction2
  pip install -v -e .
  ```
Verify

Download the config and checkpoint files.
  ```sh
  mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .
  ```
Verify the inference demo.
  ```sh
  # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
  python demo/demo.py tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
    tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth \
    demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
  ```
You will see the top-5 labels with corresponding scores in your terminal.
#### Step3. Install dependencies of MMPose.
Install

Exiting mmaction2 folder, enter mmpose folder.
  ```sh
  cd ..
  cd mmpose
  pip install -r requirements.txt
  pip install -v -e .
  ```
Verify

Download the config and checkpoint files.
  ```sh
  mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192  --dest .
  ```
Verify the inference demo.
  ```sh
  python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
    --out-file vis_results.jpg \
    --draw-heatmap
  ```
 you will be able to get the visualization result from vis_results.jpg in your current folder.
#### Step4. Install dependencies of MMPretrain.
Install

Exiting MMpose folder, enter MMpretrain folder.
  ```sh
  cd ..
  cd mmpretrain
  mim install -e .  
  ```

Verify the inference demo.
  ```sh
  python demo/image_demo.py demo/demo.JPEG resnet18_8xb32_in1k --device cpu
  ```
 You will see the output result dict including `pred_label`, `pred_score` and `pred_class`` in your terminal.

#### Step5. Install dependencies of Ultralytics.
Install

Exiting mmpretrain, enter ultralytics folder.
  ```sh
  cd ..
  cd ultralytics
  pip install -e .  
  ```

Verify the inference demo.
  ```sh
  yolo predict model=yolov8n.pt source=../../images/zidane.jpg
  ```
You will see a message on the terminal with the result of the inference.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->

#### Step6 (optional). Install dependencies of MMdepoly and torch2trt.
Our framework support tensorRT speedup. if you want to use it, please follow the steps below to install them.

Install

MMdeploy can export mmpose and mmpretrain model to tensorRT engine.
Installation please refer to 
[mmpose export](https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/04-supported-codebases/mmpose.md) 
and [mmpretrain export](https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/04-supported-codebases/mmpretrain.md). 
More detaials please refer to [MMdeploy official repo](https://github.com/open-mmlab/mmdeploy).

torch2trt can convert pytorch model to tensorRT engine. Installation please refer to 
[torch2trt official repo](https://github.com/NVIDIA-AI-IOT/torch2trt).








<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->


## Usage

#### Demo
Click [here](https://drive.google.com/file/d/1BtxQVM13vq1qSlbkWH7qYvTwOQkwC5Bm/view?usp=sharing) to download demo config and checkpoint files.
Unzip the downloaded files into the `Demo` folder, and you may choose to run the following commands:
<br />Sichuan snub-nosed monkey
```sh
python main.py Demo/configs/inference_cfg_gm.yaml Demo/videos/gm.mp4 Primates --interval 2 --show_fps --show --save_vid
```
Tiger
```sh
python main.py Demo/configs/inference_cfg_ti.yaml Demo/videos/ti.mp4 Carnivora --interval 2 --show_fps --show --save_vid
```
Brown bear
```sh
python main.py Demo/configs/inference_cfg_brb.yaml Demo/videos/brb.mp4 Carnivora --interval 2 --show_fps --show --save_vid
```
After running, the results will be generated in the `VideoOutput` directory at the same level as `main.py`. The `--show`
parameter requires a display to show the inference visuals; if there is no display, you can remove this parameter.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Liu Jia - ljlemoning@163.com

Requires for framework dataset: </p> 
[check this webpage, and send Email to professor Guo](http://biology.nwu.edu.cn/info/1567/6684.htm)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Citation
If you find this repository helpful, please cite our [paper](https://doi.org/10.1111/1749-4877.12985):

```bibtex
@article{liu2025automated,
  title={An Automated AI Framework for Quantitative Measurement of Mammalian Behavior},
  author={Liu, Jia and Liu, Tao and Hu, Zhengfeng and Wu, Fan and Guo, Wenjie and Wu, Haojie and Wang, Zhan and Men, Yiyi and Yin, Shuang and Garber, Paul A and others},
  journal={Integrative Zoology},
  year={2025},
  publisher={Wiley Online Library}
}
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This repository is built on the following open source libraries, with special thanks to the authors of the following open source libraries for their ideas and contributions.

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab pre-training toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
- [YOLOv8](https://github.com/ultralytics/ultralytics): Ultralytics for a wide range of object detection and tracking, instance segmentation, image classification and pose estimation tasks.
<p align="right">(<a href="#readme-top">back to top</a>)</p>



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
