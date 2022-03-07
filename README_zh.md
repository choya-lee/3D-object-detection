# SDCND : 传感器融合和跟踪

[English](./README.md)

这是Udacity自动驾驶汽车工程师纳米学位课程第二门课程的项目：传感器融合与跟踪。

在本项目中，您将融合来自激光雷达和相机的测量结果，并随时间跟踪车辆。您将使用Waymo Open数据集中的真实数据，检测3D点云中的对象，并应用扩展卡尔曼滤波器进行传感器融合和跟踪。

<img src="img/img_title_1.jpeg"/>

该项目由两个主要部分组成：

1. **目标检测**：在这一部分中，基于三维点云的鸟瞰视角，采用深度学习的方法来检测LiDAR数据中的车辆。同时，采用一系列性能指标对检测方法的性能进行了评估。
2. **目标跟踪**：在这一部分中，基于激光雷达检测和摄像机检测的融合，使用扩展卡尔曼滤波来跟踪车辆随时间的变化。实现了数据关联和轨迹管理。

下图包含组成算法的数据流和各个步骤的概要。

<img src="img/img_title_2_new.png"/>

此外，项目代码还包含各种任务，这些任务在代码中逐步进行了详细说明。关于算法和任务的更多信息可以在Udacity课堂上找到。

## 项目文件结构

📦project<br>
 ┣ 📂dataset --> 包含Waymo开放数据集序列 <br>
 ┃<br>
 ┣ 📂misc<br>
 ┃ ┣ evaluation.py --> 用于跟踪可视化和RMSE计算的绘图功能<br>
 ┃ ┣ helpers.py --> misc. helper功能, 例如用于加载/保存二进制文件<br>
 ┃ ┗ objdet_tools.py --> 没有学生任务的物体检测功能<br>
 ┃ ┗ params.py --> 追踪部分的参数文件<br>
 ┃ <br>
 ┣ 📂results --> 带有预先计算的中间结果的二进制文件<br>
 ┃ <br>
 ┣ 📂student <br>
 ┃ ┣ association.py --> 用于将测量结果分配给轨道的数据关联逻辑，包括学生任务 <br>
 ┃ ┣ filter.py --> 扩展卡尔曼滤波器的实施，包括学生任务 <br>
 ┃ ┣ measurements.py --> 摄像机和激光雷达的传感器和测量课程，包括学生任务  <br>
 ┃ ┣ objdet_detect.py --> 基于模型的物体检测，包括学生任务  <br>
 ┃ ┣ objdet_eval.py -->  物体检测的性能评估，包括学生任务 <br>
 ┃ ┣ objdet_pcl.py --> 点云功能，例如用于鸟瞰，包括学生的任务。 <br>
 ┃ ┗ trackmanagement.py --> 追踪和追踪管理，包括学生任务 <br>
 ┃ <br>
 ┣ 📂tools --> 外部工具<br>
 ┃ ┣ 📂objdet_models --> 物体检测的模型<br>
 ┃ ┃ ┃<br>
 ┃ ┃ ┣ 📂darknet<br>
 ┃ ┃ ┃ ┣ 📂config<br>
 ┃ ┃ ┃ ┣ 📂models --> darknet / yolo model class and tools<br>
 ┃ ┃ ┃ ┣ 📂pretrained --> 将预训练的模型文件复制到这里<br>
 ┃ ┃ ┃ ┃ ┗ complex_yolov4_mse_loss.pth<br>
 ┃ ┃ ┃ ┣ 📂utils --> 各种帮助函数<br>
 ┃ ┃ ┃<br>
 ┃ ┃ ┗ 📂resnet<br>
 ┃ ┃ ┃ ┣ 📂models --> fpn_resnet model class and tools<br>
 ┃ ┃ ┃ ┣ 📂pretrained --> 将预训练的模型文件复制到这里 <br>
 ┃ ┃ ┃ ┃ ┗ fpn_resnet_18_epoch_300.pth <br>
 ┃ ┃ ┃ ┣ 📂utils --> 各种帮助函数<br>
 ┃ ┃ ┃<br>
 ┃ ┗ 📂waymo_reader --> 用于轻量级加载Waymo序列的函数<br>
 ┃<br>
 ┣ basic_loop.py<br>
 ┣ loop_over_dataset.py<br>



## 本地运行安装说明

### 克隆项目

要创建项目的本地副本，请单击“代码”，然后单击“下载 ZIP”。或者，您当然可以为此目的使用 GitHub Desktop 或 Git Bash。

### Python

该该项目使用 Python 3.7 编写。请确保您的本地安装等于或高于此版本。

### Package Requirements

该项目所需的所有依赖项都已列在文件`requirements.txt`中。你可以使用pip逐一安装它们，也可以使用命令`pip3 install -r requirements.txt` 一次性安装它们。

### Waymo开放数据集阅读器

Waymo开放数据集阅读器是一个非常方便的工具箱，允许你从Waymo开放数据集中访问序列，而不需要安装官方工具箱附带的所有重量级依赖。安装说明可以在`tools/waymo_reader/README.md`中找到。

### Waymo开放数据集文件

这个项目利用三个不同的序列来说明物体检测和跟踪的概念。它们是：

- Sequence 1 : `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`
- Sequence 2 : `training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord`
- Sequence 3 : `training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord`

要下载这些文件，你必须先在Waymo开放数据集注册。[Open Dataset - Waymo](https://waymo.com/open/terms)，如果你还没有，请确保将 "Udacity "作为你的机构。

一旦你这样做了，请[点击这里](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files)访问存放所有序列的谷歌云容器。一旦你被Waymo批准访问（可能需要48小时），你可以下载各个序列。

上面列出的序列可以在 "training "文件夹中找到。请下载它们并将 `tfrecord` 文件放到本项目的 `dataset` 文件夹中。


### 预训练模型

本项目中使用的物体检测方法使用了由原作者提供的预训练模型。它们可以在[这里](https://drive.google.com/file/d/1Pqx7sShlqKSGmvshTYbNDcUEYyZwfn3A/view?usp=sharing) （darknet）和 [这里](https://drive.google.com/file/d/1RcEfUIF1pzDZco8PJkZ10OL-wLL2usEj/view?usp=sharing)（fpn_resnet）下载。一旦下载，请将模型文件复制到以下路径中 `/tools/objdet_models/darknet/pretrained` 和`/tools/objdet_models/fpn_resnet/pretrained` 

### 使用预先计算结果

在主文件`loop_over_dataset.py`中，你可以选择算法的哪些步骤应该被执行。如果你想调用一个特定的函数，你只需要把相应的字符串字头添加到下面的一个列表中。

- `exec_data` : 控制与传感器数据有关的步骤的执行
  - `pcl_from_rangeimage` 将Waymo开放数据的范围图像转化为三维点云
  - `load_image` 返回前置摄像头的图像

- `exec_detection` : 控制基于模型的三维物体检测的哪些步骤被执行
  - `bev_from_pcl` 将点状云转化为固定尺寸的鸟瞰视角
  - `detect_objects` 执行实际检测，并返回一组对象（只有车辆）
  - `validate_object_labels` 决定哪些地面真实标签应该被考虑（例如，基于难度或可见度）
  - `measure_detection_performance` 包含评估单帧检测性能的方法

如果你没有在列表中包括一个特定的步骤，预先计算的二进制文件将被加载。这使你能够运行算法并查看结果，即使还没有实施任何东西。中期项目的预计算结果需要使用[this](https://drive.google.com/drive/folders/1-s46dKSrtx8rrNwnObGbly2nO3i4D7r7?usp=sharing)链接加载。请先使用`darknet`文件夹。解压其中的文件，并将其内容放入`results`文件夹中。

- `exec_tracking` : 控制物体追踪算法的执行

- `exec_visualization` : 控制结果的可视化
  - `show_range_image` 显示两个LiDAR测距图像通道（范围和强度）
  - `show_labels_in_image` 将地面实况方框投射到前面的摄像机图像中
  - `show_objects_and_labels_in_bev` 将检测到的物体和标签盒投射到鸟瞰图中
  - `show_objects_in_bev_labels_in_camera` 显示一个堆叠视图，上面是摄像机图像内的标签，下面是检测到的物体的鸟瞰图
  - `show_tracks` 显示追踪结果
  - `show_detection_performance` 显示基于所有检测到的性能评估 
  - `make_tracking_movie` 渲染物体追踪结果的输出影片

即使没有解决任何任务，项目代码也可以被执行。

最后的项目使用预先计算的激光雷达探测，以便所有学生都有相同的输入数据。如果你使用工作区，数据已经在那里准备好了。否则，[下载预先计算的激光雷达探测数据](https://drive.google.com/drive/folders/1IkqFGYTF6Fh_d8J3UjQOSNJ2V42UDZpO?usp=sharing) (~1 GB)，解压缩并把它们放在`results`文件夹中。

## 外部依赖

这个项目的部分内容基于以下仓库：

- [Simple Waymo Open Dataset Reader](https://github.com/gdlg/simple-waymo-open-dataset-reader)
- [Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds](https://github.com/maudzung/SFA3D)
- [Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://github.com/maudzung/Complex-YOLOv4-Pytorch)


## License

[License](LICENSE.md)