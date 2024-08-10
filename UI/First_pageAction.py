import sys

import os
import  SimpleITK as sitk
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import pyqtSlot, QTimer, QEventLoop, QObject, pyqtSignal
from PyQt5 import QtCore, QtWidgets, QtCore,QtGui
from PyQt5.QtGui import QTextCursor
# 导入login_window.py、my_main_window.py里面全部内容
import ProcessingData
import First_page
import ProcessingData
import Read_Label
import TransLabelOrder
import UnzipData
import DataRotate
import TrainModel
import predict
import FixHole
import glob
import random
import argparse
from scipy.ndimage import rotate
import re
#import glob
from glob import glob
import numpy as np
import nibabel as nib
import random
import math
import re
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from UNet import UNet
from init import Options
from networks import build_net, update_learning_rate, build_UNETR
# from networks import build_net
import logging
#import os`
import sys
import tempfile
#from glob import glob
import argparse
import nibabel as nib
import numpy as np
#import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import create_test_image_3d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (EnsureType, Compose, LoadImaged, AddChanneld, Transpose, Activations, AsDiscrete,
                              RandGaussianSmoothd, CropForegroundd, SpatialPadd,
                              ScaleIntensityd, ToTensord, RandSpatialCropd, Rand3DElasticd, RandAffined, RandZoomd,
                              EnsureChannelFirstd, EnsureTyped,
                              Spacingd, Orientationd, Resized, ThresholdIntensityd, RandShiftIntensityd, BorderPadd,
                              RandGaussianNoised, RandAdjustContrastd, NormalizeIntensityd, RandFlipd, MapTransform)

from monai.visualize import plot_2d_or_3d_image
from PyQt5.QtWidgets import *
import sys
from show import Ui_MainWindow #导入GUI文件
from MyFigure import *#嵌入了matplotlib的文件
from pathlib import Path
import matplotlib.pyplot as plt
import subprocess


from UNet import UNet
from utils import *
import argparse
from networks import build_net, build_UNETR
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import NiftiSaver, create_test_image_3d, list_data_collate, decollate_batch
from monai.transforms import (EnsureType, Compose, LoadImaged, AddChanneld, Transpose,Activations,AsDiscrete, RandGaussianSmoothd, CropForegroundd, SpatialPadd,
                              ScaleIntensityd, ToTensord, RandSpatialCropd, Rand3DElasticd, RandAffined, RandZoomd,
                              Spacingd, Orientationd, Resized, ThresholdIntensityd, RandShiftIntensityd, BorderPadd, RandGaussianNoised, RandAdjustContrastd,NormalizeIntensityd,RandFlipd)
from scipy import ndimage
import qdarkstyle

i = 0
path_A = ""
path_B = ""


def segment(image, label, result, weights, resolution, patch_size, network, gpu_ids, out_channels):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if label is not None:
        uniform_img_dimensions_internal(image, label, True)
        files = [{"image": image, "label": label}]
    else:
        files = [{"image": image}]

    # original size, size after crop_background, cropped roi coordinates, cropped resampled roi size
    original_shape, crop_shape, coord1, coord2, resampled_size, original_resolution = statistics_crop(image,
                                                                                                      resolution)

    # -------------------------------

    if label is not None:
        if resolution is not None:

            val_transforms = Compose([
                LoadImaged(keys=['image', 'label']),
                AddChanneld(keys=['image', 'label']),
                # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),  # Threshold CT
                # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
                # CropForegroundd(keys=['image', 'label'], source_key='image'),  # crop CropForeground

                NormalizeIntensityd(keys=['image']),  # intensity
                ScaleIntensityd(keys=['image']),
                Spacingd(keys=['image', 'label'], pixdim=resolution, mode=('bilinear', 'nearest')),  # resolution

                SpatialPadd(keys=['image', 'label'], spatial_size=patch_size, method='end'),
                ToTensord(keys=['image', 'label'])])
        else:

            val_transforms = Compose([
                LoadImaged(keys=['image', 'label']),
                AddChanneld(keys=['image', 'label']),
                # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),  # Threshold CT
                # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
                # CropForegroundd(keys=['image', 'label'], source_key='image'),  # crop CropForeground

                NormalizeIntensityd(keys=['image']),  # intensity
                ScaleIntensityd(keys=['image']),

                SpatialPadd(keys=['image', 'label'], spatial_size=patch_size, method='end'),
                # pad if the image is smaller than patch
                ToTensord(keys=['image', 'label'])])

    else:
        if resolution is not None:

            val_transforms = Compose([
                LoadImaged(keys=['image']),
                AddChanneld(keys=['image']),
                # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),  # Threshold CT
                # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
                # CropForegroundd(keys=['image'], source_key='image'),  # crop CropForeground

                NormalizeIntensityd(keys=['image']),  # intensity
                ScaleIntensityd(keys=['image']),
                Spacingd(keys=['image'], pixdim=resolution, mode=('bilinear')),  # resolution

                SpatialPadd(keys=['image'], spatial_size=patch_size, method='end'),
                # pad if the image is smaller than patch
                ToTensord(keys=['image'])])
        else:

            val_transforms = Compose([
                LoadImaged(keys=['image']),
                AddChanneld(keys=['image']),
                # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),  # Threshold CT
                # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
                # CropForegroundd(keys=['image'], source_key='image'),  # crop CropForeground

                NormalizeIntensityd(keys=['image']),  # intensity
                ScaleIntensityd(keys=['image']),

                SpatialPadd(keys=['image'], spatial_size=patch_size, method='end'),
                # pad if the image is smaller than patch
                ToTensord(keys=['image'])])

    val_ds = monai.data.Dataset(data=files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, collate_fn=list_data_collate, pin_memory=False)

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
    # post_trans = Compose([Activations(softmax=True), AsDiscrete(threshold_values=True, n_classes=out_channels)])
    post_trans = Compose([EnsureType(), Activations(softmax=True),
                          AsDiscrete(argmax=True, to_onehot=6, n_classes=out_channels, threshold_values=True)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=6, n_classes=out_channels, threshold_values=True)])

    if gpu_ids != '-1':

        # try to use all the available GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
        device = torch.device("cpu")

    # build the network
    # if network == 'nnunet':
    #     net = build_net()  # nn build_net
    # elif network == 'unetr':
    #     net = build_UNETR() # UneTR
    net = UNet(1, 6)
    net = net.to(device)

    if gpu_ids == '-1':

        net.load_state_dict(new_state_dict_cpu(weights))

    else:

        net.load_state_dict(new_state_dict(weights))

    # define sliding window size and batch size for windows inference
    roi_size = patch_size
    sw_batch_size = 4

    net.eval()
    with torch.no_grad():

        if label is None:
            for val_data in val_loader:
                val_images = val_data["image"].to(device)
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

        else:
            for val_data in val_loader:
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            print("Evaluation Metric (Dice):", metric)
            metric_batch = dice_metric_batch.aggregate()
            metric_label1 = metric_batch[0].item()
            print("Label1 Metric (Dice):", metric_label1)
            metric_label2 = metric_batch[1].item()
            print("Label2 Metric (Dice):", metric_label2)
            metric_label3 = metric_batch[2].item()
            print("Label3 Metric (Dice):", metric_label3)
            metric_label4 = metric_batch[3].item()
            print("Label4 Metric (Dice):", metric_label4)
            metric_label5 = metric_batch[4].item()
            print("Label5 Metric (Dice):", metric_label5)

        result_array = val_outputs[0].squeeze().data.cpu().numpy()

        empty_array = np.zeros(result_array[0].shape)

        for i in range(out_channels):  # MULTI LABEL segmentation part
            channel_i = result_array[i]
            if i == 0:
                channel_i = np.where(channel_i == 1, 0, channel_i)
            elif i > 0:
                channel_i = np.where(channel_i == 1, int(i), channel_i)
            empty_array = empty_array + channel_i
        result_array = empty_array
        # 多类分割↑

        # Remove the pad if the image was smaller than the patch in some directions
        result_array = result_array[0:resampled_size[0], 0:resampled_size[1], 0:resampled_size[2]]

        # resample back to the original resolution
        if resolution is not None:

            result_array_np = np.transpose(result_array, (2, 1, 0))
            result_array_temp = sitk.GetImageFromArray(result_array_np)
            result_array_temp.SetSpacing(resolution)

            # save temporary label
            writer = sitk.ImageFileWriter()
            writer.SetFileName('temp_seg.nii')
            writer.Execute(result_array_temp)

            files = [{"image": 'temp_seg.nii'}]

            files_transforms = Compose([
                LoadImaged(keys=['image']),
                AddChanneld(keys=['image']),
                Spacingd(keys=['image'], pixdim=original_resolution, mode=('nearest')),
                Resized(keys=['image'], spatial_size=crop_shape, mode=('nearest')),
            ])

            files_ds = Dataset(data=files, transform=files_transforms)
            files_loader = DataLoader(files_ds, batch_size=1, num_workers=0)

            for files_data in files_loader:
                files_images = files_data["image"]

                res = files_images.squeeze().data.numpy()

            result_array = np.rint(res)

            os.remove('./temp_seg.nii')

        # recover the cropped background before saving the image
        # empty_array = np.zeros(original_shape)
        # empty_array[coord1[0]:coord2[0],coord1[1]:coord2[1],coord1[2]:coord2[2]] = result_array

        result_seg = from_numpy_to_itk(result_array, image)

        # save label
        writer = sitk.ImageFileWriter()
        writer.SetFileName(result)
        writer.Execute(result_seg)
        # print("Saved Result at:", str(result))


class GPUWidget(QWidget):
    def __init__(self, parent=None):
        super(GPUWidget, self).__init__(parent)

        # 创建用于存储GPU历史数据的列表
        self.gpu_history = []
        self.gpu_limit = 10000  # 仅保留最近100个数据点
        self.gpu_usage = 0

        # 创建一个Matplotlib图表
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim([0, 10])
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('GPU Usage (%)')
        self.canvas = FigureCanvas(self.fig)

        # 将Matplotlib图表添加到PyQt QWidget中的垂直布局中
        vbox_layout = QVBoxLayout()
        vbox_layout.addWidget(self.canvas)
        self.setLayout(vbox_layout)

        #创建Qtimer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gpu_usage)
        self.timer.start(1000)  # 刷新频率为每秒一次

    def update_gpu_usage(self):
        #获取GPU使用率并更新GPU历史数据
        gpu_usage = int(subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']))
        self.gpu_history.append(gpu_usage)
        self.gpu_history = self.gpu_history[-self.gpu_limit:]  # 仅保留最近100个数据点
        self.gpu_usage = gpu_usage

        # 清楚并绘制最新的GPU历史数据
        self.ax.clear()
        self.ax.set_ylim([0, 100])
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('GPU Usage (%)')
        self.ax.plot(list(range(len(self.gpu_history))), self.gpu_history)
        self.canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.grid_layout = QGridLayout()
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.grid_layout)
        self.setCentralWidget(self.central_widget)

        gpu_widget = GPUWidget(self)
        self.grid_layout.addWidget(gpu_widget, 0, 0)

class MainDialogImgBW(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainDialogImgBW, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("三维心脏CTA影像分割系统")
        self.setMinimumSize(0, 0)
        self.showMaximized()
        self.textEdit.append("欢迎使用三维心脏CTA影像分割系统！\n");
        self.gridlayout_4 = QGridLayout(self.groupBox_4)
        gpu_widget = GPUWidget(self)
        self.gridlayout_4.addWidget(gpu_widget, 0, 0)
        # 创建存放nii文件路径的属性
        self.nii_path = ''
        # 创建记录nii文件里面图片数量的属性


        self.mask_path = ''
        self.shape = 1

        #创建用于检查radio button选择标记的属性，选择'nii图像'，为0，现在‘mask图像’，为1
        self.check = 0
        # 定义MyFigure类的一个实例
        self.F1 = MyFigure(width=5, height=5, dpi=100)
        self.F2 = MyFigure(width=5, height=5, dpi=100)
        self.F3 = MyFigure(width=5, height=5, dpi=100)
        # 在GUI的groupBox中创建一个布局，用于添加MyFigure类的实例（即图形）后其他部件。
        self.gridlayout = QGridLayout(self.groupBox)  # 继承容器groupBox
        self.gridlayout_2 = QGridLayout(self.groupBox_2)
        self.gridlayout_3 = QGridLayout(self.groupBox_3)
        self.gridlayout.addWidget(self.F1, 0, 1)
        self.gridlayout_2.addWidget(self.F2, 0, 1)
        self.gridlayout_3.addWidget(self.F3, 0, 1)
        self.pushButton.clicked.connect(self.bindButton)

        self.pushButton_2.clicked.connect(self.bindButton2)
        self.horizontalSlider.valueChanged.connect(self.bindSlider)
        self.horizontalSlider_2.valueChanged.connect(self.bindSlider_2)
        self.horizontalSlider_3.valueChanged.connect(self.bindSlider_3)
        self.radioButton.clicked.connect(self.bindradiobutton)
        self.radioButton_2.clicked.connect(self.bindradiobutton)



    def showimage(self, slice_idx):
        data_nii = nib.load(Path(self.nii_path))
        data1 = data_nii.get_fdata()
        self.shape = data1.shape[-1]
        self.horizontalSlider.setRange(1, data1.shape[-1])
        self.horizontalSlider_2.setRange(1, data1.shape[-1])
        self.horizontalSlider_3.setRange(1, data1.shape[-1])

        if not self.mask_path=='':
            data_mask = nib.load(Path(self.mask_path))
            data2 = data_mask.get_fdata()

        fig1 = self.F1.figure
        fig2 = self.F2.figure
        fig3 = self.F3.figure
        fig1.clear()
        fig2.clear()
        fig3.clear()

        ax1 = fig1.add_subplot(111)
        ax1.axhline(y=data1.shape[0]/2, color='w', linestyle='--')
        ax1.axvline(x=data1.shape[0]/2, color='w', linestyle='--')
        ax2 = fig2.add_subplot(111)
        ax2.axhline(y=data1.shape[0]/2, color='w', linestyle='--')
        ax2.axvline(x=data1.shape[0]/2, color='w', linestyle='--')
        ax3 = fig3.add_subplot(111)  # 将画布划成1*1的大小并将图像放在1号位置，给画布加上一个坐标轴
        ax3.axhline(y=data1.shape[0]/2, color='w', linestyle='--')
        ax3.axvline(x=data1.shape[0]/2, color='w', linestyle='--')

        ax1.imshow(data1[:, :, slice_idx - 1], cmap='gray')

        if self.check == 1:
            array1 = list(data2[:, :, slice_idx - 1])
            a = len(array1)
            b = len(array1[0])
            pic = [[0] * b for i in range(a)]
            for i in range(0, a):
                for j in range(0, b):
                    if array1[i][j] == 0:
                        pic[i][j] = [0, 0, 0, 0]
                    elif array1[i][j] == -12.0:
                        pic[i][j] = [255, 0, 0, 100]
                    elif array1[i][j] == -51.0:
                        pic[i][j] = [0, 255, 0, 100]
                    elif array1[i][j] == 38.0:
                        pic[i][j] = [0, 0, 255, 100]
                    elif array1[i][j] == -92.0:
                        pic[i][j] = [238, 255, 83, 100]
                    elif array1[i][j] == 88.0:
                        pic[i][j] = [255, 144, 255, 100]

            ax1.imshow(pic, cmap='viridis')
            del array1
            del pic


        ax2.imshow(data1[:, slice_idx - 1, :], cmap='gray')
        if self.check == 1:
            array1 = list(data2[:, slice_idx - 1, :])
            a = len(array1)
            b = len(array1[0])
            pic = [[0] * b for i in range(a)]
            for i in range(0, a):
                for j in range(0, b):
                    if array1[i][j] == 0:
                        pic[i][j] = [0, 0, 0, 0]
                    elif array1[i][j] == -12.0:
                        pic[i][j] = [255, 0, 0, 100]
                    elif array1[i][j] == -51.0:
                        pic[i][j] = [0, 255, 0, 100]
                    elif array1[i][j] == 38.0:
                        pic[i][j] = [0, 0, 255, 100]
                    elif array1[i][j] == -92.0:
                        pic[i][j] = [238, 255, 83, 100]
                    elif array1[i][j] == 88.0:
                        pic[i][j] = [255, 144, 255, 100]

            ax2.imshow(pic, cmap='viridis')
            del array1
            del pic
        # Display sagittal view on third subplot
        ax3.imshow(data1[slice_idx - 1, :, :], cmap='gray')
        if self.check == 1:
            array1 = list(data2[slice_idx - 1, :, :])
            a = len(array1)
            b = len(array1[0])
            pic = [[0] * b for i in range(a)]
            for i in range(0, a):
                for j in range(0, b):
                    if array1[i][j] == 0:
                        pic[i][j] = [0, 0, 0, 0]
                    elif array1[i][j] == -12.0:
                        pic[i][j] = [255, 0, 0, 100]
                    elif array1[i][j] == -51.0:
                        pic[i][j] = [0, 255, 0, 100]
                    elif array1[i][j] == 38.0:
                        pic[i][j] = [0, 0, 255, 100]
                    elif array1[i][j] == -92.0:
                        pic[i][j] = [238, 255, 83, 100]
                    elif array1[i][j] == 88.0:
                        pic[i][j] = [255, 144, 255, 100]

            ax3.imshow(pic, cmap='viridis')

        fig1.canvas.draw()
        fig2.canvas.draw()
        fig3.canvas.draw()

    def bindradiobutton(self):
        if self.radioButton.isChecked():
            self.check = 0
        else:
            self.check = 1
        slice_idx = self.horizontalSlider.value()
        self.showimage(slice_idx)


    def bindSlider(self):
        slice_idx = self.horizontalSlider.value()
        self.showimage(slice_idx)
    def bindSlider_2(self):
        slice_idx = self.horizontalSlider_2.value()
        self.showimage(slice_idx)
    def bindSlider_3(self):
        slice_idx = self.horizontalSlider_3.value()
        self.showimage(slice_idx)
    def bindButton(self):
        file_name = QFileDialog.getOpenFileName(None, "Open File", "./", "nii(*.nii.gz;*.nii)")
        if file_name[0] == '':
            #QMessageBox.Warning( QMainWindow,'提示','请选择文件！')
            return
        self.nii_path = file_name[0]
        slice_idx = self.horizontalSlider.value()
        self.showimage(slice_idx)

    def bindButton2(self):
        file_name = QFileDialog.getOpenFileName(None, "Open File", "./", "nii(*.nii.gz;*.nii)")
        self.mask_path = file_name[0]


class Options():
    """This class defines options used during both training and test time."""

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):

        # basic parameters
        parser.add_argument('--images_folder', type=str, default= path_A)
        parser.add_argument('--labels_folder', type=str, default= path_B)
        parser.add_argument('--increase_factor_data', default=0, help='Increase data number per epoch')
        parser.add_argument('--preload', type=str, default=None)
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')

        # dataset parameters
        parser.add_argument('--network', default='nnunet', help='nnunet, unetr')
        parser.add_argument('--patch_size', default=(128, 128, 128),
                            help='Size of the patches extracted from the image')
        parser.add_argument('--spacing', default=None, help='Original Resolution')
        parser.add_argument('--resolution', default=None,
                            help='New Resolution, if you want to resample the data in training. I suggest to resample in organize_folder_structure.py, otherwise in train resampling is slower')
        parser.add_argument('--batch_size', type=int, default=1, help='batch size, depends on your machine')  # 4
        parser.add_argument('--in_channels', default=1, type=int, help='Channels of the input')
        parser.add_argument('--out_channels', default=6, type=int, help='Channels of the output')

        # training parameters
        parser.add_argument('--epochs', default=400, help='Number of epochs')
        parser.add_argument('--lr', default=0.0001, help='Learning rate')
        parser.add_argument('--benchmark', default=True)

        # Inference
        # This is just a trick to make the predict script working, do not touch it now for the training.
        parser.add_argument('--result', default=None, help='Keep this empty and go to predict_single_image script')
        parser.add_argument('--weights', default=None, help='Keep this empty and go to predict_single_image script')

        self.initialized = True
        return parser

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt = parser.parse_args()
        # set gpu ids
        if opt.gpu_ids != '-1':
            os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        return opt


def join_channels(result_array, channels):
    result_array = result_array[0].squeeze().data.cpu().numpy()
    empty_array = np.zeros(result_array[0].shape)
    for i in range(channels):
        channel_i = result_array[i]
        if i == 0:
            channel_i = np.where(channel_i == 1, 0, channel_i)
        elif i > 0:
            channel_i = np.where(channel_i == 1, int(i), channel_i)
        empty_array = empty_array + channel_i
    result_array = empty_array
    result_array = result_array[np.newaxis,np.newaxis, :, :, :]
    return torch.from_numpy(result_array)



class First_page_window(First_page.Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(First_page_window, self).__init__()
        self.setupUi(self)
        #self.pushButton.clicked.connect(self.close)


class Process_window(ProcessingData.Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Process_window, self).__init__()
        self.setupUi(self)


class ReadLabel_window(Read_Label.Ui_MainWindow, QMainWindow):

    def __init__(self):
        super(ReadLabel_window, self).__init__()
        self.setupUi(self)


        #self.pushButton.clicked.connect(self.close)



    def msg(self, Filepath):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", "")  # 起始路径
        #self.fileT.setText(directory)
        self.lineEdit.setText(directory)

    def Read(self):
        j=0
        print(self.lineEdit.text())

        num_files = len(glob(self.lineEdit.text() + '/*'))
        print('222')
        while j < num_files:
            print(j)
            label_path = self.lineEdit.text()+r"/label" + str(j) + ".nii"
            mask = sitk.ReadImage(label_path, sitk.sitkInt8)
            label_array = sitk.GetArrayFromImage(mask)

            #print(f'label_path: {label_path}')
            #self.Read_singal.emit(f'label_path: {label_path}')  # 发射信号
            self.Read_textEdit.append(f'label_path: {label_path}'+"\n")
            QtWidgets.QApplication.processEvents()
            # 查看label里面有几种值
            #print(f'标签中有几种值: {np.unique(label_array)}')
            #self.Read_textEdit.append(str(np.unique(label_array)))
            #self.Read_singal.emit(f'标签中有几种值: {np.unique(label_array)}')  # 发射信号
            self.Read_textEdit.append(f'标签中有几种值: {np.unique(label_array)}' + "\n")
            QtWidgets.QApplication.processEvents()
            # 查看每个标签对应多少像素
            #print(f'每个标签像素数量：', np.unique(label_array, return_counts=True))
            #self.Read_textEdit.append(str('每个标签像素数量：'+np.unique(label_array, return_counts=True)))%%%%%%
            QtWidgets.QApplication.processEvents()
            #self.Read_singal.emit(f'每个标签像素数量：', np.unique(label_array, return_counts=True))  # 发射信号
            j = j + 1



class TransLabelOrder_window(TransLabelOrder.Ui_MainWindow, QMainWindow):

    def __init__(self):
        super(TransLabelOrder_window, self).__init__()
        self.setupUi(self)



        #self.pushButton.clicked.connect(self.close)
    def msg(self, Filepath):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", "")  # 起始路径
        #self.fileT.setText(directory)
        self.lineEdit.setText(directory)



    def Trans(self):
        num_files = len(glob(self.lineEdit.text() + '/*'))
        path = self.lineEdit.text()
        path_label = path
        path_ct_cut = path + '/ct-change/'
        path_label_cut = path + '/label-change/'
        if not os.path.exists(path_ct_cut):
            os.makedirs(path_ct_cut)
        if not os.path.exists(path_label_cut):
            os.makedirs(path_label_cut)

        for root, dirs, files in os.walk(path_label):
            for file in files:
                #path_ct_file = os.path.join(root, file)
                path_ct_file = root+"/"+file
                #print("现对 " + file + " 文件裁剪：")
                self.Trans_textEdit.append("现对 " + file + " 文件裁剪：")
                QtWidgets.QApplication.processEvents()
                path_label_file = path_label + "/"+file
                mask = sitk.ReadImage(path_label_file, sitk.sitkInt8)
                mask_array = sitk.GetArrayFromImage(mask)
                mask_array = np.where(mask_array == -92, 1, mask_array)
                mask_array = np.where(mask_array == -51, 2, mask_array)
                mask_array = np.where(mask_array == -12, 3, mask_array)
                mask_array = np.where(mask_array == 38, 4, mask_array)
                mask_array = np.where(mask_array == 88, 5, mask_array)
                path_label_cut_file = path_label_cut + file
                new = sitk.GetImageFromArray(mask_array)
                sitk.WriteImage(new, path_label_cut_file)

                #print(file + '--转换完成！')
                self.Trans_textEdit.append(file + '--转换完成！')
                QtWidgets.QApplication.processEvents()



class UnzipData_window(UnzipData.Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(UnzipData_window, self).__init__()
        self.setupUi(self)
        #self.pushButton.clicked.connect(self.close)

class DataRotate_window(DataRotate.Ui_MainWindow, QMainWindow):

    def __init__(self):
        super(DataRotate_window, self).__init__()
        self.setupUi(self)
        self.lineEdit_3.setPlaceholderText('默认角度为30')


        #self.pushButton.clicked.connect(self.close)




    def msg1(self, Filepath):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", "")  # 起始路径
        #self.fileT.setText(directory)
        self.lineEdit.setText(directory)

    def msg2(self, Filepath):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", "")  # 起始路径
        #self.fileT.setText(directory)
        self.lineEdit_2.setText(directory)

    def rotate_nii(nii_path, degree, currentIndex, oldIndex):
            # 加载nii文件
            nii = nib.load(nii_path)
            img = nii.get_fdata()
            affine = nii.affine

            # 在三个轴上进行旋转
            rotate_axis = random.choice([0, 1, 2])
            if rotate_axis == 0:
                img = np.rot90(img, k=degree // 90, axes=(0, 1))
                affine = np.dot(nib.affines.from_matvec(np.diag([1, -1, -1]), np.zeros(3)), affine)
            elif rotate_axis == 1:
                img = np.rot90(img, k=degree // 90, axes=(0, 2))
                affine = np.dot(nib.affines.from_matvec(np.diag([-1, 1, -1]), np.zeros(3)), affine)
            else:
                img = np.rot90(img, k=degree // 90, axes=(1, 2))
                affine = np.dot(nib.affines.from_matvec(np.diag([-1, -1, 1]), np.zeros(3)), affine)

            # 保存旋转后的nii文件
            rotated_nii = nib.Nifti1Image(img, affine)
            # nib.save(rotated_nii, nii_path[:-6] + '56.nii')
            if oldIndex < 10:
                nib.save(rotated_nii, nii_path[:-5] + str(currentIndex) + '.nii')
            elif oldIndex < 100:
                nib.save(rotated_nii, nii_path[:-6] + str(currentIndex) + '.nii')

    def get_numeric_suffix(filename):
        # 从文件名中获取最后的数字后缀
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        else:
            return 0
    def Rotate(self):
        currentImage = ""
        currentLabel = ""
        TempImageStr = ""
        TempLabelStr = ""
        Imagefolder_path = self.lineEdit.text()
        Labelfolder_path = self.lineEdit_2.text()
        Image_nii_files = glob(os.path.join(Imagefolder_path, "*.nii"))
        Label_nii_files = glob(os.path.join(Labelfolder_path, "*.nii"))

        # rotate_nii("C:/Users/12234/Desktop/mr_data/imagesCorrect/image18.nii",30.0)
        i=0
        while i < min(len(Image_nii_files), len(Label_nii_files)):
            self.Rotate_textEdit.append('正在处理'+Image_nii_files[i])
            QtWidgets.QApplication.processEvents()
            self.Rotate_textEdit.append('正在处理'+Label_nii_files[i])
            QtWidgets.QApplication.processEvents()
            #print('正在处理'+Image_nii_files[i])
            #print('正在处理'+Label_nii_files[i])
            TempImageStr = Image_nii_files[i][-12:]
            TempLabelStr = Label_nii_files[i][-12:]
            currentImage = DataRotate_window.get_numeric_suffix(str(TempImageStr))
            currentImage = currentImage + len(Image_nii_files)
            currentLabel = DataRotate_window.get_numeric_suffix(str(TempLabelStr))
            currentLabel = currentLabel + len(Label_nii_files)
            angle = ""
            angle = self.lineEdit_3.text()
            if angle == "":
                angle = 30

            DataRotate_window.rotate_nii(Image_nii_files[i], angle, currentImage, DataRotate_window.get_numeric_suffix(str(TempImageStr)))
            DataRotate_window.rotate_nii(Label_nii_files[i], angle, currentLabel, DataRotate_window.get_numeric_suffix(str(TempLabelStr)))

            i = i + 1
        self.Rotate_textEdit.append("处理完成！")
        QtWidgets.QApplication.processEvents()
        #print("处理完成！")

class TrainModel_window(TrainModel.Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(TrainModel_window, self).__init__()
        self.setupUi(self)
        #self.pushButton.clicked.connect(self.close)

    def msg1(self, Filepath):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", "")  # 起始路径
        #self.fileT.setText(directory)
        self.lineEdit.setText(directory)

    def msg2(self, Filepath):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", "")  # 起始路径
        #self.fileT.setText(directory)
        self.lineEdit_2.setText(directory)

    def Train(self):
        global path_A
        global path_B
        path_A = self.lineEdit.text()
        path_B = self.lineEdit_2.text()
        opt = Options().parse()
        # monai.config.print_config()
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        # check gpus
        if opt.gpu_ids != '-1':
            num_gpus = len(opt.gpu_ids.split(','))
        else:
            num_gpus = 0
        #print('number of GPU:', num_gpus)
        self.textEdit.append('number of GPU:'+ str(num_gpus)+'\n')
        QtWidgets.QApplication.processEvents()
        # Data loader creation
        # train images
        train_images = sorted(glob(os.path.join(opt.images_folder, 'train', '*.nii')))
        train_segs = sorted(glob(os.path.join(opt.labels_folder, 'train', '*.nii')))

        train_images_for_dice = sorted(glob(os.path.join(opt.images_folder, 'train', '*.nii')))
        train_segs_for_dice = sorted(glob(os.path.join(opt.labels_folder, 'train', '*.nii')))

        # validation images
        val_images = sorted(glob(os.path.join(opt.images_folder, 'val', '*.nii')))
        val_segs = sorted(glob(os.path.join(opt.labels_folder, 'val', '*.nii')))

        # test images
        # test_images = sorted(glob(os.path.join(opt.images_folder, 'test', 'image*.nii')))
        # test_segs = sorted(glob(os.path.join(opt.labels_folder, 'test', 'label*.nii')))

        # augment the data list for training
        for i in range(int(opt.increase_factor_data)):
            train_images.extend(train_images)
            train_segs.extend(train_segs)

        #print('Number of training patches per epoch:', len(train_images))
        self.textEdit.append('Number of training patches per epoch:'+ str(train_images)+'\n')
        QtWidgets.QApplication.processEvents()
        #print('Number of training images per epoch:', len(train_images_for_dice))
        self.textEdit.append('Number of training images per epoch:' + str(len(train_images_for_dice)) + '\n')
        QtWidgets.QApplication.processEvents()
        #print('Number of validation images per epoch:', len(val_images))
        self.textEdit.append('Number of validation images per epoch:' + str(len(val_images)) + '\n')
        QtWidgets.QApplication.processEvents()
        # print('Number of test images per epoch:', len(test_images))

        # Creation of data directories for data_loader

        train_dicts = [{'image': image_name, 'label': label_name}
                       for image_name, label_name in zip(train_images, train_segs)]

        train_dice_dicts = [{'image': image_name, 'label': label_name}
                            for image_name, label_name in zip(train_images_for_dice, train_segs_for_dice)]

        val_dicts = [{'image': image_name, 'label': label_name}
                     for image_name, label_name in zip(val_images, val_segs)]

        # test_dicts = [{'image': image_name, 'label': label_name}
        #              for image_name, label_name in zip(test_images, test_segs)]

        # Transforms list
        # Need to concatenate multiple channels here if you want multichannel segmentation
        # Check other examples on Monai webpage.

        if opt.resolution is not None:
            train_transforms = [
                LoadImaged(keys=['image', 'label']),
                # EnsureChannelFirstd(keys="image"),
                # EnsureTyped(keys=["image", "label"]),
                # ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                AddChanneld(keys=['image', 'label']),
                # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),  # CT HU filter
                # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
                # CropForegroundd(keys=['image', 'label'], source_key='image'),               # crop CropForeground

                NormalizeIntensityd(keys=['image']),  # augmentation
                ScaleIntensityd(keys=['image']),  # intensity
                Spacingd(keys=['image', 'label'], pixdim=opt.resolution, mode=('bilinear', 'nearest')),  # resolution

                RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=1),
                RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=0),
                RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=2),
                # RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                #             rotate_range=(np.pi / 36, np.pi / 36, np.pi * 2), padding_mode="zeros"),
                # RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                #             rotate_range=(np.pi / 36, np.pi / 2, np.pi / 36), padding_mode="zeros"),
                # RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                #             rotate_range=(np.pi / 2, np.pi / 36, np.pi / 36), padding_mode="zeros"),
                # Rand3DElasticd(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                #                sigma_range=(5, 8), magnitude_range=(100, 200), scale_range=(0.15, 0.15, 0.15),
                #                padding_mode="zeros"),
                # RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15), prob=0.1,),
                # RandAdjustContrastd(keys=['image'], gamma=(0.5, 2.5), prob=0.1),
                # RandGaussianNoised(keys=['image'], prob=0.1, mean=np.random.uniform(0, 0.5), std=np.random.uniform(0, 15)),
                # RandShiftIntensityd(keys=['image'], offsets=np.random.uniform(0,0.3), prob=0.1),

                SpatialPadd(keys=['image', 'label'], spatial_size=opt.patch_size, method='end'),
                # pad if the image is smaller than patch
                # RandSpatialCropd(keys=['image', 'label'], roi_size=opt.patch_size, random_size=False),
                ToTensord(keys=['image', 'label'])
            ]

            val_transforms = [
                LoadImaged(keys=['image', 'label']),
                # EnsureChannelFirstd(keys="image"),
                # EnsureTyped(keys=["image", "label"]),
                # ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                AddChanneld(keys=['image', 'label']),
                # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),
                # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
                # CropForegroundd(keys=['image', 'label'], source_key='image'),                   # crop CropForeground

                NormalizeIntensityd(keys=['image']),  # 强度变换对图像值强度进行变换的，像CT和MRI的值都是从-1000—+3000多的不等，通常需要进行归一化
                ScaleIntensityd(keys=['image']),
                Spacingd(keys=['image', 'label'], pixdim=opt.resolution, mode=('bilinear', 'nearest')),  # resolution

                SpatialPadd(keys=['image', 'label'], spatial_size=opt.patch_size, method='end'),
                # pad if the image is smaller than patch
                ToTensord(keys=['image', 'label'])
            ]
        else:
            train_transforms = [
                LoadImaged(keys=['image', 'label']),
                # EnsureChannelFirstd(keys="image"),
                # EnsureTyped(keys=["image", "label"]),
                # ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                AddChanneld(keys=['image', 'label']),
                # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),
                # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
                # CropForegroundd(keys=['image', 'label'], source_key='image'),               # crop CropForeground

                NormalizeIntensityd(keys=['image']),  # augmentation
                ScaleIntensityd(keys=['image']),  # intensity

                # RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=1),
                # RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=0),
                # RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=2),
                # RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                #             rotate_range=(np.pi / 36, np.pi / 36, np.pi * 2), padding_mode="zeros"),
                # RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                #             rotate_range=(np.pi / 36, np.pi / 2, np.pi / 36), padding_mode="zeros"),
                # RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                #             rotate_range=(np.pi / 2, np.pi / 36, np.pi / 36), padding_mode="zeros"),
                # Rand3DElasticd(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                #                sigma_range=(5, 8), magnitude_range=(100, 200), scale_range=(0.15, 0.15, 0.15),
                #                padding_mode="zeros"),
                # RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15), prob=0.1,),
                # RandAdjustContrastd(keys=['image'], gamma=(0.5, 2.5), prob=0.1),
                # RandGaussianNoised(keys=['image'], prob=0.1, mean=np.random.uniform(0, 0.5), std=np.random.uniform(0, 1)),
                # RandShiftIntensityd(keys=['image'], offsets=np.random.uniform(0,0.3), prob=0.1),

                SpatialPadd(keys=['image', 'label'], spatial_size=opt.patch_size, method='end'),
                # pad if the image is smaller than patch
                RandSpatialCropd(keys=['image', 'label'], roi_size=opt.patch_size, random_size=False),
                ToTensord(keys=['image', 'label'])
            ]

            val_transforms = [
                LoadImaged(keys=['image', 'label']),
                # EnsureChannelFirstd(keys="image"),
                # EnsureTyped(keys=["image", "label"]),
                # ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                AddChanneld(keys=['image', 'label']),
                # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),
                # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
                # CropForegroundd(keys=['image', 'label'], source_key='image'),                   # crop CropForeground

                NormalizeIntensityd(keys=['image']),  # intensity
                ScaleIntensityd(keys=['image']),

                SpatialPadd(keys=['image', 'label'], spatial_size=opt.patch_size, method='end'),
                # pad if the image is smaller than patch
                ToTensord(keys=['image', 'label'])
            ]

        train_transforms = Compose(train_transforms)
        val_transforms = Compose(val_transforms)

        # create a training data loader
        check_train = monai.data.Dataset(data=train_dicts, transform=train_transforms)
        train_loader = DataLoader(check_train, batch_size=opt.batch_size, shuffle=True, collate_fn=list_data_collate,
                                  num_workers=opt.workers, pin_memory=False)

        # create a training_dice data loader
        check_val = monai.data.Dataset(data=train_dice_dicts, transform=val_transforms)
        train_dice_loader = DataLoader(check_val, batch_size=1, num_workers=opt.workers, collate_fn=list_data_collate,
                                       pin_memory=False)

        # create a validation data loader
        check_val = monai.data.Dataset(data=val_dicts, transform=val_transforms)
        val_loader = DataLoader(check_val, batch_size=1, num_workers=opt.workers, collate_fn=list_data_collate,
                                pin_memory=False)

        # create a validation data loader
        # check_val = monai.data.Dataset(data=test_dicts, transform=val_transforms)
        # test_loader = DataLoader(check_val, batch_size=1, num_workers=opt.workers, collate_fn=list_data_collate, pin_memory=False)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # build the network
        # if opt.network == 'nnunet':
        #     net = build_net()  # nn build_net
        # elif opt.network == 'unetr':
        #     net = build_UNETR() # UneTR
        net = UNet(1, 6)
        net.cuda()

        if num_gpus > 1:
            net = torch.nn.DataParallel(net)
        if opt.preload is not None:
            net.load_state_dict(torch.load(opt.preload))

        # loss_function = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        loss_function = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)

        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)

        post_trans = Compose([EnsureType(), Activations(softmax=True),
                              AsDiscrete(argmax=True, to_onehot=6, n_classes=opt.out_channels, threshold_values=True)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=6, n_classes=opt.out_channels, threshold_values=True)])

        # enable cuDNN benchmark
        torch.backends.cudnn.benchmark = opt.benchmark

        optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        # optim = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.99, weight_decay=3e-5, nesterov=True, )
        # net_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda epoch: (1 - epoch / opt.epochs) ** 0.9)

        # start a typical PyTorch training
        val_interval = 1
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = list()
        writer = SummaryWriter()
        metric_train_values = list()
        metric_values = list()
        metric_label1_values = list()
        metric_label2_values = list()
        metric_label3_values = list()
        metric_label4_values = list()
        metric_label5_values = list()

        for epoch in range(opt.epochs):
            #print("-" * 10)
            self.textEdit.append("-" * 10 + '\n')
            QtWidgets.QApplication.processEvents()
            #print(f"epoch {epoch + 1}/{opt.epochs}")
            self.textEdit.append(f"epoch {epoch + 1}/{opt.epochs}"+ '\n')
            QtWidgets.QApplication.processEvents()
            net.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data["image"].cuda(), batch_data["label"].cuda()
                # inputs, labels = (
                #     batch_data["image"].cuda(),
                #     batch_data["label"].cuda(),
                # )
                optim.zero_grad()
                outputs = net(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
                epoch_len = len(check_train) // train_loader.batch_size
                #print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                self.textEdit.append(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}"+ '\n')
                QtWidgets.QApplication.processEvents()
                # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            #print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            self.textEdit.append(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}"+ '\n')
            QtWidgets.QApplication.processEvents()
            # update_learning_rate(net_scheduler, optim)

            if (epoch + 1) % val_interval == 0:
                net.eval()
                with torch.no_grad():

                    def plot_dice(images_loader):

                        val_images = None
                        val_labels = None
                        val_outputs = None
                        for data in images_loader:
                            val_images, val_labels = data["image"].cuda(), data["label"].cuda()
                            # val_images, val_labels = (
                            #     data["image"].cuda(),
                            #     data["label"].cuda(),
                            # )
                            roi_size = opt.patch_size
                            sw_batch_size = 4
                            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
                            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                            dice_metric(y_pred=val_outputs, y=val_labels)
                            dice_metric_batch(y_pred=val_outputs, y=val_labels)

                        metric = dice_metric.aggregate().item()
                        # print("metric: ",metric)
                        metric_batch = dice_metric_batch.aggregate()
                        # print("metric_batch: ",metric_batch)

                        metric_label1 = metric_batch[0].item()
                        metric_label2 = metric_batch[1].item()
                        metric_label3 = metric_batch[2].item()
                        metric_label4 = metric_batch[3].item()
                        metric_label5 = metric_batch[4].item()

                        dice_metric.reset()
                        dice_metric_batch.reset()

                        return metric, metric_label1, metric_label2, metric_label3, metric_label4, metric_label5, val_images, val_labels, val_outputs

                    metric, metric_label1, metric_label2, metric_label3, metric_label4, metric_label5, val_images, val_labels, val_outputs = plot_dice(
                        val_loader)

                    metric_train, metric_train_label1, metric_train_label2, metric_train_label3, metric_train_label4, metric_train_label5, train_images, train_labels, train_outputs = plot_dice(
                        train_dice_loader)

                    # Save best model
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        if epoch >= 30:
                            torch.save(net.state_dict(), f"./model/epoch__{epoch}__{metric}_model.pth")
                            #print("saved new best metric model")
                            self.textEdit.append("saved new best metric model" + '\n')
                            QtWidgets.QApplication.processEvents()
                    if epoch + 1 == 150:
                        torch.save(net.state_dict(), f"./model/epoch__{epoch}__{metric}_the_last_model.pth")

                    # metric_test, metric_test_label1, metric_test_label2, metric_test_label3, metric_test_label4, metric_test_label5, test_images, test_labels, test_outputs = plot_dice(test_loader)
                    # metric_train, train_images, train_labels, train_outputs = plot_dice(train_dice_loader)
                    # metric_test, test_images, test_labels, test_outputs = plot_dice(test_loader)

                    # Logger bar
                    # print(
                    #     "current epoch: {} Training dice: {:.4f} Validation dice: {:.4f} Val_label1 dice: {:.4f} Val_label2 dice: {:.4f} Val_label3 dice: {:.4f} Val_label4 dice: {:.4f} Val_label5 dice: {:.4f} Best Validation dice: {:.4f} at epoch {}".format(
                    #         epoch + 1, metric_train, metric, metric_label1, metric_label2, metric_label3, metric_label4, metric_label5, best_metric, best_metric_epoch
                    #     )
                    # )
                    #print(
                    #    "current epoch: {} \n "
                    #    "Validation dice: {:.4f} Val_label1 dice: {:.4f} Val_label2 dice: {:.4f} Val_label3 dice: {:.4f} Val_label4 dice: {:.4f} Val_label5 dice: {:.4f} \n "
                    #    "Training dice: {:.4f} train_label1 dice: {:.4f} train_label2 dice: {:.4f} train_label3 dice: {:.4f} train_label4 dice: {:.4f} train_label5 dice: {:.4f} \n "
                    #    "Best Validation dice: {:.4f} at epoch {}".format(
                    #        epoch + 1,
                    #        metric, metric_label1, metric_label2, metric_label3, metric_label4, metric_label5,
                    #        metric_train, metric_train_label1, metric_train_label2, metric_train_label3,
                    #        metric_train_label4, metric_train_label5,
                    #        best_metric, best_metric_epoch
                    #    )
                    #)
                    self.textEdit.append("current epoch: {} \n "
                        "Validation dice: {:.4f} Val_label1 dice: {:.4f} Val_label2 dice: {:.4f} Val_label3 dice: {:.4f} Val_label4 dice: {:.4f} Val_label5 dice: {:.4f} \n "
                        "Training dice: {:.4f} train_label1 dice: {:.4f} train_label2 dice: {:.4f} train_label3 dice: {:.4f} train_label4 dice: {:.4f} train_label5 dice: {:.4f} \n "
                        "Best Validation dice: {:.4f} at epoch {}".format(
                            epoch + 1,
                            metric, metric_label1, metric_label2, metric_label3, metric_label4, metric_label5,
                            metric_train, metric_train_label1, metric_train_label2, metric_train_label3,
                            metric_train_label4, metric_train_label5,
                            best_metric, best_metric_epoch
                        ) + '\n')
                    QtWidgets.QApplication.processEvents()
                    writer.add_scalar("Mean_epoch_loss", epoch_loss, epoch + 1)
                    # writer.add_scalar("Testing_dice", metric_test, epoch + 1)
                    # writer.add_scalar("Testing_label1_dice", metric_test_label1, epoch + 1)
                    # writer.add_scalar("Testing_label2_dice", metric_test_label2, epoch + 1)
                    # writer.add_scalar("Testing_label3_dice", metric_test_label3, epoch + 1)
                    # writer.add_scalar("Testing_label4_dice", metric_test_label4, epoch + 1)
                    # writer.add_scalar("Testing_label5_dice", metric_test_label5, epoch + 1)
                    writer.add_scalar("Training_dice", metric_train, epoch + 1)
                    writer.add_scalar("Training_label1_dice", metric_train_label1, epoch + 1)
                    writer.add_scalar("Training_label2_dice", metric_train_label2, epoch + 1)
                    writer.add_scalar("Training_label3_dice", metric_train_label3, epoch + 1)
                    writer.add_scalar("Training_label4_dice", metric_train_label4, epoch + 1)
                    writer.add_scalar("Training_label5_dice", metric_train_label5, epoch + 1)
                    writer.add_scalar("Validation_dice", metric, epoch + 1)
                    writer.add_scalar("Validation_label1_dice", metric_label1, epoch + 1)
                    writer.add_scalar("Validation_label2_dice", metric_label2, epoch + 1)
                    writer.add_scalar("Validation_label3_dice", metric_label3, epoch + 1)
                    writer.add_scalar("Validation_label4_dice", metric_label4, epoch + 1)
                    writer.add_scalar("Validation_label5_dice", metric_label5, epoch + 1)

                    # 多类分割
                    val_outputs = join_channels(val_outputs, opt.out_channels)
                    val_labels = join_channels(val_labels, opt.out_channels)
                    # test_outputs = join_channels(test_outputs, opt.out_channels)
                    # test_labels = join_channels(test_labels, opt.out_channels)
                    train_outputs = join_channels(train_outputs, opt.out_channels)
                    train_labels = join_channels(train_labels, opt.out_channels)

                    # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                    val_outputs = (val_outputs.sigmoid() >= 0.5).float()
                    plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="validation image")
                    plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="validation label")
                    plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="validation inference")
                    # plot_2d_or_3d_image(test_images, epoch + 1, writer, index=0, tag="test image")
                    # plot_2d_or_3d_image(test_labels, epoch + 1, writer, index=0, tag="test label")
                    # plot_2d_or_3d_image(test_outputs, epoch + 1, writer, index=0, tag="test inference")
                    plot_2d_or_3d_image(train_images, epoch + 1, writer, index=0, tag="train image")
                    plot_2d_or_3d_image(train_labels, epoch + 1, writer, index=0, tag="train label")
                    plot_2d_or_3d_image(train_outputs, epoch + 1, writer, index=0, tag="train inference")

        #print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        self.textEdit.append(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}" + '\n')
        QtWidgets.QApplication.processEvents()
        # writer.close()

    def start_train(self):
        self.Train()


class Predict_window(predict.Ui_MainWindow, QMainWindow):

    def __init__(self):
        super(Predict_window, self).__init__()
        self.setupUi(self)



        #self.pushButton.clicked.connect(self.close)
    def msg1(self, Filepath):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", "")  # 起始路径
        #self.fileT.setText(directory)
        self.lineEdit.setText(directory)
    def msg2(self, Filepath):
        directory = QtWidgets.QFileDialog.getOpenFileName(None, "选取文件夹", "")  # 起始路径
        #self.fileT.setText(directory)
        self.lineEdit_2.setText(directory[0])


    def start(self):
        path = self.lineEdit.text()
        path_ct = path + '/val/'
        path_label = path + '/val_label_ca_new/'
        for root, dirs, files in os.walk(path_ct):
            for file in files:
                path_ct_file = os.path.join(root, file)

                #print("现对 " + file + " 文件预测：")
                #self.textEdit.append("现对 " + file + " 文件预测："+ '\n')
                QtWidgets.QApplication.processEvents()
                path_label_file = path_label + "label_" + file
                segment(path_ct_file, None, path_label_file,
                        self.lineEdit_2.text(), None,
                        (112, 112, 112),
                        'nnunet', '0', 6)
                #self.textEdit.append(file+'预测完成' + '\n')
                QtWidgets.QApplication.processEvents()
        self.textEdit.append('全部预测完成' + '\n')
        QtWidgets.QApplication.processEvents()


class FixHole_window(FixHole.Ui_MainWindow, QMainWindow):

    def __init__(self):
        super(FixHole_window, self).__init__()
        self.setupUi(self)


    def msg1(self, Filepath):
        directory = QtWidgets.QFileDialog.getOpenFileName(None, "选取文件", "")  # 起始路径
        #self.fileT.setText(directory)
        self.lineEdit.setText(directory[0])
    def start(self):
        filename = self.lineEdit.text()
        img = nib.load(filename)
        data = img.get_fdata()

        mask = data > 0
        labels, num = ndimage.label(mask)
        holes = np.setdiff1d(np.arange(1, num + 1), np.unique(labels))

        for hole in holes:
            hole_mask = labels == hole
            hole_slices = ndimage.find_objects(hole_mask)
            for hole_slice in hole_slices:
                data[hole_slice][hole_mask[hole_slice]] = ndimage.uniform_filter(data[hole_slice], size=3)[
                    hole_mask[hole_slice]]

        new_img = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(new_img, 'new_filename.nii')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    #app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    First_page_Obj = First_page_window()
    ReadLabel_Obj = ReadLabel_window()
    Process_Obj = Process_window()
    TransLabelOrder_Obj = TransLabelOrder_window()
    UnzipData_Obj = UnzipData_window()
    DataRotate_Obj = DataRotate_window()
    TrainModel_Obj = TrainModel_window()
    show_Obj = MainDialogImgBW()
    predict_Obj = Predict_window()
    FixHole_Obj = FixHole_window()

    #First_page_Obj.show()
    show_Obj.show()

    First_page_Obj.pushButton.clicked.connect(Process_Obj.show)
    First_page_Obj.pushButton.clicked.connect(First_page_Obj.hide)
    First_page_Obj.pushButton_2.clicked.connect(DataRotate_Obj.show)
    First_page_Obj.pushButton_3.clicked.connect(TrainModel_Obj.show)
    First_page_Obj.pushButton_4.clicked.connect(show_Obj.show)
    Process_Obj.pushButton.clicked.connect(ReadLabel_Obj.show)
    Process_Obj.pushButton_2.clicked.connect(TransLabelOrder_Obj.show)
    Process_Obj.pushButton_3.clicked.connect(UnzipData_Obj.show)
    Process_Obj.pushButton_4.clicked.connect(First_page_Obj.show)
    Process_Obj.pushButton_4.clicked.connect(Process_Obj.hide)
    ReadLabel_Obj.pushButton.clicked.connect(ReadLabel_Obj.msg)
    ReadLabel_Obj.pushButton_2.clicked.connect(ReadLabel_Obj.Read)
    TransLabelOrder_Obj.pushButton.clicked.connect(TransLabelOrder_Obj.msg)
    TransLabelOrder_Obj.pushButton_2.clicked.connect(TransLabelOrder_Obj.Trans)
    DataRotate_Obj.pushButton.clicked.connect(DataRotate_Obj.msg1)
    DataRotate_Obj.pushButton_2.clicked.connect(DataRotate_Obj.msg2)
    DataRotate_Obj.pushButton_3.clicked.connect(DataRotate_Obj.Rotate)
    TrainModel_Obj.pushButton.clicked.connect(TrainModel_Obj.msg1)
    TrainModel_Obj.pushButton_2.clicked.connect(TrainModel_Obj.msg2)
    TrainModel_Obj.pushButton_3.clicked.connect(TrainModel_Obj.start_train)
    show_Obj.pushButton_3.clicked.connect(ReadLabel_Obj.show)
    show_Obj.pushButton_4.clicked.connect(TransLabelOrder_Obj.show)
    show_Obj.pushButton_5.clicked.connect(UnzipData_Obj.show)
    show_Obj.pushButton_6.clicked.connect(DataRotate_Obj.show)
    show_Obj.pushButton_7.clicked.connect(TrainModel_Obj.show)
    show_Obj.pushButton_9.clicked.connect(predict_Obj.show)
    show_Obj.pushButton_10.clicked.connect(FixHole_Obj.show)
    predict_Obj.pushButton.clicked.connect(predict_Obj.msg1)
    predict_Obj.pushButton_2.clicked.connect(predict_Obj.msg2)
    predict_Obj.pushButton_3.clicked.connect(predict_Obj.start)
    FixHole_Obj.pushButton.clicked.connect(FixHole_Obj.msg1)
    FixHole_Obj.pushButton_2.clicked.connect(FixHole_Obj.start)

    sys.exit(app.exec_())