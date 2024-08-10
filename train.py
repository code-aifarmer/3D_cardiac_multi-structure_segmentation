import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
from UNet2 import UNet
from init import Options
from networks import build_net, update_learning_rate, build_UNETR
# from networks import build_net
import logging
#import os`
import sys
import tempfile
from glob import glob

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


# class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
#     """
#     Convert labels to multi channels based on brats classes:
#     label 1 is the peritumoral edema
#     label 2 is the GD-enhancing tumor
#     label 3 is the necrotic and non-enhancing tumor core
#     The possible classes are TC (Tumor core), WT (Whole tumor)
#     and ET (Enhancing tumor).
#
#     """
#
#     def __call__(self, data):
#         d = dict(data)
#         for key in self.keys:
#             result = []
#
#             result.append((d[key] == 0))
#             # merge label 2 and label 3 to construct TC
#             result.append(d[key] == 1)
#             # merge labels 1, 2 and 3 to construct WT
#             result.append(d[key] == 2)
#             # label 2 is ET
#             result.append(d[key] == 3)
#             d[key] = torch.stack(result, axis=0).float()
#         return d


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


# def CELoss(input: torch.Tensor, target: torch.Tensor):
#     """
#     Compute CrossEntropy loss for the input and target.
#     Will remove the channel dim according to PyTorch CrossEntropyLoss:
#     https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.
#
#     """
#     ce_func = torch.nn.CrossEntropyLoss(reduction="mean")
#     n_pred_ch = input.shape[1]
#     target = one_hot(target, num_classes=n_pred_ch)
#     n_target_ch = target.shape[1]
#     if n_pred_ch == n_target_ch:
#         # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
#         target = torch.argmax(target, dim=1)
#     else:
#         target = torch.squeeze(target, dim=1)
#     target = target.long()
#     return ce_func(input, target)


def main():
    opt = Options().parse()
    # monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # check gpus
    if opt.gpu_ids != '-1':
        num_gpus = len(opt.gpu_ids.split(','))
    else:
        num_gpus = 0
    print('number of GPU:', num_gpus)

    # Data loader creation
    # train images
    train_images = sorted(glob(os.path.join(opt.images_folder, 'train', '*.nii')))
    train_segs = sorted(glob(os.path.join(opt.labels_folder, 'train', '*.nii')))

    train_images_for_dice = sorted(glob(os .path.join(opt.images_folder, 'train', '*.nii')))
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

    print('Number of training patches per epoch:', len(train_images))
    print('Number of training images per epoch:', len(train_images_for_dice))
    print('Number of validation images per epoch:', len(val_images))
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
            #EnsureChannelFirstd(keys="image"),
            #EnsureTyped(keys=["image", "label"]),
            #ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            AddChanneld(keys=['image', 'label']),
            # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),  # CT HU filter
            # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
            #CropForegroundd(keys=['image', 'label'], source_key='image'),               # crop CropForeground

            NormalizeIntensityd(keys=['image']),                                          # augmentation
            ScaleIntensityd(keys=['image']),                                              # intensity
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

            SpatialPadd(keys=['image', 'label'], spatial_size=opt.patch_size, method= 'end'),  # pad if the image is smaller than patch
            #RandSpatialCropd(keys=['image', 'label'], roi_size=opt.patch_size, random_size=False),
            ToTensord(keys=['image', 'label'])
        ]

        val_transforms = [
            LoadImaged(keys=['image', 'label']),
            #EnsureChannelFirstd(keys="image"),
            #EnsureTyped(keys=["image", "label"]),
            #ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            AddChanneld(keys=['image', 'label']),
            # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),
            # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
            #CropForegroundd(keys=['image', 'label'], source_key='image'),                   # crop CropForeground

            NormalizeIntensityd(keys=['image']),
            ScaleIntensityd(keys=['image']),
            Spacingd(keys=['image', 'label'], pixdim=opt.resolution, mode=('bilinear', 'nearest')),  # resolution

            SpatialPadd(keys=['image', 'label'], spatial_size=opt.patch_size, method= 'end'),  # pad if the image is smaller than patch
            ToTensord(keys=['image', 'label'])
        ]
    else:
        train_transforms = [
            LoadImaged(keys=['image', 'label']),
            #EnsureChannelFirstd(keys="image"),
            #EnsureTyped(keys=["image", "label"]),
            #ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            AddChanneld(keys=['image', 'label']),
            # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),
            # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
            #CropForegroundd(keys=['image', 'label'], source_key='image'),               # crop CropForeground

            NormalizeIntensityd(keys=['image']),                                          # augmentation
            ScaleIntensityd(keys=['image']),                                              # intensity

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

            SpatialPadd(keys=['image', 'label'], spatial_size=opt.patch_size, method= 'end'),  # pad if the image is smaller than patch
            RandSpatialCropd(keys=['image', 'label'], roi_size=opt.patch_size, random_size=False),
            ToTensord(keys=['image', 'label'])
        ]

        val_transforms = [
            LoadImaged(keys=['image', 'label']),
            #EnsureChannelFirstd(keys="image"),
            #EnsureTyped(keys=["image", "label"]),
            #ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            AddChanneld(keys=['image', 'label']),
            # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),
            # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
            #CropForegroundd(keys=['image', 'label'], source_key='image'),                   # crop CropForeground

            NormalizeIntensityd(keys=['image']),                                      # intensity
            ScaleIntensityd(keys=['image']),

            SpatialPadd(keys=['image', 'label'], spatial_size=opt.patch_size, method= 'end'),  # pad if the image is smaller than patch
            ToTensord(keys=['image', 'label'])
        ]

    train_transforms = Compose(train_transforms)
    val_transforms = Compose(val_transforms)

    # create a training data loader
    check_train = monai.data.Dataset(data=train_dicts, transform=train_transforms)
    train_loader = DataLoader(check_train, batch_size=opt.batch_size, shuffle=True, collate_fn=list_data_collate, num_workers=opt.workers, pin_memory=False)

    # create a training_dice data loader
    check_val = monai.data.Dataset(data=train_dice_dicts, transform=val_transforms)
    train_dice_loader = DataLoader(check_val, batch_size=1, num_workers=opt.workers, collate_fn=list_data_collate, pin_memory=False)

    # create a validation data loader
    check_val = monai.data.Dataset(data=val_dicts, transform=val_transforms)
    val_loader = DataLoader(check_val, batch_size=1, num_workers=opt.workers, collate_fn=list_data_collate, pin_memory=False)

    # create a validation data loader
    # check_val = monai.data.Dataset(data=test_dicts, transform=val_transforms)
    # test_loader = DataLoader(check_val, batch_size=1, num_workers=opt.workers, collate_fn=list_data_collate, pin_memory=False)


    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # build the network
    # if opt.network == 'nnunet':
    #     net = build_net()  # nn build_net
    # elif opt.network == 'unetr':
    #     net = build_UNETR() # UneTR
    net = UNet(1,6)
    net.cuda()


    if num_gpus > 1:
        net = torch.nn.DataParallel(net)
    if opt.preload is not None:
        net.load_state_dict(torch.load(opt.preload))


    #loss_function = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
    loss_function = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)

    post_trans = Compose([EnsureType(), Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=6, n_classes=opt.out_channels, threshold_values=True)])
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
        print("-" * 10)
        print(f"epoch {epoch + 1}/{opt.epochs}")
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
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        #update_learning_rate(net_scheduler, optim)

        if (epoch + 1) % val_interval == 0:
            net.eval()
            with torch.no_grad():

                def plot_dice(images_loader):

                    val_images = None
                    val_labels = None
                    val_outputs = None
                    print(len(images_loader))
                    for data in images_loader:
                        val_images, val_labels = data["image"].cuda(), data["label"].cuda()
                        print(f"Type of val_images: {type(val_images)}")
                        print(f"Type of val_labels: {type(val_labels)}")

                        # val_images, val_labels = (
                        #     data["image"].cuda(),
                        #     data["label"].cuda(),
                        # )
                        roi_size = opt.patch_size
                        sw_batch_size = 4
                        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
                        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                        val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                        print(f"Type of processed val_outputs: {type(val_outputs[0])}")
                        print(f"Type of processed val_labels: {type(val_labels[0])}")

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
                metric, metric_label1, metric_label2, metric_label3, metric_label4, metric_label5, val_images, val_labels, val_outputs = plot_dice(val_loader)


                metric_train, metric_train_label1, metric_train_label2, metric_train_label3, metric_train_label4, metric_train_label5, train_images, train_labels, train_outputs = plot_dice(train_dice_loader)


                # Save best model
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    if epoch >= 30:
                        torch.save(net.state_dict(), f"./model/epoch__{epoch}__{metric}_model.pth")
                        print("saved new best metric model")
                if epoch + 1 == 150:
                    torch.save(net.state_dict(), f"./model/epoch__{epoch}__{metric}_the_last_model.pth")


                #metric_test, metric_test_label1, metric_test_label2, metric_test_label3, metric_test_label4, metric_test_label5, test_images, test_labels, test_outputs = plot_dice(test_loader)
                #metric_train, train_images, train_labels, train_outputs = plot_dice(train_dice_loader)
                #metric_test, test_images, test_labels, test_outputs = plot_dice(test_loader)

                # Logger bar
                # print(
                #     "current epoch: {} Training dice: {:.4f} Validation dice: {:.4f} Val_label1 dice: {:.4f} Val_label2 dice: {:.4f} Val_label3 dice: {:.4f} Val_label4 dice: {:.4f} Val_label5 dice: {:.4f} Best Validation dice: {:.4f} at epoch {}".format(
                #         epoch + 1, metric_train, metric, metric_label1, metric_label2, metric_label3, metric_label4, metric_label5, best_metric, best_metric_epoch
                #     )
                # )
                print(
                    "current epoch: {} \n "
                    "Validation dice: {:.4f} Val_label1 dice: {:.4f} Val_label2 dice: {:.4f} Val_label3 dice: {:.4f} Val_label4 dice: {:.4f} Val_label5 dice: {:.4f} \n "
                    "Training dice: {:.4f} train_label1 dice: {:.4f} train_label2 dice: {:.4f} train_label3 dice: {:.4f} train_label4 dice: {:.4f} train_label5 dice: {:.4f} \n "
                    "Best Validation dice: {:.4f} at epoch {}".format(
                        epoch + 1,
                        metric, metric_label1, metric_label2, metric_label3, metric_label4, metric_label5,
                        metric_train, metric_train_label1, metric_train_label2, metric_train_label3,metric_train_label4, metric_train_label5,
                        best_metric, best_metric_epoch
                    )
                )


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

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    # writer.close()


if __name__ == "__main__":
    main()
