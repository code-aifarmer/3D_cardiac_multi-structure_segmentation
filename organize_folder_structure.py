import os
import re
import argparse
import SimpleITK as sitk
import numpy as np
import random
from utils import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default='C:/Users/12234/Desktop/zip_data/image/', help='path to the images')
    parser.add_argument('--labels', default='C:/Users/12234/Desktop/zip_data/label/', help='path to the labels')
    parser.add_argument('--split_val', default=0, help='number of images for validation')
    parser.add_argument('--split_test', default=0, help='number of images for testing')
    parser.add_argument('--resolution', default=None, help='New Resolution to resample the data to same spacing')
    parser.add_argument('--smooth', default=False, help='Set True if you want to smooth a bit the binary mask')
    args = parser.parse_args()

    list_images = lstFiles(args.images)
    list_labels = lstFiles(args.labels)

    mapIndexPosition = list(zip(list_images, list_labels))  # shuffle order list
    random.shuffle(mapIndexPosition)
    list_images, list_labels = zip(*mapIndexPosition)

    os.mkdir('./Data_folder/image6')
    os.mkdir('./Data_folder/label6')

    # 1
    if not os.path.isdir('./Data_folder/image6s/train'):
        os.mkdir('./Data_folder/image6/train/')
    # 2
    if not os.path.isdir('./Data_folder/image6/val'):
        os.mkdir('./Data_folder/image6/val')

    # 3
    if not os.path.isdir('./Data_folder/image6/test'):
        os.mkdir('./Data_folder/image6/test')

    # 4
    if not os.path.isdir('./Data_folder/label6/train'):
        os.mkdir('./Data_folder/label6/train')

    # 5
    if not os.path.isdir('./Data_folder/label6/val'):
        os.mkdir('./Data_folder/label6/val')

    # 6
    if not os.path.isdir('./Data_folder/label6/test'):
        os.mkdir('./Data_folder/label6/test')

    for i in range(len(list_images)-int(args.split_test + args.split_val)):

        a = list_images[int(args.split_test + args.split_val)+i]
        b = list_labels[int(args.split_test + args.split_val)+i]

        print('train',i, a,b)

        label = sitk.ReadImage(b)
        image = sitk.ReadImage(a)

        #image = resample_sitk_image(image, spacing=args.resolution, interpolator='linear', fill_value=0)
        #image, label = uniform_img_dimensions(image, label, nearest=True)
        if args.smooth is True:
            label = gaussian2(label)

        image_directory = os.path.join('./Data_folder/image6/train', f"image{i:d}.nii")
        label_directory = os.path.join('./Data_folder/label6/train', f"label{i:d}.nii")

        sitk.WriteImage(image, image_directory)
        sitk.WriteImage(label, label_directory)

    for i in range(int(args.split_val)):

        a = list_images[int(args.split_test)+i]
        b = list_labels[int(args.split_test)+i]

        print('val',i, a,b)

        label = sitk.ReadImage(b)
        image = sitk.ReadImage(a)

        #image = resample_sitk_image(image, spacing=args.resolution, interpolator='linear', fill_value=0)
        #image, label = uniform_img_dimensions(image, label, nearest=True)
        if args.smooth is True:
            label = gaussian2(label)

        image_directory = os.path.join('./Data_folder/image6/val', f"image{i:d}.nii")
        label_directory = os.path.join('./Data_folder/label6/val', f"label{i:d}.nii")

        sitk.WriteImage(image, image_directory)
        sitk.WriteImage(label, label_directory)

    for i in range(int(args.split_test)):

        a = list_images[i]
        b = list_labels[i]

        print('test',i,a,b)

        label = sitk.ReadImage(b)
        image = sitk.ReadImage(a)

        #image = resample_sitk_image(image, spacing=args.resolution, interpolator='linear', fill_value=0)
        #image, label = uniform_img_dimensions(image, label, nearest=True)
        if args.smooth is True:
            label = gaussian2(label)

        image_directory = os.path.join('./Data_folder/image6/test', f"image{i:d}.nii")
        label_directory = os.path.join('./Data_folder/label6/test', f"label{i:d}.nii")

        sitk.WriteImage(image, image_directory)
        sitk.WriteImage(label, label_directory)

