import numpy as np
import SimpleITK as sitk
from glob import glob
import os
path = r'D:\Project\multi-seg\multi-seg\Data_folder\label6\train\\'

path_label = path
path_ct_cut = path + '/ct-change/'
path_label_cut = path + '/label-change/'
# path = r'E:/code2/Pytorch--3D-Medical-Images-Segmentation--SALMON-master/Data_folder'
# path_ct = path + '/CT/'
# path_label = path + '/CT_labels/'
# path_ct_cut = path + '/CT-1/'
# path_label_cut = path + '/CT_labels-1/'

if not os.path.exists(path_ct_cut):
        os.makedirs(path_ct_cut)
if not os.path.exists(path_label_cut):
        os.makedirs(path_label_cut)

for root,dirs,files in os.walk(path_label):
        for file in files:
                path_ct_file = os.path.join(root,file)

                print("现对 " + file + " 文件裁剪：")

                path_label_file = path_label + file
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

                print(file + '--转换完成！')
                print()