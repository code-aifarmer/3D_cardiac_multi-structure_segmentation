import numpy as np
import SimpleITK as sitk
from glob import glob
import os

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int) #spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled


path = r'E:/data/data-one/cut400al'
path_ct = path + '/ct/'
path_label = path + '/label/'
path_ct_cut = path + '/ct-160/'
path_label_cut = path + '/label-160/'


if not os.path.exists(path_ct_cut):
        os.makedirs(path_ct_cut)
if not os.path.exists(path_label_cut):
        os.makedirs(path_label_cut)


path_ct_file = 'C:/Users/12234/Desktop/multi-seg/Data_folder/ResizeData/images/train/image18.nii'

print("现对 " + path_ct_file + " 文件裁剪：")

path_label_file = 'C:/Users/12234/Desktop/multi-seg/Data_folder/ResizeData/labels/train/label18.nii'

image = sitk.ReadImage(path_ct_file, sitk.sitkFloat32)
image_array = sitk.GetArrayFromImage(image)
path_ct_cut_file = 'C:/Users/12234/Desktop/multi-seg/Data_folder/ResizeData/cutImage/image37.nii'
print(path_ct_cut_file)
itkimgResampled = resize_image_itk(image, (256, 256, 256),resamplemethod=sitk.sitkLinear)  # 这里要注意：mask用最近邻插值，CT图像用线性插值
sitk.WriteImage(itkimgResampled, path_ct_cut_file)


mask=sitk.ReadImage(path_label_file, sitk.sitkInt8)
mask_array = sitk.GetArrayFromImage(mask)
itkmaskResampled = resize_image_itk(mask, (256, 256, 256),resamplemethod= sitk.sitkNearestNeighbor) #这里要注意：mask用最近邻插值，CT图像用线性插值

path_label_cut_file = 'C:/Users/12234/Desktop/multi-seg/Data_folder/ResizeData/cutlabel/label37.nii'
#print(path_label_cut_file)

sitk.WriteImage(itkmaskResampled, path_label_cut_file)

print(path_ct_file + '--转换完成！')
print()