import nibabel as nib
import numpy as np
import os
import random
import argparse

import scipy
from scipy.ndimage import rotate
import re
import glob
import numpy as np
import nibabel as nib
import random
import math
import re
i =0
currentImage = ""
currentLabel =""
TempImageStr = ""
TempLabelStr = ""

def rotate_nii(nii_path, degree, currentIndex, oldIndex):
    # Load the NIfTI file
    nii = nib.load(nii_path)
    img = nii.get_fdata()
    affine = nii.affine

    rotate_axis = random.choice([(0, 1), (0, 2), (1, 2)])

    # Rotate the image using scipy.ndimage.rotate
    img_rotated = scipy.ndimage.rotate(img, angle=degree, axes=rotate_axis, reshape=False, mode='nearest')

    new_path = nii_path[:-7] + str(currentIndex) + ".nii"
    rotated_nii = nib.Nifti1Image(img_rotated, affine)
    nib.save(rotated_nii, new_path)
def rotate_nii(nii_path, degree, currentIndex,oldIndex):
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
    #nib.save(rotated_nii, nii_path[:-6] + '56.nii')
    if oldIndex<10:
        nib.save(rotated_nii, nii_path[:-5] + str(currentIndex)+'.nii')
    elif oldIndex<100:
        nib.save(rotated_nii, nii_path[:-6] + str( currentIndex)+'.nii')





def get_numeric_suffix(filename):
    # 从文件名中获取最后的数字后缀
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        return 0


if __name__ == '__main__':
    Imagefolder_path = r"C:\Users\12234\Desktop\TestRorate\imagesCorrect"
    Labelfolder_path = r"C:\Users\12234\Desktop\TestRorate\labelsCorrect"
    Image_nii_files = glob.glob(os.path.join(Imagefolder_path, "*.nii"))
    Label_nii_files = glob.glob(os.path.join(Labelfolder_path, "*.nii"))



    #rotate_nii("C:/Users/12234/Desktop/mr_data/imagesCorrect/image18.nii",30.0)

    while i<min(len(Image_nii_files),len(Label_nii_files)):
        TempImageStr = Image_nii_files[i][-12:]
        TempLabelStr = Label_nii_files[i][-12:]
        currentImage = get_numeric_suffix(str(TempImageStr))
        currentImage = currentImage + len(Image_nii_files)
        currentLabel = get_numeric_suffix(str(TempLabelStr))
        currentLabel = currentLabel + len(Label_nii_files)
        rotate_nii(Image_nii_files[i], 30.0, currentImage, get_numeric_suffix(str(TempImageStr)))
        rotate_nii(Label_nii_files[i], 30.0, currentLabel, get_numeric_suffix(str(TempLabelStr)))

        i=i+1
    print("处理完成！")




