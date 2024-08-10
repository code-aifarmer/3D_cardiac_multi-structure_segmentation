import  SimpleITK as sitk
import numpy as np
# 读取标签数据
i=0
while i<55:

    label_path = r"C:/Users/12234/Desktop/New_data/label/label"+str(i)+".nii"
    mask = sitk.ReadImage(label_path, sitk.sitkInt8)
    label_array = sitk.GetArrayFromImage(mask)

    print(f'label_path: {label_path}')
    # 查看label里面有几种值
    print(f'标签中有几种值: {np.unique(label_array)}')

    # 查看每个标签对应多少像素
    print(f'每个标签像素数量：', np.unique(label_array, return_counts=True))
    i = i+1


