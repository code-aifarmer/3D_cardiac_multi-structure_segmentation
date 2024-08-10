import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
def load_nifti_image(file_path):
    # 加载NIfTI文件
    nii = nib.load(file_path)
    image = nii.get_fdata()
    return image, nii.affine
def apply_ct_windowing(image, window_center, window_width):
    # 计算窗宽的上下界
    lower_bound = window_center - window_width / 2
    upper_bound = window_center + window_width / 2
    # 应用窗宽窗位
    windowed_image = np.clip(image, lower_bound, upper_bound)
    return windowed_image
def display_images(original, windowed):
    # 使用matplotlib显示图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original[:, :, original.shape[2]//2], cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(windowed[:, :, windowed.shape[2]//2], cmap='gray')
    plt.title('Windowed Image')
    plt.show()
def main():
    file_path = 'C:/Users/12234/Desktop/Traindata/multi-seg/multi-seg/Data_folder/images/train/image7.nii'  # 更改为你的NIfTI文件路径
    image, affine = load_nifti_image(file_path)

    # 设置心脏CT的窗宽窗位，这里是一个常见的软组织窗
    window_center = 90  # Hounsfield units
    window_width = 1200  # Hounsfield units

    # 应用窗宽窗位调整
    windowed_image = apply_ct_windowing(image, window_center, window_width)

    # 显示结果
    display_images(image, windowed_image)

if __name__ == '__main__':
    main()
