#coding:utf-8
import torch
from monai.transforms import Compose, RandHistogramShiftD, Flipd, Rotate90d
import matplotlib.pyplot as plt
import SimpleITK as sitk
# start a chain of transforms
KEYS = ("image", "label")
class aug():
    def __init__(self):
        self.random_rotated = Compose([
            Rotate90d(KEYS, k=1, spatial_axes=(2,3),allow_missing_keys=True),
            Flipd(KEYS, spatial_axis=(1,2,3),allow_missing_keys=True),
            RandHistogramShiftD(KEYS,  prob=1, num_control_points=30, allow_missing_keys=True),
            # ToTensorD(KEYS),
        ])
    def forward(self,x):
        x = self.random_rotated(x)
        return x

# start a dataset
def save(before_x, after_x, new_path,new_name=""):
    after_x = after_x[0, 0,...]
    if new_name=="image":
        ct = sitk.ReadImage(before_x, sitk.sitkInt16)
    else:
        ct = sitk.ReadImage(before_x, sitk.sitkUInt8)
    predict_seg = sitk.GetImageFromArray(after_x)
    predict_seg.SetDirection(ct.GetDirection())
    predict_seg.SetOrigin(ct.GetOrigin())
    predict_seg.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(predict_seg,new_path)


if __name__ == "__main__":
    image = r"C:\Users\12234\Desktop\multi-seg\Data_folder\images\train\image0.nii"   # 原图
    label = r"C:\Users\12234\Desktop\multi-seg\Data_folder\labels\train\label0.nii"   #标签
    new_path = r"C:\Users\12234\Desktop\multi-seg\Data_folder\images\train\image20.nii"  #增强后的原图
    new_path1 = r"C:\Users\12234\Desktop\multi-seg\Data_folder\labels\train\label20.nii"  #增强后的标签

    ct = sitk.ReadImage(image)
    ct1 = sitk.GetArrayFromImage(ct)
    seg = sitk.ReadImage(label)
    seg1 = sitk.GetArrayFromImage(seg)

    ct = ct1[None, None,...]
    seg = seg1[None, None,...]

    ct = torch.from_numpy(ct)
    seg = torch.from_numpy(seg)
    m = {"image": ct,
         "label":seg}
    augs = aug()
    print(m["image"].shape)
    data_dict= augs.forward(m)

    save(image, data_dict["image"], new_path, "image")
    save(label, data_dict["label"], new_path1, "label")


    print(data_dict["image"].shape)
    plt.subplots(1, 3)
    plt.subplot(1, 3, 1);
    plt.imshow(ct1[66,...])
    plt.subplot(1, 3, 2);
    plt.imshow(data_dict["image"][0,0, 66,...])
    plt.subplot(1, 3, 3);
    plt.imshow(data_dict["label"][0,0, 66,...])
    plt.show()