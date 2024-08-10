import argparse
import os


class Options():

    """This class defines options used during both training and test time."""

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):

        # basic parameters
        parser.add_argument('--images_folder', type=str, default='./Data_folder/images/')
        parser.add_argument('--labels_folder', type=str, default='./Data_folder/labels/')
        parser.add_argument('--increase_factor_data',  default=0, help='Increase data number per epoch')
        parser.add_argument('--preload', type=str, default=None)
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')

        # dataset parameters
        parser.add_argument('--network', default='nnunet', help='nnunet, unetr')
        parser.add_argument('--patch_size', default=(128, 128, 128), help='Size of the patches extracted from the image')
        parser.add_argument('--spacing', default=None, help='Original Resolution')
        parser.add_argument('--resolution', default=None, help='New Resolution, if you want to resample the data in training. I suggest to resample in organize_folder_structure.py, otherwise in train resampling is slower')
        parser.add_argument('--batch_size', type=int, default=4, help='batch size, depends on your machine') #4
        parser.add_argument('--in_channels', default=1, type=int, help='Channels of the input')
        parser.add_argument('--out_channels', default=6, type=int, help='Channels of the output')

        # training parameters
        parser.add_argument('--epochs', default=400, help='Number of epochs')
        parser.add_argument('--lr', default=0.0001, help='Learning rate')
        parser.add_argument('--benchmark', default=True)
        # augmentation parameters




        parser.add_argument('--rotation', type=float, default=0.1, help='Rotation for data augmentation')
        parser.add_argument('--flip', type=bool, default=True, help='Flip for data augmentation')
        parser.add_argument('--scale', type=float, default=0.1, help='Scaling for data augmentation')

        # learning rate scheduler
        parser.add_argument('--lr_scheduler', default='cosine', help='Learning rate scheduler: cosine, plateau, etc.')
        parser.add_argument('--lr_patience', default=10, type=int, help='Patience for ReduceLROnPlateau scheduler')

        # optimizer
        parser.add_argument('--optimizer', default='AdamW', help='Optimizer to use: Adam, AdamW, SGD, etc.')








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





