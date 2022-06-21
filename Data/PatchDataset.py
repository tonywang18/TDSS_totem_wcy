import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class PatchDataset(Dataset):

    """
    A PyTorch dataset that returns preprocessed tissue image patches

    Attributes
    -----------
    path_to_patches : str
        standard linux path string to the directory that stores tissue image patches
    transform : function
        torchvision.transforms function that preprocesses the image patch
    """

    _VALID_EXTENSIONS = ["PNG", "JPG", "JPEG", "BMP"]

    def __init__(self, path_to_patches, transform):

        self.transform = transform
        
        # 判断提供的路径是否存在
        assert os.path.exists(path_to_patches), "Supplied path [{}] either does not exist or is not a directory. Double check the path_to_patches config value.".format(path_to_patches)

        #判断提供路径中是否存在着图片文件
        all_file_names = os.listdir(path_to_patches)

        # 过滤掉任何没有有效图像文件格式的文件
        all_image_filenames = list(filter(lambda fn: fn.split(".")[-1].upper() in self._VALID_EXTENSIONS, all_file_names))

        assert len(all_image_filenames) > 0, "Supplied path [{}] does not contain any image files.".format(path_to_patches)

        self.patch_paths = [os.path.join(path_to_patches, fn) for fn in all_image_filenames]

    def __len__(self):
        return len(self.patch_paths)


    #在这里获取名字跟tensor
    def __getitem__(self, index):

        path = self.patch_paths[index]

        patch = Image.open(path)
        item_name=path.split(".")[0].split("/")[::-1][0]

        return item_name,self.transform(patch)

