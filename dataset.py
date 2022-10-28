import os
import tarfile
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision import transforms
from PIL import Image


class Imagenette(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Imagenette, self).__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()
        if train:
            data_folder = os.path.join(self.root,'imagenette2','train')
        else:
            data_folder = os.path.join(self.root,'imagenette2','val')
        
        self.classes =  os.listdir(data_folder)
        self.targets = []
        self.filenames = []

        for idx,img_folder in enumerate(os.listdir(data_folder)):
            for image_path in os.listdir(os.path.join(data_folder,img_folder)):
                img_full_path = os.path.join(data_folder,img_folder,image_path)
                self.filenames.append(img_full_path)
                self.targets.append(idx)

    def __getitem__(self, idx):
        filename, target = self.filenames[idx], self.targets[idx]
        image = Image.open(filename)
        image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def download(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        if not os.path.exists(os.path.join(self.root,'imagenette2')):
            url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'
            download_url(url,self.root)
            file_dir = os.path.join(self.root,'imagenette2.tgz')
            with tarfile.open(file_dir, 'r:gz') as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, path=self.root)

    def __len__(self):
        return len(self.filenames)
