import os
from pathlib import Path
from monai import transforms

import torch
from torch.utils.data import Dataset


PATH = '/gpfs/accounts/eecs598s007f23_class_root/eecs598s007f23_class/shared_data/zhtianyu/kvasir_seg'


def build_dataset(seed, image_size):
    image_list = get_file_list(PATH + '/image', suffix='*.png')
    label_list = get_file_list(PATH + '/label', suffix='*.png')
    data = [{'image': image, 'label': label} for (image, label) in zip(image_list, label_list)]
    train_data, val_data = split_dataset(data, train_ratio=0.8, random=True)

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.ToTensord(keys=["image", "label"]),
            PadLongestSided(keys=["image", "label"], padder=None),
            transforms.Resized(
                keys=["image", "label"], spatial_size=(image_size, image_size), 
                mode=["bilinear", "nearest"], anti_aliasing=[True, False], align_corners=[False, None],
            ),        
            transforms.ToTensord(keys=["image", "label"], track_meta=False),

        ]
    )
    train_transform.set_random_state(seed)
    tr_ds = BasicDataset(data=train_data, transform=train_transform)
    
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.ToTensord(keys=["image", "label"]),
            PadLongestSided(keys=["image", "label"], padder=None),
            transforms.Resized(
                keys=["image", "label"], spatial_size=(image_size, image_size), 
                mode=["bilinear", "nearest"], anti_aliasing=[True, False], align_corners=[False, None],
            ),        
        ]
    )  
    vl_ds = BasicDataset(data=val_data, transform=val_transform)
    
    return tr_ds, vl_ds

class BasicDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(data)

        sample = {'image': data['image'], 'label': data['label']}

        image_name = str(self.data[idx]['image']).split('/')[-1].split('.')[0].split('_')[-1]
        label_name = str(self.data[idx]['label']).split('/')[-1].split('.')[0].split('_')[-1]
        assert image_name == label_name

        sample['name'] = image_name
        return (sample['image']/255, (sample['label']//255).float(), sample['name'])


def get_file_list(path, suffix):
    return sorted([p for p in Path(path).rglob(suffix)])


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return


def split_dataset(data, train_ratio, random=True):
    total_len = len(data)
    train_len = int(total_len * train_ratio)
    
    indices = torch.randperm(total_len) if random else torch.arange(0, total_len)
    train_data = [data[i] for i in indices[:train_len]]
    val_data = [data[i] for i in indices[train_len:]]
    return train_data, val_data


class PadLongestSided(transforms.Padd):
    def __call__(self, data):
        d = dict(data)
        for key, m in self.key_iterator(d, self.mode):
            _, h, w = d[key].shape
            _max = max(h, w)
            padding_left = (_max - w) // 2
            padding_right = padding_left\
                if 2*padding_left == _max-w else padding_left+1
            padding_top = (_max - h) // 2
            padding_bottom = padding_top\
                if 2*padding_top == _max-h else padding_top+1
            pad = (padding_left, padding_right, padding_top, padding_bottom)
            d[key] = torch.nn.functional.pad(d[key], pad=pad, mode=m)
        return d