import math
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import lmdb
import six
from PIL import Image
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY

def buf2PIL(imgbuf, mode='RGB'):
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf).convert(mode)
    return im

@DATASET_REGISTRY.register()
class SingleLMDBPairedDataset(data.Dataset):
    """适用于 TextZoom 格式的单一 LMDB 文件的配对数据集。

    TextZoom 的 LMDB 结构中：
    - 有 num-samples 存储总样本数
    - 每个样本 i (1-based) 有三个键：
      * image_hr-%09d: HR 图像
      * image_lr-%09d: LR 图像
      * label-%09d: 文本标签

    修改点：
    - 在读取完成后，将HR图像调整为 (32, 128)，将LR图像调整为 (16, 64)。
    """

    def __init__(self, opt):
        super(SingleLMDBPairedDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        self.scale = opt['scale']

        self.lmdb_path = opt['dataroot_lq']  # 实际同一个LMDB文件存放HR/LR/label
        # 打开LMDB并获取样本数
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        if not self.env:
            raise IOError(f'Cannot open lmdb dataset {self.lmdb_path}')

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get(b'num-samples'))
        self.env.close()

        self.io_backend_opt['db_paths'] = [self.lmdb_path]
        self.io_backend_opt['client_keys'] = ['main']

        self.phase = self.opt['phase']
        self.gt_size = self.opt.get('gt_size', None)
        self.use_hflip = self.opt.get('use_hflip', False)
        self.use_rot = self.opt.get('use_rot', False)

        # 定义目标大小
        self.hr_size = (32, 128)  # HR: 32x128 (H=32, W=128)
        self.lr_size = (16, 64)   # LR: 16x64  (H=16, W=64)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # LMDB是1-based索引
        index += 1

        hr_key = f'image_hr-{index:09d}'
        lr_key = f'image_lr-{index:09d}'
        label_key = f'label-{index:09d}'

        hr_bytes = self.file_client.get(hr_key, 'main')
        lr_bytes = self.file_client.get(lr_key, 'main')
        label = self.file_client.get(label_key, 'main')
        if label is None:
            label_str = ''
        else:
            label_str = label.decode()

        # 解码为float32类型的RGB图像[0,1]
        img_hr = imfrombytes(hr_bytes, float32=True)
        img_lq = imfrombytes(lr_bytes, float32=True)

        # 调整图像尺寸
        # imfrombytes返回的是float32 [0,1]，cv2.resize需要 [0,255] 或 [0,1] 都可，
        # 在这里我们先转到[0,255]再resize，然后再回到[0,1]
        img_hr = (img_hr * 255.0).astype(np.uint8)
        img_hr = cv2.resize(img_hr, (self.hr_size[1], self.hr_size[0]), interpolation=cv2.INTER_CUBIC)
        img_hr = img_hr.astype(np.float32) / 255.0

        img_lq = (img_lq * 255.0).astype(np.uint8)
        img_lq = cv2.resize(img_lq, (self.lr_size[1], self.lr_size[0]), interpolation=cv2.INTER_CUBIC)
        img_lq = img_lq.astype(np.float32) / 255.0

        # 如果是训练阶段，并且有 gt_size，进行随机裁剪和数据增强
        if self.phase == 'train' and self.gt_size is not None:
            img_hr, img_lq = paired_random_crop(img_hr, img_lq, self.gt_size, self.scale, hr_key)
            img_hr, img_lq = augment([img_hr, img_lq], self.use_hflip, self.use_rot)

        # 测试/验证阶段，根据需要对HR进行裁剪使其与LQ对齐
        # 在本例中，HR和LR已经固定大小为16x64(LR)和32x128(HR)是倍数关系(2x),
        # 若需要可以执行下面的逻辑:
        if self.phase != 'train':
            # 假设HR应为LQ的scale倍大小，这里scale=2，此时HR应为(16*2,64*2)=(32,128)
            # 已经是对应大小，无需再裁剪，如果需要，可以强行按比例裁剪:
            # img_hr = img_hr[0:img_lq.shape[0]*self.scale, 0:img_lq.shape[1]*self.scale, :]
            pass

        # 转成tensor
        img_hr, img_lq = img2tensor([img_hr, img_lq], bgr2rgb=True, float32=True)

        # 可选归一化
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_hr, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_hr,
            'lq_path': lr_key,
            'gt_path': hr_key,
            'label': label_str
        }

    def __len__(self):
        return self.nSamples
