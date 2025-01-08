import string
from collections import OrderedDict
import torch
import torch.nn.functional as F
import logging

import cv2
import numpy as np

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.metrics import utils_moran
from basicsr.metrics.crnn import CRNN
from basicsr.metrics.labelmaps import get_vocabulary
from basicsr.metrics.moran import MORAN
from basicsr.metrics.recognizer import RecognizerBuilder
from basicsr.metrics.utils_moran import loadData
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_moran_accuracy(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    # 获取 opt 和 label
    opt = kwargs.get('opt', None)
    if opt is None:
        raise ValueError("Option 'opt' is required for TextRecognitionMetric.")

    labels = kwargs.get('label', [])
    if not labels:
        logging.warning("No labels found. Skipping MORAN accuracy calculation for this batch.")
        return 0.0

    # 初始化 MORAN (只在第一次调用时)
    if not hasattr(calculate_moran_accuracy, 'moran'):
        moran_config = opt['models']['moran']
        alphabet = ':'.join(string.digits + string.ascii_lowercase + '$')
        calculate_moran_accuracy.converter_moran = utils_moran.strLabelConverterForAttention(alphabet, ':')

        moran = MORAN(
            1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True,
            inputDataType='torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor',
            CUDA=torch.cuda.is_available()
        )
        logging.info(f"Loading pre-trained MORAN model from {moran_config['pretrained']}")
        state_dict = torch.load(moran_config['pretrained'], map_location='cuda' if torch.cuda.is_available() else 'cpu')
        moran_state_dict_rename = OrderedDict(
            (k.replace("module.", ""), v) for k, v in state_dict.items()
        )
        moran.load_state_dict(moran_state_dict_rename)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        moran = moran.to(device)
        moran.eval()
        calculate_moran_accuracy.moran = moran

    device = next(calculate_moran_accuracy.moran.parameters()).device

    # 处理 img (HWC->CHW->NCHW)
    if isinstance(img, np.ndarray):
        # 假设img是HWC
        img = torch.from_numpy(img).float().permute(2,0,1) # CHW
    else:
        # img是Tensor
        img = img.float()
        if input_order == 'HWC':
            img = img.permute(2, 0, 1)

    img = img.unsqueeze(0).to(device)  # NCHW

    # 插值到(32, 100)
    sr_images_resized = F.interpolate(img, size=(32, 100), mode='bicubic', align_corners=True)  # N,C,H,W

    # 通道提取
    R = sr_images_resized[:, 0:1, :, :]
    G = sr_images_resized[:, 1:2, :, :]
    B = sr_images_resized[:, 2:3, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B  # N,1,H,W
    batch_size = tensor.shape[0]

    # dummy text
    text = torch.LongTensor(batch_size * 5).fill_(0).to(device)  # assuming max_iter=5
    length = torch.IntTensor(batch_size).fill_(5).to(device)
    max_iter = 20
    t, l = calculate_moran_accuracy.converter_moran.encode(['0' * max_iter] * batch_size)
    loadData(text, t)
    loadData(length, l)

    # Run MORAN
    moran_output = calculate_moran_accuracy.moran(tensor, length, text, text)
    moran_pred = decode_output_moran(moran_output, calculate_moran_accuracy.converter_moran)

    # Calculate accuracy
    correct = sum(1 for pred, label in zip(moran_pred, labels) if pred.strip().lower() == label.strip().lower())
    accuracy = correct / len(labels)

    return accuracy




@METRIC_REGISTRY.register()
def calculate_crnn_accuracy(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """
    Calculate CRNN-based text recognition accuracy.

    Args:
        img (ndarray or torch.Tensor): SR image with range [0, 255].
        img2 (ndarray or torch.Tensor): GT image with range [0, 255]. (Unused)
        crop_border (int): Crop border (Unused).
        input_order (str): Order of dimensions ('HWC' or 'CHW'). Default: 'HWC'.
        test_y_channel (bool): Test on Y channel (Unused).
        **kwargs: Additional arguments, expects 'label' and 'opt'.

    Returns:
        float: CRNN accuracy for the batch.
    """
    if not hasattr(calculate_crnn_accuracy, 'crnn'):
        # Initialize CRNN model
        opt = kwargs.get('opt', None)
        if opt is None:
            raise ValueError("Option 'opt' is required for TextRecognitionMetric.")

        crnn_config = opt['models']['crnn']
        calculate_crnn_accuracy.converter_crnn = {
            'char2id': {char: idx for idx, char in enumerate(string.digits + string.ascii_lowercase + '<BLANK>')},
            'id2char': {idx: char for idx, char in enumerate(string.digits + string.ascii_lowercase + '<BLANK>')}
        }

        crnn = CRNN(32, 1, 37, 256).to('cuda' if torch.cuda.is_available() else 'cpu')  # 37 classes
        logging.info(f"Loading pre-trained CRNN model from {crnn_config['pretrained']}")
        state_dict = torch.load(crnn_config['pretrained'], map_location='cpu')
        crnn.load_state_dict(state_dict)
        crnn = crnn.to('cuda' if torch.cuda.is_available() else 'cpu')
        crnn.eval()
        calculate_crnn_accuracy.crnn = crnn

    # Get labels
    labels = kwargs.get('label', [])
    if not labels:
        logging.warning("No labels found. Skipping CRNN accuracy calculation for this batch.")
        return 0.0

    # Process img
    if isinstance(img, torch.Tensor):
        sr_images = img.to(calculate_crnn_accuracy.crnn.device)
    else:
        sr_images = torch.from_numpy(img).float().to(calculate_crnn_accuracy.crnn.device)

    # Resize to (32, 100)
    sr_images_resized = F.interpolate(sr_images.unsqueeze(0), size=(32, 100), mode='bicubic',
                                      align_corners=True).squeeze(0)
    R = sr_images_resized[:, 0:1, :, :]
    G = sr_images_resized[:, 1:2, :, :]
    B = sr_images_resized[:, 2:3, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B

    # Run CRNN
    crnn_output = calculate_crnn_accuracy.crnn(tensor)
    crnn_pred = decode_output_crnn(crnn_output, calculate_crnn_accuracy.converter_crnn)

    # Calculate accuracy
    correct = 0
    for pred, label in zip(crnn_pred, labels):
        if pred.strip().lower() == label.strip().lower():
            correct += 1
    accuracy = correct / len(labels)

    return accuracy


@METRIC_REGISTRY.register()
def calculate_aster_accuracy(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """
    Calculate ASTER-based text recognition accuracy.

    Args:
        img (ndarray or torch.Tensor): SR image with range [0, 255].
        img2 (ndarray or torch.Tensor): GT image with range [0, 255]. (Unused)
        crop_border (int): Crop border (Unused).
        input_order (str): Order of dimensions ('HWC' or 'CHW'). Default: 'HWC'.
        test_y_channel (bool): Test on Y channel (Unused).
        **kwargs: Additional arguments, expects 'label' and 'opt'.

    Returns:
        float: ASTER accuracy for the batch.
    """
    if not hasattr(calculate_aster_accuracy, 'aster'):
        # Initialize ASTER model
        opt = kwargs.get('opt', None)
        if opt is None:
            raise ValueError("Option 'opt' is required for TextRecognitionMetric.")

        aster_config = opt['models']['aster']
        aster_info = AsterInfo(aster_config['voc_type'])
        calculate_aster_accuracy.aster_info = aster_info

        aster = RecognizerBuilder(
            arch='ResNet_ASTER',
            rec_num_classes=aster_info.rec_num_classes,
            sDim=512,
            attDim=512,
            max_len_labels=aster_info.max_len,
            eos=aster_info.char2id[aster_info.EOS],
            STN_ON=True
        )
        logging.info(f"Loading pre-trained ASTER model from {aster_config['pretrained']}")
        state_dict = \
        torch.load(aster_config['pretrained'], map_location='cuda' if torch.cuda.is_available() else 'cpu')[
            'state_dict']
        # 移除 'module.' 前缀（如果存在）
        state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
        aster.load_state_dict(state_dict)
        aster = aster.to('cuda' if torch.cuda.is_available() else 'cpu')
        aster.eval()
        calculate_aster_accuracy.aster = aster

    # Get labels
    labels = kwargs.get('label', [])
    if not labels:
        logging.warning("No labels found. Skipping ASTER accuracy calculation for this batch.")
        return 0.0

    # Process img
    if isinstance(img, torch.Tensor):
        sr_images = img.to(calculate_aster_accuracy.aster.device)
    else:
        sr_images = torch.from_numpy(img).float().to(calculate_aster_accuracy.aster.device)

    # Resize to (32, 100)
    sr_images_resized = F.interpolate(sr_images.unsqueeze(0), size=(32, 100), mode='bicubic',
                                      align_corners=True).squeeze(0)

    # Normalize to [-1, 1]
    sr_images_normalized = sr_images_resized * 2 - 1

    # Prepare input dict
    aster_info = calculate_aster_accuracy.aster_info
    input_dict = {
        'images': sr_images_normalized,
        'rec_targets': torch.IntTensor(sr_images_normalized.shape[0], aster_info.max_len).fill_(1).to(
            calculate_aster_accuracy.aster.device),
        'rec_lengths': [aster_info.max_len] * sr_images_normalized.shape[0]
    }

    # Run ASTER
    aster_output = calculate_aster_accuracy.aster(input_dict)
    aster_pred = decode_output_aster(aster_output, calculate_aster_accuracy.aster)

    # Calculate accuracy
    correct = 0
    for pred, label in zip(aster_pred, labels):
        if pred.strip().lower() == label.strip().lower():
            correct += 1
    accuracy = correct / len(labels)

    return accuracy


@METRIC_REGISTRY.register()
def calculate_average_accuracy(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """
    Calculate average text recognition accuracy from MORAN, CRNN, and ASTER.

    Args:
        img (ndarray or torch.Tensor): SR image with range [0, 255].
        img2 (ndarray or torch.Tensor): GT image with range [0, 255]. (Unused)
        crop_border (int): Crop border (Unused).
        input_order (str): Order of dimensions ('HWC' or 'CHW'). Default: 'HWC'.
        test_y_channel (bool): Test on Y channel (Unused).
        **kwargs: Additional arguments, expects 'label' and 'opt'.

    Returns:
        float: Average accuracy across MORAN, CRNN, and ASTER.
    """
    moran_acc = calculate_moran_accuracy(img, img2, crop_border, input_order, test_y_channel, **kwargs)
    crnn_acc = calculate_crnn_accuracy(img, img2, crop_border, input_order, test_y_channel, **kwargs)
    aster_acc = calculate_aster_accuracy(img, img2, crop_border, input_order, test_y_channel, **kwargs)

    average_acc = (moran_acc + crnn_acc + aster_acc) / 3
    return average_acc


def decode_output_moran(output, converter):
    """
    解码 MORAN 模型的输出为可读文本。

    Args:
        output (tuple): MORAN 模型的输出。
        converter (strLabelConverterForAttention): 转换器对象。

    Returns:
        list of str: 解码后的文本列表。
    """
    preds, preds_reverse = output[0]
    _, preds = preds.max(1)
    sim_preds = []
    for pred in preds:
        pred_chars = []
        for p in pred:
            char = converter.id2char.get(int(p), '')
            if char == '$':
                break
            pred_chars.append(char)
        sim_preds.append(''.join(pred_chars))
    return sim_preds


def decode_output_crnn(output, converter):
    """
    解码 CRNN 模型的输出为可读文本。

    Args:
        output (torch.Tensor): CRNN 模型的输出。
        converter (dict): 包含 'char2id' 和 'id2char' 的字典。

    Returns:
        list of str: 解码后的文本列表。
    """
    preds = output.argmax(dim=2)  # [batch, seq_length]
    preds = preds.cpu().numpy()
    pred_texts = []
    for pred in preds:
        pred_text = ''.join([converter['id2char'][int(x)] for x in pred if x != converter['char2id']['<BLANK>']])
        pred_texts.append(pred_text)
    return pred_texts


def decode_output_aster(output, aster_model):
    """
    解码 ASTER 模型的输出为可读文本。

    Args:
        output (dict): ASTER 模型的输出。
        aster_model (RecognizerBuilder): ASTER 模型对象。

    Returns:
        list of str: 解码后的文本列表。
    """
    if 'pred_rec' in output['output']:
        pred = output['output']['pred_rec']
        pred = pred.permute(1, 0, 2).contiguous()
        preds = aster_model.string_process(pred)
        return preds
    else:
        # 根据 ASTER 的实际输出结构调整
        return []

class AsterInfo:
    def __init__(self, voc_type):
        self.voc_type = voc_type
        assert voc_type in ['digit', 'lower', 'upper', 'all', 'chinese']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)