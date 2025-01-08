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
def calculate_ocr_accuracy(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """
    用指定的OCR模型(aster, moran, crnn)对SR图像进行文本识别，并与label对比，计算识别准确率。

    Args:
        img (ndarray/Tensor): SR图像
        img2 (ndarray/Tensor): GT图像(未用)
        crop_border (int): 裁剪边界(未用)
        input_order (str): 输入图像顺序('HWC'或'CHW')，默认为'HWC'
        test_y_channel (bool): 是否只测试Y通道(未用)
        **kwargs:
            'opt': dict, 配置文件信息，包含 'metric': {'ocr_type': ..., 'models': {...}}
            'label': list of str, 与img对应的标签列表
    Returns:
        float: OCR识别准确率
    """
    opt = kwargs.get('opt', None)
    labels = kwargs.get('label', [])
    if opt is None:
        raise ValueError("Need 'opt' in kwargs for OCR metric.")
    if not labels:
        logging.warning("No labels found. OCR accuracy will return 0.0")
        return 0.0

    # 从opt中获取OCR配置
    metric_opt = opt['metric']
    ocr_type = metric_opt['ocr_type']  # 'aster', 'moran', 'crnn'等
    ocr_models_opt = metric_opt.get('models', {})

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ngpu = opt.get('num_gpu', 1)

    # 将img转换为Tensor, CHW格式
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float()
    if input_order == 'HWC':
        img = img.permute(2, 0, 1)  # CHW
    img = img.to(device) / 255.0

    # 使用方法属性缓存模型，避免重复初始化
    if not hasattr(calculate_ocr_accuracy, 'ocr_model'):
        calculate_ocr_accuracy.ocr_model = None
        calculate_ocr_accuracy.converter_moran = None
        calculate_ocr_accuracy.aster_info = None

    if calculate_ocr_accuracy.ocr_model is None or calculate_ocr_accuracy.ocr_model['type'] != ocr_type:
        # 初始化模型
        if ocr_type.lower() == 'moran':
            pretrained_path = ocr_models_opt['moran']['pretrained']
            model, converter_moran = init_moran_model(pretrained_path, device, ngpu)
            calculate_ocr_accuracy.ocr_model = {'type': 'moran', 'model': model, 'converter_moran': converter_moran}

        elif ocr_type.lower() == 'crnn':
            pretrained_path = ocr_models_opt['crnn']['pretrained']
            model, aster_info = init_crnn_model(pretrained_path, device,
                                                voc_type=ocr_models_opt.get('crnn', {}).get('voc_type', 'all'))
            calculate_ocr_accuracy.ocr_model = {'type': 'crnn', 'model': model, 'aster_info': aster_info}

        elif ocr_type.lower() == 'aster':
            aster_conf = ocr_models_opt['aster']
            pretrained_path = aster_conf['pretrained']
            voc_type = aster_conf.get('voc_type', 'all')
            model, aster_info = init_aster_model(pretrained_path, device, voc_type, ngpu)
            calculate_ocr_accuracy.ocr_model = {'type': 'aster', 'model': model, 'aster_info': aster_info}
        else:
            raise ValueError(f"Unsupported OCR type: {ocr_type}")

    # 推理过程
    model_dict = calculate_ocr_accuracy.ocr_model
    model = model_dict['model']

    if ocr_type.lower() == 'moran':
        tensor, length, text, text_rev = parse_moran_data(img, model_dict['converter_moran'], device)
        with torch.no_grad():
            moran_out = model(tensor, length, text, text_rev, test=True)
        preds = decode_moran_output(moran_out, model_dict['converter_moran'])

    elif ocr_type.lower() == 'crnn':
        tensor = parse_crnn_data(img, device)
        with torch.no_grad():
            crnn_out = model(tensor)
        preds = decode_crnn_output(crnn_out, model_dict['aster_info'])

    elif ocr_type.lower() == 'aster':
        input_dict = parse_aster_data(img, model_dict['aster_info'], device)
        with torch.no_grad():
            aster_out = model(input_dict)
        preds = decode_aster_output(aster_out, model_dict['aster_info'])

    else:
        preds = []

    # 计算准确率
    correct = sum(1 for p, l in zip(preds, labels) if p.strip().lower() == l.strip().lower())
    accuracy = correct / len(labels) if len(labels) > 0 else 0.0
    return accuracy


class AsterInfo(object):
    def __init__(self, voc_type):
        super(AsterInfo, self).__init__()
        assert voc_type in ['digit', 'lower', 'upper', 'all', 'chinese']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)


def init_moran_model(pretrained_path, device, ngpu=1):
    """初始化 MORAN 模型。"""
    alphabet = ':'.join(string.digits + string.ascii_lowercase + '$')
    converter_moran = utils_moran.strLabelConverterForAttention(alphabet, ':')
    MORAN_model = MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True,
                        inputDataType='torch.cuda.FloatTensor' if 'cuda' in device else 'torch.FloatTensor',
                        CUDA='cuda' in device)
    logging.info(f'Loading pre-trained MORAN model from {pretrained_path}')
    state_dict = torch.load(pretrained_path, map_location=device)
    MORAN_state_dict_rename = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    MORAN_model.load_state_dict(MORAN_state_dict_rename)
    MORAN_model = MORAN_model.to(device)
    if ngpu > 1:
        MORAN_model = torch.nn.DataParallel(MORAN_model, device_ids=range(ngpu))
    MORAN_model.eval()
    return MORAN_model, converter_moran


def init_crnn_model(pretrained_path, device, voc_type='all'):
    """初始化 CRNN 模型。"""
    model = CRNN(32, 1, 37, 256)  # 初始化模型
    logging.info(f"Loading pretrained CRNN model from {pretrained_path}")

    # 强制先加载到 CPU
    state_dict = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(state_dict)

    # 将模型移动到目标设备
    model = model.to(device)
    model.eval()

    # 创建字符集信息
    aster_info = AsterInfo(voc_type)
    return model, aster_info


def init_aster_model(pretrained_path, device, voc_type='all', ngpu=1):
    """初始化 Aster 模型。"""
    aster_info = AsterInfo(voc_type)
    aster = RecognizerBuilder(arch='ResNet_ASTER',
                              rec_num_classes=aster_info.rec_num_classes,
                              sDim=512,
                              attDim=512,
                              max_len_labels=aster_info.max_len,
                              eos=aster_info.char2id[aster_info.EOS],
                              STN_ON=True)
    logging.info(f"Loading pre-trained ASTER model from {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location=device)['state_dict']
    state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    aster.load_state_dict(state_dict)
    aster = aster.to(device)
    if ngpu > 1:
        aster = torch.nn.DataParallel(aster, device_ids=range(ngpu))
    aster.eval()
    return aster, aster_info


def parse_moran_data(img, converter_moran, device, width=128):
    # img: NCHW or single image CHW
    if img.dim() == 3:
        img = img.unsqueeze(0)  # NCHW
    imgs_input = F.interpolate(img, (32, width), mode='bicubic', align_corners=True)
    R = imgs_input[:, 0:1, :, :]
    G = imgs_input[:, 1:2, :, :]
    B = imgs_input[:, 2:3, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B
    batch_size = tensor.size(0)
    max_iter = 20
    text = torch.LongTensor(batch_size * max_iter).fill_(0).to(device)
    length = torch.IntTensor(batch_size).fill_(max_iter).to(device)
    t, l = converter_moran.encode(['0' * max_iter] * batch_size)
    utils_moran.loadData(text, t)
    utils_moran.loadData(length, l)
    return tensor, length, text, text


def parse_crnn_data(img, device, width=128):
    if img.dim() == 3:
        img = img.unsqueeze(0)  # NCHW
    imgs_input = F.interpolate(img, (32, width), mode='bicubic', align_corners=True)
    R = imgs_input[:, 0:1, :, :]
    G = imgs_input[:, 1:2, :, :]
    B = imgs_input[:, 2:3, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B
    return tensor


def parse_aster_data(img, aster_info, device, width=128):
    if img.dim() == 3:
        img = img.unsqueeze(0)  # NCHW
    imgs_input = F.interpolate(img, (32, width), mode='bicubic', align_corners=True).to(device)
    input_dict = {}
    input_dict['images'] = imgs_input * 2 - 1
    batch_size = imgs_input.shape[0]
    input_dict['rec_targets'] = torch.IntTensor(batch_size, aster_info.max_len).fill_(1).to(device)
    input_dict['rec_lengths'] = [aster_info.max_len] * batch_size
    return input_dict


def decode_moran_output(output, converter_moran):
    """
    解码 MORAN 输出。如果 use_ctc_like=True，视为 CRNN/CTC 方式。
    如果模型真的只输出 (N, C) 并把 '$' 当EOS，也要单独处理。
    """
    # 1) 若 MORAN 的输出是 (T, N, C)，可以复用 CRNN 的方式:

    # 2) 如果 MORAN 真的是 (N, C) 并 '$' 做EOS
    #    下面仅示例: 取一次 argmax, 如果是 '$' 就结束
    preds, _ = output  # output 可能是 (preds, preds_reverse)
    # preds shape: (N, C)
    results = []
    for pred in preds:  # preds shape: (N, C)
        idx = pred.argmax().item()
        char = converter_moran.alphabet[idx]
        if char == '$':
            break

            # 如果是空格 ' ' => 不加到结果
        if char == ' ':
            results.append("")
            continue

            # 正常字符
        results.append(char)
    print("moran", results)
    return results


def decode_crnn_output(output, aster_info=None):
    """
    CRNN输出维度通常为(N, T, C)，这里统一转换为(T, N, C)，
    然后对每个时间步 argmax，过滤blank(i==0) 并 去掉重复。

    说明:
      - 不再动态使用 aster_info.voc
      - 强行指定alphabet = '-0123456789abcdefghijklmnopqrstuvwxyz'
      - 保留原先去空白+去重的逻辑不变
    """
    # 1) 固定alphabet，假定[0] = '-'是blank
    alphabet = '-0123456789abcdefghijklmnopqrstuvwxyz'

    # 2) 维度转换: (N,T,C)->(T,N,C)，若已经(T,N,C)，就注意判定
    if output.dim() == 3 and output.shape[0] != output.shape[1]:
        out = output.permute(1, 0, 2).contiguous()  # (T,N,C)
    else:
        out = output  # 如果本身就是 (T,N,C)，就不 permute

    # 3) 解码
    seq_len, batch_size, num_classes = out.shape
    predict_result = []

    for t in range(seq_len):
        # 取当前时间步 (N, C)
        step_probs = out[t]
        # argmax -> (N,)
        max_index = step_probs.argmax(dim=1)

        # 为这个时间步生成一个结果串 (实际上batch_size个)
        out_str = ""
        last = ""
        for i in max_index:
            # i是类别索引
            if i >= len(alphabet):
                # 索引超范围,跳过
                continue

            # 查看alphabet[i]
            char = alphabet[i]

            # (1) 去空白: 如果 i==0 或 char=='-',就不拼接
            if i == 0:
                # blank
                last = ""  # 复位 last，防止blank后又出现same char无法输出
                continue

            # (2) 去重复: 如果跟上一次相同，就跳过
            if char != last:
                out_str += char
                last = char
            else:
                # 如果你想跳过重复字符,就continue
                continue

        predict_result.append(out_str)

    print("crnn:", predict_result)
    return predict_result


def decode_aster_output(output, aster_info):
    """
    解决“将 T 个时间步当成 N= T 个样本”的错误，将 pred_rec 变成 (N, T) 后只输出 N 个字符串。

    1) 如果 pred_rec 原始 shape = [T, N], 并且 T>1, N=1, 则 permute(1,0) 得 [1, T], batch=1。
    2) 若 shape 已经是 [N, T]，则不用 permute。
    3) 对每个样本 n => row: (T,) 进行解码:
        - 遇到 end_label/padding_label => break
        - 遇到 unknown_label => skip
    4) 打印索引, 仅 1 次 for n in range(N)，不会出现 100 次“sample”日志。

    Returns:
        pred_list (List[str]): 每个样本对应的解码结果
    """

    # 若 'pred_rec' 不存在，返回空
    if 'pred_rec' not in output['output']:
        return []

    # 1) 取出 pred_rec: shape 可能是 [T, N] or [N, T].
    pred_rec = output['output']['pred_rec']
    shape0, shape1 = pred_rec.shape

    # 2) 根据形状判断是否 permute
    # 目标: (N, T).
    # 常见: 训练的 pred_rec 是 (T, N)= (seq_len, batch_size) => permute(1,0).
    # 若 shape0>shape1，可能是 (T=100, N=1) => permute => (1, 100)
    # 若 shape0<shape1, 可能已经是 (N, T).
    if shape0 > shape1:

        pred_rec = pred_rec.permute(1, 0).contiguous()  # (N,T)
    else:
        print(" 1 ")

    pred_rec = pred_rec.cpu().numpy()  # => (N,T)
    N, T = pred_rec.shape

    # 3) 取出 aster_info 中的特殊 label
    end_label = aster_info.char2id[aster_info.EOS]
    padding_label = aster_info.char2id[aster_info.PADDING]
    unknown_label = aster_info.char2id[aster_info.UNKNOWN]

    pred_list = []

    # 4) 遍历 batch 内每个样本
    for n in range(N):
        row = pred_rec[n]  # shape (T,)

        pred_chars = []
        for idx in row:
            # (a) 遇到EOS or PADDING => break
            if idx == end_label or idx == padding_label:
                break

            # (b) 遇到 UNKNOWN => skip
            if idx == unknown_label:
                continue

            # (c) 若越界 => skip
            if idx < 0 or idx >= len(aster_info.id2char):
                continue

            # (d) 普通字符
            ch = aster_info.id2char[idx]
            pred_chars.append(ch)

        decoded_str = ''.join(pred_chars)
        pred_list.append(decoded_str)

    print("Decoded ASTER results:", pred_list)
    return pred_list

