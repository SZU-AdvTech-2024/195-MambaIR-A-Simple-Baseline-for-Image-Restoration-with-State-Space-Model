import string
from collections import OrderedDict
import torch
import torch.nn.functional as F

from basicsr.metrics.crnn import CRNN
from basicsr.metrics.labelmaps import get_vocabulary
from basicsr.metrics.moran import MORAN
from basicsr.metrics.recognizer import RecognizerBuilder
from basicsr.metrics.utils_moran import loadData
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
class TextRecognitionMetric:
    def __init__(self, opt=None, **kwargs):
        """
        初始化文本识别率工具，加载 MORAN、CRNN 和 ASTER 模型。
        Args:
            opt (dict): BasicSR 配置文件中的配置选项。可以为空。
            **kwargs: 允许传入其他额外参数，防止意外报错。
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.opt = opt or {}  # 确保 opt 为 None 时设置为默认值

        # 初始化模型
        self.moran = self.MORAN_init(self.opt.get('models', {}).get('moran', {}))
        self.crnn = self.CRNN_init(self.opt.get('models', {}).get('crnn', {}))
        self.aster = self.ASTER_init(self.opt.get('models', {}).get('aster', {}))

    def MORAN_init(self, moran_config):
        """
        初始化 MORAN 模型。
        """
        alphabet = ':'.join(string.digits + string.ascii_lowercase + '$')
        moran = MORAN(
            1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True,
            inputDataType='torch.cuda.FloatTensor', CUDA=True
        )
        print(f"Loading pre-trained MORAN model from {moran_config['pretrained']}")
        state_dict = torch.load(moran_config['pretrained'], map_location=self.device)
        moran_state_dict_rename = OrderedDict(
            (k.replace("module.", ""), v) for k, v in state_dict.items()
        )
        moran.load_state_dict(moran_state_dict_rename)
        moran = moran.to(self.device)
        moran.eval()
        return moran

    def CRNN_init(self, crnn_config):
        """
        初始化 CRNN 模型。
        """
        if 'pretrained' not in crnn_config:
            raise ValueError("CRNN configuration is missing the 'pretrained' key.")
        crnn = CRNN(32, 1, 37, 256).to(self.device)
        print(f"Loading pre-trained CRNN model from {crnn_config['pretrained']}")
        state_dict = torch.load(crnn_config['pretrained'], map_location='cpu')

        # 去掉 "module." 前缀
        crnn_state_dict_rename = OrderedDict(
            (k.replace("module.", ""), v) for k, v in state_dict.items()
        )
        crnn.load_state_dict(crnn_state_dict_rename)
        crnn = crnn.to(self.device)
        crnn.eval()
        return crnn

    def ASTER_init(self, aster_config):
        """
        初始化 ASTER 模型。
        """
        if 'pretrained' not in aster_config:
            raise ValueError("ASTER configuration is missing the 'pretrained' key.")
        aster_info = AsterInfo(aster_config['voc_type'])
        aster = RecognizerBuilder(
            arch='ResNet_ASTER',
            rec_num_classes=aster_info.rec_num_classes,
            sDim=512,
            attDim=512,
            max_len_labels=aster_info.max_len,
            eos=aster_info.char2id[aster_info.EOS],
            STN_ON=True
        )
        print(f"Loading pre-trained ASTER model from {aster_config['pretrained']}")
        state_dict = torch.load(aster_config['pretrained'], map_location=self.device)['state_dict']

        # 去掉 "module." 前缀
        aster_state_dict_rename = OrderedDict(
            (k.replace("module.", ""), v) for k, v in state_dict.items()
        )
        aster.load_state_dict(aster_state_dict_rename)
        aster = aster.to(self.device)
        aster.eval()
        return aster

    def parse_moran_data(self, imgs_input):
        """
        预处理图像数据以适配 MORAN 模型。
        """
        batch_size = imgs_input.shape[0]
        imgs_input = F.interpolate(imgs_input, (32, 100), mode='bicubic')
        R, G, B = imgs_input[:, 0:1, :, :], imgs_input[:, 1:2, :, :], imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        text = torch.LongTensor(batch_size * 5)
        length = torch.IntTensor(batch_size)
        max_iter = 20
        t, l = self.converter_moran.encode(['0' * max_iter] * batch_size)
        loadData(text, t)
        loadData(length, l)
        return tensor, length, text, text

    def evaluate(self, dataset, max_samples=None):
        """
        输入数据集，计算整个测试集的文本识别率。
        """
        moran_predictions, crnn_predictions, aster_predictions = [], [], []
        labels = []

        for data in dataset:
            hr_img = data['gt'].unsqueeze(0).to(self.device)  # 添加 batch 维度
            label = data['label']
            labels.append(label)

            # MORAN 预测
            tensor, length, text, _ = self.parse_moran_data(hr_img)
            moran_predictions.append(self.decode_output(self.moran(tensor, length, text, text)))

            # CRNN 预测
            tensor = self.parse_crnn_data(hr_img)
            crnn_predictions.append(self.decode_output(self.crnn(tensor)))

            # ASTER 预测
            input_dict = self.parse_aster_data(hr_img)
            aster_predictions.append(self.decode_output(self.aster(input_dict)))

        # 计算三个模型的准确率
        moran_acc = self.calculate_accuracy(moran_predictions, labels)
        crnn_acc = self.calculate_accuracy(crnn_predictions, labels)
        aster_acc = self.calculate_accuracy(aster_predictions, labels)

        return (moran_acc + crnn_acc + aster_acc) / 3


class AsterInfo(object):
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
