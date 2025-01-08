import logging
import torch
from os import path as osp
import sys

# 确保你的 PYTHONPATH 包含了 BasicSR 的根目录
# sys.path.append('/data1/guohang/MambaIR-main')

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
from basicsr.metrics.text_recognition_metric import TextRecognitionMetric


def test_pipeline(root_path):
    # 解析配置选项，设置分布式参数，设置随机种子
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # 创建目录并初始化日志记录器
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # 创建测试数据集和数据加载器
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed']
        )
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # 创建模型
    model = build_model(opt)
    model.eval()  # 确保模型处于评估模式

    # 初始化 TextRecognitionMetric
    text_recognition_metric = TextRecognitionMetric(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')

        for data in test_loader:
            # 假设数据集中包含 'lq'（低分辨率图像）、'gt'（高分辨率图像）、'label'（文本标签）
            lq_images = data['lq'].to(text_recognition_metric.device)  # [batch, C, H, W]
            labels = data['label']  # list of str

            # 使用超分辨率模型生成 SR 图像
            with torch.no_grad():
                # 根据你的模型定义，这里假设 model.forward 返回 SR 图像
                sr_images = model(lq_images)  # [batch, C, H, W]

            # 更新 Metrics
            text_recognition_metric.update(sr_images, labels)

    # 获取并记录最终的文本识别率
    metrics = text_recognition_metric.get_metric(reset=True)
    logger.info(f"Text Recognition Metrics: {metrics}")


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
