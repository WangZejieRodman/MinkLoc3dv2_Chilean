# Warsaw University of Technology
# Train MinkLoc model on Chilean Underground Mine Dataset

import argparse
import torch

from training.trainer_chilean import do_train_chilean
from misc.utils import TrainingParams


if __name__ == '__main__':
    # 直接设置参数，不使用命令行解析
    class Args:
        def __init__(self):
            self.config = '../config/config_chilean.txt'
            #self.model_config = '../models/minkloc3dv1.txt'
            self.model_config = '../models/minkloc3dv2.txt'  # 可选择更强的模型
            self.debug = False

    args = Args()
    print('Training config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print('Debug mode: {}'.format(args.debug))

    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    do_train_chilean(params)