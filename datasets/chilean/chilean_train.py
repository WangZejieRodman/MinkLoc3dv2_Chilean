# Chilean Underground Mine datasets
# Dataset specific training classes

import torchvision.transforms as transforms

from datasets.augmentation import JitterPoints, RemoveRandomPoints, RandomTranslation, RemoveRandomBlock
from datasets.base_datasets import TrainingDataset
from datasets.chilean.chilean_raw import ChileanPointCloudLoader


class ChileanTrainingDataset(TrainingDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_loader = ChileanPointCloudLoader()


class ChileanTrainTransform:
    # 适合Chilean地下巷道数据集的数据增强
    def __init__(self, aug_mode):
        self.aug_mode = aug_mode
        if self.aug_mode == 1:
            # 针对地下巷道环境的增强策略
            # 使用较小的jitter和translation，因为地下空间相对封闭
            t = [JitterPoints(sigma=0.001, clip=0.002),
                 RemoveRandomPoints(r=(0.0, 0.05)),  # 较少的点移除
                 RandomTranslation(max_delta=0.005),  # 较小的平移
                 RemoveRandomBlock(p=0.3)]  # 较少的块移除概率
        elif self.aug_mode == 2:
            # 更保守的增强策略
            t = [JitterPoints(sigma=0.0005, clip=0.001),
                 RemoveRandomPoints(r=(0.0, 0.03)),
                 RandomTranslation(max_delta=0.003)]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e