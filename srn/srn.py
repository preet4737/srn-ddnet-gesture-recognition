import os
import argparse
import torch.nn.parallel
from torch.utils.data import DataLoader

from .module_2d import *
from .dataLoader import dhg
from .generateFeature import GFM
from .config import config


class HandPose(object):
    def __init__(self, config):
        self.model_dir = config.model_dir
        self.train_dataset = config.train_dataset
        self.gpu_id = config.gpu_id

        # net para
        self.G_type = config.G_type
        self.joint_num = config.joint_num

        # load image para
        self.cube_size = config.cube_size
        self.input_size = config.input_size

        # multi_net
        self.feature_type = config.feature_type
        self.feature_sum = config.feature_sum
        self.dim_accumulate = config.dim_accumulate
        self.stage_type = config.stage_type

        self.deconv_size = config.deconv_size

        self.G = multi_stage(config)

    def run(self, data_dir):
        poses = list()
        self.testData = dhg.realtime_loader(data_dir, (615.866, 615.866, 316.584, 228.38),
                                        cube_size=self.cube_size) 

        self.testLoader = DataLoader(self.testData, batch_size=1, shuffle=False, num_workers=1)
        self.G.load_state_dict(torch.load(os.path.join(self.model_dir), map_location=lambda storage, loc: storage))
        self.GFM_ = GFM()
        self.G.eval()

        for batch_idx, data in enumerate(self.testLoader):
            img, center, center_uvd, M, cube = data
            with torch.no_grad():

                outputs, features = self.G(img, self.GFM_, self.testData, M, cube, center)
                for index in range(0, len(outputs), 3):
                    output = outputs[index][0].view(1, -1, 3)
                    joints_xyz = self.testData.uvd_nl2xyznl_tensor(output, M, cube, center)

                    hands172dhg = [0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20]
                    center[0][0] = center[0][0] * -1 + 320
                    center[0][1] += 240
                    cube[0][0] *= -1
                    joints_xyz = \
                    (joints_xyz * cube.view(1, 1, 3) / 2 + center.view(1, 1, 3)).cpu().numpy()[0]
                    poses.append(joints_xyz)
        return poses


if __name__ == '__main__':
    predictor = HandPose(config)
    poses = predictor.run('./data/essai_1')

    print(poses)