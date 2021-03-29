import os
import argparse
import torch.nn.parallel
import cv2
import torch
from torch.utils.data import DataLoader

from .module_2d import *
from .dataLoader import dhg
from .generateFeature import GFM
from .config import config
from .vis_tool import draw_pose

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

        self.draw_dir = config.draw_dir
        
    def run(self, data_dir):
        poses_image = list() # To store image coordinates
        poses_world = list() # To store DHG's world coordinates
        
        # paras = (615.866, 615.866, 316.584, 228.38) # Copied from realsense dataset
        paras = (440.44232, 461.0357, -0.00015258789, 3.0517578e-05) # 
    
        hands172dhg = [0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20]

        self.testData = dhg.realtime_loader(data_dir, paras, cube_size=self.cube_size) 

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
                    
                    # Transform out of cube
                    joints_xyz = joints_xyz * cube.view(1, 1, 3) / 2 + center.view(1, 1, 3)
            
                    # image coordinates
                    joints_image = self.testData.joints3DToImg(joints_xyz).cpu().numpy()[0][hands172dhg, :]
                    
                    # DHG's world coordinates
                    joints_world = joints_image.copy()
                    joints_world[:, 2] = joints_world[:, 2] * 0.00093 
                    joints_world[:, 0] = ((640 - joints_world[:, 0]) - paras[2]) * joints_world[:, 2] / paras[0]
                    joints_world[:, 1] = ((joints_world[:, 1]) - paras[3]) * joints_world[:, 2] / -paras[1]
                    poses_world.append(joints_world)

                    # Pose Image generation
                    if False:
                        img_draw = (img.cpu().numpy() + 1) / 2 * 255
                        joints_draw = (output.detach().cpu().numpy() + 1) * (self.input_size / 2)
                        img_dir =  self.draw_dir + str(batch_idx) +  '_depth.png'
                        img_show = draw_pose("hands17", cv2.cvtColor(img_draw[0, 0], cv2.COLOR_GRAY2RGB),
                                             joints_draw[0])
                        cv2.imwrite(img_dir, img_show)

        # return world coordinates
        return np.array(poses_world)
    
handpose = HandPose(config)

if __name__ == '__main__':
    predictor = HandPose(config)
    poses = predictor.run('./data/essai_1')

    print(poses)