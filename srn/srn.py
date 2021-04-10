import os
import argparse
import torch.nn.parallel
import cv2
import torch
from torch.utils.data import DataLoader
import PIL
import pathlib
import numpy as np

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


    def drawpose_on_DHG(self, joints_image: np.ndarray, data_dir: str, batch_idx: int):
        joints_draw = joints_image.copy()
        joints_draw[:,0] = 640 - joints_draw[:,0]
        img = cv2.imread(data_dir + f'{batch_idx}_depth.png', cv2.IMREAD_ANYDEPTH)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = (img - img.min())/(img.max() - img.min()) * 255
        img_dir =  self.draw_dir + str(batch_idx) +  '_depth.png'
        img = draw_pose("msra", img, joints_draw)
        img = img.astype('uint8')
        return img


    def run(self, data_dir: str):
        poses_image = list() # To store image coordinates
        poses_world = list() # To store DHG's world coordinates
        nls = list()
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
                    poses_world.append(self.testData.transform_to_DHG_world(joints_image, paras))

                    # Pose Image generation
                    if True:
                        img = self.drawpose_on_DHG(joints_image, data_dir, batch_idx)
                        nls.append(PIL.Image.fromarray(img))

        # Generate gif                
        gif_dir = pathlib.Path(self.draw_dir)
        name = str(gif_dir / 'action.gif')
        duration = 1000/30
        nls[0].save(
            name,
            append_images=nls[1:],
            save_all=True,
            duration=duration,
            loop=0
        )
        # return DHG World coordinates
        return np.array(poses_world)
    
handpose = HandPose(config)

if __name__ == '__main__':
    predictor = HandPose(config)
    poses = predictor.run('./data/essai_1')

    print(poses)