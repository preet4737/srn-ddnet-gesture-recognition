from .loader import *
import glob


class realtime_loader(loader):
    def __init__(self, data_path, paras, img_size=128, cube_size=[300, 300, 300]):
        super(realtime_loader, self).__init__('', 'test', img_size, 'realtime')
        self.data_path = data_path
        self.cube_size = cube_size
        self.paras = paras
        self.flip = 1
        self.frame_len = len(list(glob.iglob(data_path + "/*.png")))

    def __getitem__(self, index):
        img_path = self.data_path + '/'+ f'{index}_depth.png'
        image = cv2.imread(img_path, -1)
        depth = np.asarray(image, dtype=np.float32)

        depth = depth[:, ::-1]
        depth[depth == 0] = depth.max()

        # center_uvd = get_center_fast(depth)
        center_uvd = get_center_adopt(depth)
        center_xyz = self.jointImgTo3D(center_uvd)
        cube_size = self.cube_size

        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, dsize=(self.img_size, self.img_size))

        imgD = self.normalize_img(depth_crop.max(), depth_crop, center_xyz, cube_size)
        curCube = np.array(cube_size)
        com2D = center_uvd
        M = trans

        com3D = self.jointImgTo3D(com2D)

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)
        center = torch.from_numpy(com3D).float()
        center_uvd = torch.from_numpy(center_uvd).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(curCube).float()
        return data, center, center_uvd, M, cube

    def __len__(self):
        return self.frame_len
