
class Config(object):
    model_dir = '/media/dexter/7786037C0F8CEE68/BE_Project/gesture-backend/srn/checkpoint/offset_000_240.pth'
    gpu_id = 0
    train_dataset = 'hand17'

    # net para
    G_type = 'multi_net_resnet18'
    joint_num = 21

    # load image para
    cube_size = [250, 250, 250]
    input_size = 128

    # multi_net
    feature_type = 'offset'
    feature_sum = False
    dim_accumulate = True
    stage_type = [0, 0, 0]
    deconv_size = 64
    heatmap_std = 3.4
    pool_factor = 4
    # feature_name_list = 

config = Config()