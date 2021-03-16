from scipy.ndimage import interpolation as inter
from scipy.signal import medfilt
import numpy as np
import random


def zoom(p: np.array, target_l=30):
    """ Resize framerate to desired length """
    l = p.shape[0]
    p_new = np.empty([target_l, 21, 3])
    for m in range(21):
        for n in range(3):
            p[:, m, n] = medfilt(p[:, m, n], 3)
            p_new[:, m, n] = inter.zoom(p[:, m, n], target_l / l)[:target_l]
    return p_new


def sampling_frame(p: list):
    """ I don't know what this does """
    p = np.array(p)
    p = np.reshape(p, (-1, 21, 3))
    full_l = p.shape[0]  # full length
    if random.uniform(0, 1) < 0.5:  # aligment sampling
        valid_l = np.round(np.random.uniform(0.9, 1) * full_l)
        s = random.randint(0, full_l - int(valid_l))
        e = s + valid_l  # sample end point
        p = p[int(s) : int(e), :, :]
    else:  # without aligment sampling
        valid_l = np.round(np.random.uniform(0.9, 1) * full_l)
        index = np.sort(np.random.choice(range(0, full_l), int(valid_l), replace=False))
        p = p[index, :, :]
    p = zoom(p)
    return p
