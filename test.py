import cv2
import numpy as np
import torch
import torch.nn as nn 
from tqdm import tqdm

import test_config
import FILE
import Nets
import RRDB
import Real_ESRGAN


if __name__ == "__main__":
    test_cig = test_config.set_test_config()
    print('WPR-SRGAN Test Parameters Setting:')
    print('Model: {}'.format(test_cig.model_type))
    print('Scale Factor: {}'.format(test_cig.srf))

    if test_cig.model_type == 'SRRes':
        G = Nets.SRRes()
    elif test_cig.model_type == 'RRDB':
        G = RRDB.RRDBNet(3,3,64,23)
    elif test_cig.model_type == 'Real-RRDB':
        G = Real_ESRGAN.RRDBNet(3,3,test_cig.srf)

    G.load_state_dict(torch.load(test_cig.model_path, map_location='cpu'))
    G.cuda().eval()
    print('Model Load Complete')

    lrs = FILE.load_img(FILE.load_file_path(test_cig.lr_path), Normlize='A', CHW=True)
    FILE.mkdir(test_cig.save_path)

    for i in tqdm(range(len(lrs))):
        lr = np.expand_dims(lrs[i], 0)
        lr = torch.Tensor(lr).cuda()
        sr = G(lr)
        sr = FILE.tensor2img(sr)

        cv2.imwrite(test_cig.save_path+'/'+str(i+1).zfill(3)+'.png', cv2.cvtColor(sr, cv2.COLOR_BGR2RGB))

    print('Reconstruct Complete')
