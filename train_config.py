import argparse

def set_train_config():
    parser = argparse.ArgumentParser(description='Train setting for WPR-SRGAN')

    #--------------------------------------------------------
    # model setting
    #--------------------------------------------------------

    # x2 or x4
    parser.add_argument('-srf', default='x4')

    # SRRes, RRDB, or Real-RRDB
    parser.add_argument('-model_type', default='RRDB')

    # not use big batch-size for wpr.
    # Big batch-size will make D_van converge too fast to utilize their information.
    # If you have more gpu, you can use big HR size instead big batch size
    parser.add_argument('-batch_size', default=16)

    #--------------------------------------------------------
    # path setting
    #--------------------------------------------------------
    
    # trainset path
    parser.add_argument('-trainset_path', default='./data/train')

    # pretrained model path
    # if None initial model randomly
    parser.add_argument('-pretrained_G', default='./pretrained/psnr')
    parser.add_argument('-pretrained_Dvan', default=None)
    parser.add_argument('-pretrained_Dins', default=None)

    # valid-set path
    parser.add_argument('-valid_folder', default='./data/valid/bsd100')

    # valid log save path
    parser.add_argument('-train_log', default='./data/train_log')

    # model save path
    parser.add_argument('-model_save_path', default='./data/trained')

    #--------------------------------------------------------
    # parameter setting
    #--------------------------------------------------------
    
    # Total epoch
    parser.add_argument('-total_epoch', default=600)

    # Test spacing. How many itertimes*1000 for test valida-set.
    # test on bsd100 need 15s, DIV2K-Valid need 110s.
    # Change it if your valid set is too large.
    parser.add_argument('-valid_spacing', default=1)   

    # lerning rate
    parser.add_argument('-init_g_lr', default=1e-4)
    parser.add_argument('-init_dvan_lr', default=1e-4)
    parser.add_argument('-init_dins_lr', default=1e-4)
    parser.add_argument('-decay_epoch', default=[50, 100, 200])

    # trade off paramter
    parser.add_argument('-alpha', default=1e-2)

    parser.add_argument('-beta1', default=5e-3)
    parser.add_argument('-beta2', default=5e-3)

    parser.add_argument('-gamma1', default=[0.1,0.1,1.,1.,1.])
    parser.add_argument('-gamma2', default=[0.1,0.1,1.,1.,1.])

    # gan type, vanlila, relative, ls, wgan, wgan-gp
    parser.add_argument('-gan_type', default='relative')

    #--------------------------------------------------------
    # wpr setting
    #--------------------------------------------------------

    # HR size
    parser.add_argument('-hr_size', default=(128,128))

    # replace portion
    parser.add_argument('-replace_portion', default=0.5)

    # replace size
    parser.add_argument('-replace_size', default=(16,16))

    # replace type, random or wpr
    parser.add_argument('-replace_type', default='wpr')

    #--------------------------------------------------------
    args = parser.parse_args()

    return args
