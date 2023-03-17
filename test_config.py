import argparse

def set_test_config():
    parser = argparse.ArgumentParser(description='Test setting for WPR-SRGAN')
 
    #--------------------------------------------------------
    # path setting
    #--------------------------------------------------------
    parser.add_argument('-lr_path', default='./data/test/lr')
    parser.add_argument('-save_path', default='./data/test/results')
    parser.add_argument('-model_path', default='./pretrained/wpr')

    #--------------------------------------------------------
    # model setting
    #--------------------------------------------------------

    # upsacale factor and model type for load model
    # x2 or x4
    parser.add_argument('-srf', default='x4')

    # SRRes, RRDB, or Real-RRDB
    parser.add_argument('-model_type', default='RRDB')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    train_opt = set_test_config()

    print(train_opt.lr_path)