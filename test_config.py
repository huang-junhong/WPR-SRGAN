import argparse

def set_test_config():
    parser = argparse.ArgumentParser(description='Test setting for WPR-SRGAN')
 
    #--------------------------------------------------------
    # path setting
    #--------------------------------------------------------
    parser.add_argument('-lr_path', '--LR_PATH', default='./data/test/lr')
    parser.add_argument('-save_path', '--SAVE_PATH', default='./data/test/results')
    parser.add_argument('-model_path', '--Model_PATH', default='./pretrained/wpr')

    #--------------------------------------------------------
    # model setting
    #--------------------------------------------------------

    # upsacale factor and model type for load model
    # x2 or x4
    parser.add_argument('-srf', '--SRF', default='x4')

    # SRRes, RRDB, or Real-RRDB
    parser.add_argument('-model_type', '--Model_Type', default='RRDB')

    args = parser.parse_args()

    return args