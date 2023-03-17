# WPR-SRGAN

# Test
Download our [pretrained model](https://drive.google.com/drive/folders/1yMxgA8wpfcHQ1CRUrJyiU__D_tD3fi4K?usp=sharing).

Test on your code
----------------------------------------------------

WPR-SRGAN's Generator Structure is [ESRGAN](https://github.com/xinntao/ESRGAN)

Real-WPRSRGAN's Generator Structure is [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

so you can use thirs code or basicsr to fast test our model.

Test on our code
----------------------------------------------------
If you want test WPR-SRRes or use or our test code you should :

Modify test_config.py

Run test.py

Metric
----------------------------------------------------
The different OS (windows or linux's different version) will effect the final value. (The value will beter than our report in Windows11)\
Except PSNR calculate in matlab-2017b, others calculated in [iqa-pytorch](https://github.com/chaofengc/IQA-PyTorch), with enviorment RTX3090 + Intel gold 6330, pytroch==1.10.0, Ubuntu 20.04 and cuda 11.3.

----------------------------------------------------
# Train

If you want implement by yourself, note follow details:

* 1. Don't use big batch-size like 64. Big batch-szie will make discriminator converge fast, it not a good property for WPR-SRGAN. You can use big HR-size to instead increas the batch-size.
* 2. 
* 3. 

Begin Train:
---------------------------------------------------
1. Change the file path in make_dateset.py and run it. Or use your Date_Loader instead ours.
2. Change train_config.py
3. Run train.py
