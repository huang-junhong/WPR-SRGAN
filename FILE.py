import os
import cv2
import numpy as np

def mkdir(path):
  
  folder=os.path.exists(path)
  
  if not folder:
    
    os.makedirs(path)
    
    print(path,' Folder Created')
    
  else:
    
    print(path,' Already Exist')

def load_file_path(PATH):
    filenames=[]
    for root,dir,files in os.walk(PATH):
        for file in files:
            if os.path.splitext(file)[1]=='.jpg' or os.path.splitext(file)[1]=='.png' or os.path.splitext(file)[1]=='.bmp' or os.path.splitext(file)[1]=='.tif':
                filenames.append(os.path.join(root,file))
    filenames=sorted(filenames)
    return filenames

def load_img(Paths, Normlize=False, as_array=False, Gray=False, CHW=False):
    imgs=[]
    for i in range(len(Paths)):
        temp=None
        if Gray:
            temp=cv2.imread(Paths[i],0)
            temp=np.expand_dims(temp,2)
        else:
            temp=cv2.imread(Paths[i])
            temp=cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)
        if Normlize == 'A':
            temp = temp.astype('float32')
            temp = temp / 255.
        elif Normlize == 'B':
            temp = temp.astype('float32')
            temp = temp / 127.5 - 1.
        if CHW:
            temp=np.transpose(temp,[2,0,1])

        imgs.append(temp)
    if as_array:
        imgs=np.array(imgs)
    return imgs

def tensor2img(input, recover='A'):

    current = input.squeeze().detach().cpu().numpy()
    if len(current) == 3:
        current = np.transpose(current, [1,2,0])
    if recover == 'A':
        current = np.clip(current*255.,0,255).astype('uint8')

    return current