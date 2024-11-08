from tqdm import tqdm
import os
import numpy as np 
import cv2

def pad_resize(im):
    im = cv2.resize(im, (320, 360))
    extra = np.zeros((24, 320, 3))
    final = np.concatenate((im, extra), axis=0)
    return final

def preprocess(image_path):
    """Preprocess image for yolo (resize, pad, channel transpose)"""
    im0 = cv2.imread(image_path)  # BGR
    im0 = im0[:, int(im0.shape[1]/2):, :]
    img = pad_resize(im0)
    img1 = img.copy()
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img) / 255.0
    return img[None, :].astype(np.float32), img1



#for new model 
BASE = "/home/ubuntu/ashok/Backtesting"
images_dir = f"{BASE}/samples"
binaries_dir = f"{BASE}/samples_binary"




#change to binary
for image in tqdm(os.listdir(images_dir)):
    binary_path = os.path.join(binaries_dir,image.split('.')[0])
    with open(binary_path, 'wb') as fp:
        im,_ = preprocess(os.path.join(images_dir,image))
        fp.write(im.astype(np.float32))

