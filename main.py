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
folder_path = f"{BASE}/samples"
binary_images_dir = f"{BASE}/samples_binary"



if False:
    #change to binary
    for image in tqdm(os.listdir(folder_path)):
        binary_path = os.path.join(binary_images_dir,image.split('.')[0])
        with open(binary_path, 'wb') as fp:
            # im, _ = preprocess('./frames/9d0a2b91-794d-4c8e-b835-668cf1848f55/{}'.format(image))
            im,_ = preprocess(os.path.join(folder_path,image))
            fp.write(im.astype(np.float32))


# inference on trt

# #for new model
# trt_outputs_dir = "/home/ubuntu/ashok/Backtesting/trt_outputs"
# engine_path = '/home/ubuntu/ashok/Backtesting/model_files/FINAL_v5_3_epoch_7.trt' 
# input_name_in_onnx = "input" # this is the name given during onnx conversion. check in onnx visualizer


# #for baseline model
# trt_outputs_dir = "/home/ubuntu/ashok/Backtesting/trt_outputs_baseline"
# engine_path = '/home/ubuntu/ashok/Backtesting/model_files/baseline_trtmodel8_B2' 
# input_name_in_onnx = "data" # this is the name given during onnx conversion. check in onnx visualizer


#for final_2
trt_outputs_dir = "/home/ubuntu/ashok/Backtesting/trt_outputs/final_2"
engine_path = '/home/ubuntu/ashok/Backtesting/model_files/FINAL_2/FINAL_bn_copied.trt' 
input_name_in_onnx = "input" # this is the name given during onnx conversion. check in onnx visualizer



for file in tqdm(os.listdir(binary_images_dir)):
    inputs_path = f"{input_name_in_onnx}:{os.path.join(binary_images_dir,file)}" #f'images:./binary/{file}'
    output_json_path = os.path.join(trt_outputs_dir, file.split('.')[0] + '.json') #f'./trt_output_pose/{file.split('.')[0]}.json'

    cmd = f"/usr/src/tensorrt/bin/trtexec \
    --loadEngine={engine_path} \
    --loadInputs={inputs_path} \
    --exportOutput={output_json_path} \
    --avgRuns=1 \
    --iterations=1 \
    --duration=0"

    os.system(cmd)


