from tqdm import tqdm
import os

binary_images_dir = "/home/ubuntu/ashok/Backtesting/samples/images_binary"

#for final_2
trt_outputs_dir = "/home/ubuntu/ashok/Backtesting/trt_outputs/final_2"
engine_path = '/home/ubuntu/ashok/Backtesting/model_files/FINAL_2/FINAL_bn_copied.trt' 
input_name_in_onnx = "input" # this is the name given during onnx conversion. check in onnx visualizer



for file in tqdm(os.listdir(binary_images_dir)):
    inputs_path = f"{input_name_in_onnx}:{os.path.join(binary_images_dir,file)}" #f'images:./binary/{file}'
    output_json_path = os.path.join(trt_outputs_dir, file.split('.')[0] + '.json')

    cmd = f"/usr/src/tensorrt/bin/trtexec \
    --loadEngine={engine_path} \
    --loadInputs={inputs_path} \
    --exportOutput={output_json_path} \
    --avgRuns=1 \
    --iterations=1 \
    --duration=0"

    os.system(cmd)
