import os

onnx_path = '/home/ubuntu/ashok/Backtesting/model_files/FINAL_2/FINAL_bn_copied.onnx'
trt_path = '/home/ubuntu/ashok/Backtesting/model_files/FINAL_2/FINAL_bn_copied.trt'
logs_path = './onnx2trt_logs.txt'

os.system(f"""/usr/src/tensorrt/bin/trtexec --onnx={onnx_path} \
            --saveEngine={trt_path}  \
            --verbose  \
            --dumpProfile \
            --separateProfileRun \
            --tacticSources=-CUDNN,-CUBLAS_LT,-CUBLAS \
            --workspace=2048 >{logs_path}""")
