
import glob
import os
from utils import prepare_result_dir
import configs
from time import sleep
import sys
import ZSSR
import matplotlib as plt
import os.path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="1"
conf = configs.Config()
res_dir = prepare_result_dir(conf)

files = [file_path for file_path in glob.glob('./set/img/*.png')]
depths = [depth_path for depth_path in glob.glob('./set/dep/*.png')]
files = sorted(files)
depths = sorted(depths)

kernels_ = [kernel_path for kernel_path in glob.glob('./set/ker/*.mat')] # if don't have kernel, annot.
kernels_ = sorted(kernels_) # if don't have kernel, annot.

for file_ind, input_file in enumerate(files):

    conf.result_path = res_dir

    input_depth = depths[file_ind]
    kernels = kernels_[file_ind]  # if don't have kernel, annot.
    kernels = [kernels] # if don't have kernel, annot.


    print('input file : ', input_file)
    print('input depth : ', input_depth)
    print('input kernel : ', kernels)
    net = ZSSR.ZSSR(input_file,input_depth=input_depth,conf=conf,kernels=kernels)  # if don't have kernel, 'change kerenls = None'
    net.run()
