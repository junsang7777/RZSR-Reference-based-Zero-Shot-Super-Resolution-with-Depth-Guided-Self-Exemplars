
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

files = [file_path for file_path in glob.glob('../set/img/*.png')]
#depths = [depth_path for depth_path in glob.glob('./set14_test/*.jpeg')]
#files = [file_path for file_path in glob.glob('../RGB_OUT/*.png')]
depths = [depth_path for depth_path in glob.glob('../set/dep/*.png')]
files = sorted(files)
depths = sorted(depths)

Kernel_switch = False
kernels_ = [kernel_path for kernel_path in glob.glob('../set/ker/*.mat')]
kernels_ = sorted(kernels_)
#gimgs = [g_path for g_path in glob.glob('../RGB/*.png')]
#gimgs = sorted(gimgs)
for file_ind, input_file in enumerate(files):

    conf.result_path = res_dir

    input_depth = depths[file_ind]
    #input_depth = kkk + '/' + y + '_disp.jpeg'
    kernels = kernels_[file_ind]
    kernels = [kernels]
    #g_img = gimgs[file_ind]


    print('input file : ', input_file)
    print('input depth : ', input_depth)
    print('input kernel : ', kernels)
    net = ZSSR.ZSSR(input_file,input_depth=input_depth,conf=conf,kernels=kernels)
    net.run()
