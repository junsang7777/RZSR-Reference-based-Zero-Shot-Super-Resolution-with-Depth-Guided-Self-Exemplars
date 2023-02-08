import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.gridspec import GridSpec
from configs import Config
from utils import *
#from simplenet import simpleNet
import cv2
from simplenet_0108 import simpleNet_patch, simpleNet, RZSR
#from simplenet_1102 import MINE
import js_utill_0217
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="1"


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def Add_jpg_compression(img):
    quality_std = np.random.randint(6)
    img = img*255.0
    img = np.around(img)
    img = img.clip(0,255)
    img=img.astype(np.uint8)

    # encode param image quality 0 to 100. default:95
    # if you want to shrink data size, choose low image quality.
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 84 + quality_std]

    result, encimg = cv2.imencode('a.jpg', img, encode_param)
    if False==result:
        print('fuck')
        quit()
    decimg = cv2.imdecode(encimg, 1)
    decimg = decimg.astype(np.float32)
    decimg /= 255.
    return decimg
def distance(center_x,center_y, x,y, patch_size):
    dist = np.sqrt((x-center_x)**2 + (y-center_y)**2)
    max_dist = np.sqrt((patch_size/2)**2+(patch_size/2)**2)
    return dist / max_dist
def index_rot(y,x, shape_y, shape_x,rotation_var): # y,x (200, 100), y,x (25,39)
    if rotation_var==0:
        return y, x
    elif rotation_var==1:
        return x, shape_y-y-1
    elif rotation_var == 2:
        return shape_y-y-1, shape_x-x-1
    elif rotation_var == 3:
        return shape_x-x-1, y
    elif rotation_var == 4:
        return y, shape_x-x-1
    elif rotation_var == 5:
        return x, y
    elif rotation_var == 6:
        return shape_y-y-1, x
    elif rotation_var == 7:
        return shape_x-x-1, shape_y-y-1


class ZSSR:
    # Basic current state variables initialization / declaration
    kernel = None
    learning_rate = None
    hr_father = None
    lr_son = None
    sr = None
    sf = None
    gt_per_sf = None
    final_sr = None
    hr_fathers_sources = []

    # Output variables initialization / declaration
    reconstruct_output = None
    train_output = None
    output_shape = None

    # Counters and logs initialization
    iter = 0
    base_sf = 1.0
    base_ind = 0
    sf_ind = 0
    mse = []
    mse_rec = []
    interp_rec_mse = []
    interp_mse = []
    mse_steps = []
    loss = []
    learning_rate_change_iter_nums = []
    fig = None

    # Network tensors (all tensors end with _t to distinguish)
    learning_rate_t = None
    lr_son_t = None
    hr_father_t = None
    filters_t = None
    layers_t = None
    net_output_t = None
    loss_t = None
    train_op = None
    init_op = None

    # Parameters related to plotting and graphics
    plots = None
    loss_plot_space = None
    lr_son_image_space = None
    hr_father_image_space = None
    out_image_space = None

    def __init__(self, input_img, input_depth=None, conf=Config(), ground_truth=None, kernels=None, ground_img=None):
        # Acquire meta parameters configuration from configuration class as a class variable
        self.conf = conf
        self.cuda = conf.cuda
        # Read input image (can be either a numpy array or a path to an image file)
        # self.input = input_img if type(input_img) is not str else img.imread(input_img)
        self.Reference = conf.reference
        if input_depth == None:
            self.Reference = False

        self.Reference = True
        self.str_input_img = str(input_img)
        self.input = input_img if type(input_img) is not str else ((cv2.cvtColor(
            (cv2.imread(input_img, cv2.IMREAD_COLOR)), cv2.COLOR_BGR2RGB)).astype(np.float32)) / 255.
        self.hhh = self.input.shape[0]
        self.www = self.input.shape[1]
        if input_depth != None :
            self.depth = input_depth if type(input_depth) is not str else(cv2.imread(input_depth, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.)

        self.shift = []
        ###########################
        fake_width = self.www
        fake_height = self.hhh
        while(1):
            if fake_height % 16 == 0 and fake_width % 16 == 0:
                break
            elif fake_height % 16 != 0:
                fake_height += 1
                #print('h')
            elif fake_width % 16 != 0 :
                fake_width += 1
                #print('w')
        self.fake_input = np.zeros((fake_height,fake_width,3),dtype=np.float64)
        self.fake_input[:self.hhh,:self.www,:] = self.input
        self.fake_depth = np.zeros((fake_height,fake_width),dtype=np.float64)
        self.fake_depth[:self.hhh,:self.www] = self.depth

        ###########################
        self.Y = False
        if len(self.input) == 2: # is it gray? (never)
            self.Y = True
        # input is ndarray
        # For evaluation purposes, ground-truth image can be supplied.
        self.gt = ground_truth if type(ground_truth) is not str else img.imread(ground_truth)
        # gt is ndarray
        # Preprocess the kernels. (see function to see what in includes).
        self.kernels = preprocess_kernels(kernels, conf)

        ####################
        """
        ker = self.kernels[0]
        ker[ker<0.05] = 0
        print(np.sum(ker))
        ker = ker/((np.sum(ker))+0.0000001)
        self.kernels = [ker]
        """
        #######################

        # downsample kernel custom
        # Prepare TF default computational graph
        # declare model here severs as initial model
        print(self.Y)
        if self.Reference == True:
            self.model = RZSR(self.Y)
        else :
            self.model = simpleNet(self.Y)
        # Build network computational graph
        # self.build_network(conf)

        # Initialize network weights and meta parameters
        self.init_parameters()

        # The first hr father source is the input (source goes through augmentation to become a father)
        # Later on, if we use gradual sr increments, results for intermediate scales will be added as sources.
        self.hr_fathers_sources = [self.fake_input]

        # We keep the input file name to save the output with a similar name. If array was given rather than path
        # then we use default provided by the configs
        self.file_name = input_img if type(input_img) is str else conf.name


        patch_construct_start = time.time()

        if input_depth != None and self.Reference == True:
            js_depth = cv2.imread(input_depth,cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
            #js_depth = 255-js_depth
            js_input = cv2.imread(input_img, cv2.IMREAD_COLOR).astype(np.float32)/255.
            js_fake_height = js_depth.shape[0]
            js_fake_width = js_depth.shape[1]
            while(1):
                if js_fake_height % 16 == 0 and js_fake_width % 16 == 0:
                    break
                elif js_fake_width % 16 != 0 :
                    js_fake_width += 1
                elif js_fake_height % 16 != 0 :
                    js_fake_height += 1
            js_fake_depth = np.zeros((js_fake_height,js_fake_width))
            js_fake_input = np.zeros((js_fake_height,js_fake_width,3))
            js_fake_depth[0:js_depth.shape[0],0:js_depth.shape[1]] = js_depth
            js_fake_input[0:js_depth.shape[0],0:js_depth.shape[1],:] = js_input
            js_fake_depth = (js_fake_depth*255).astype(np.uint8)
            js_fake_input = (js_fake_input*255).astype(np.uint8)
        self.js_PM = js_utill_0217.Patchmatch_VGG_MATCHING(js_fake_depth,js_fake_input,crop_size=self.conf.crop_size,scale_factor=self.conf.scale_factors)
        patch_construct_end = time.time()
        print('patch_construct end.... time : %.2f' % (patch_construct_end - patch_construct_start))
        self.patch_construct_time = patch_construct_end-patch_construct_start

    def run(self):
        # Run gradually on all scale factors (if only one jump then this loop only happens once)
        for self.sf_ind, (sf, self.kernel) in enumerate(zip(self.conf.scale_factors, self.kernels)):
            # verbose
            print('** Start training for sf=', sf, ' **')

            # Relative_sf (used when base change is enabled. this is when input is the output of some previous scale)
            # safe
            if np.isscalar(sf):
                sf = [sf, sf]
            self.sf = np.array(sf) / np.array(self.base_sf)
            self.output_shape = np.uint(np.ceil(np.array(self.fake_input.shape[0:2]) * sf))
            print('input shape', self.fake_input.shape)
            #print('input depth shape', self.depth.shape)
            # Initialize network
            # reinit all for each scale factors, each gradual level
            self.init_parameters()
            if self.conf.init_net_for_each_sf == True:
                if self.Reference == True:
                    self.model = RZSR(self.Y)
                else :
                    self.model = simpleNet(self.Y)
            if self.cuda:
                self.model = self.model.to(device)

            # Train the network
            # should be modified
            print('train start')
            train_start = time.time()
            self.train_MINE()
            train_end = time.time()
            print('train end.... time : %.2f'%(train_end-train_start))
            self.train_time = train_end-train_start
            # Use augmented outputs and back projection to enhance result. Also save the result.

            test_start = time.time()
            post_processed_output = self.final_test()
            test_end = time.time()
            print('test complete.... time : %.2f'%(test_end-test_start))
            self.test_time = test_end-test_start

            # Keep the results for the next scale factors SR to use as dataset
            test_start = time.time()
            self.hr_fathers_sources.append(post_processed_output)

            # In some cases, the current output becomes the new input. If indicated and if this is the right scale to
            # become the new base input. all of these conditions are checked inside the function.
            self.base_change()

            post_processed_output = post_processed_output[:self.conf.scale_factors[0][1]*self.hhh,:self.conf.scale_factors[0][0]*self.www,:]

            # Save the final output if indicated
            if self.conf.save_results:
                sf_str = ''.join('X%.2f' % s for s in self.conf.scale_factors[self.sf_ind])
                plt.imsave('%s/%s_RZSR_%s.png' %
                           (self.conf.result_path, os.path.basename(self.file_name)[:-4], sf_str),
                           post_processed_output, vmin=0, vmax=1)

            # verbose
            #f = open("./time_text_urban100_bicubic_32x32.txt",'r')
            #data = f.read()
            #f.close()
            #f = open("./time_text_urban100_bicubic_32x32.txt",'w')
            #data = data + str(self.str_input_img) + '\t' + str(self.patch_construct_time) + '\t' + str(self.train_time) + '\t' + str(self.test_time) + '\n'
            #f.write(data)
            #f.close()
            print('** Done training for sf=', sf, ' **')

        # Return the final post processed output.
        # noinspection PyUnboundLocalVariable
        return post_processed_output

    def init_parameters(self):
        # Sometimes we only want to initialize some meta-params but keep the weights as they were
        # no need to init weight, done as model declaration
        # Initialize all counters etc
        # no need to change. For record here
        self.loss = [None] * self.conf.max_iters
        self.mse, self.mse_rec, self.interp_mse, self.interp_rec_mse, self.mse_steps = [], [], [], [], []
        self.iter = 0
        self.learning_rate = self.conf.learning_rate
        self.learning_rate_change_iter_nums = [0]

        # Downscale ground-truth to the intermediate sf size (for gradual SR).
        # This only happens if there exists ground-truth and sf is not the last one (or too close to it).
        # We use imresize with both scale and output-size, see comment in forward_backward_pass.
        # noinspection PyTypeChecker
        # scale_factor[-1] means the ground truth
        # hence sf/conf.scale_factors[-1] = 1.5/2, the target size for gt.
        if self.gt_per_sf != None:
            self.gt_per_sf = (imresize(self.gt,
                                       scale_factor=self.sf / self.conf.scale_factors[-1],
                                       output_shape=self.output_shape,
                                       kernel=self.conf.downscale_gt_method)
                              if (self.gt is not None and
                                  self.sf is not None and
                                  np.any(np.abs(self.sf - self.conf.scale_factors[-1]) > 0.01))
                            else self.gt)

    def forward_backward_pass(self, lr_son, hr_father, criterion, optimizer):
        # First gate for the lr-son into the network is interpolation to the size of the father
        # Note: we specify both output_size and scale_factor. best explained by example: say father size is 9 and sf=2,
        # small_son size is 4. if we upscale by sf=2 we get wrong size, if we upscale to size 9 we get wrong sf.
        # The current imresize implementation supports specifying both.
        interpolated_lr_son = imresize(lr_son, self.sf, hr_father.shape, self.conf.upscale_method)
        if self.Y == True:
            lr_son_input = torch.Tensor(interpolated_lr_son).unsqueeze_(0).unsqueeze_(0)
            hr_father = torch.Tensor(hr_father).unsqueeze_(0).unsqueeze_(0)
        else:
            lr_son_input = torch.Tensor(interpolated_lr_son).permute(2, 0, 1).unsqueeze_(0)
            hr_father = torch.Tensor(hr_father).permute(2, 0, 1).unsqueeze_(0)
        lr_son_input = lr_son_input.requires_grad_()

        if self.cuda == True:
            hr_father = hr_father.cuda()
            lr_son_input = lr_son_input.cuda()

        train_output = self.model(lr_son_input)
        loss = criterion(hr_father, train_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.loss[self.iter] = loss

        return np.clip(np.squeeze(train_output.cpu().detach().numpy()), 0, 1)

    def forward_backward_pass_patch(self, lr_son, hr_father_ori, criterion, optimizer, hr_cousin_ori=None):
        # First gate for the lr-son into the network is interpolation to the size of the father
        # Note: we specify both output_size and scale_factor. best explained by example: say father size is 9 and sf=2,
        # small_son size is 4. if we upscale by sf=2 we get wrong size, if we upscale to size 9 we get wrong sf.
        # The current imresize implementation supports specifying both.
        hr_father = hr_father_ori.copy()
        if hr_cousin_ori is not None:
            hr_cousin = hr_cousin_ori.copy()
        interpolated_lr_son = imresize(lr_son, self.sf, hr_father.shape, self.conf.upscale_method)
        if self.Y == True:
            lr_son_input = torch.Tensor(interpolated_lr_son).unsqueeze_(0).unsqueeze_(0)
            hr_father = torch.Tensor(hr_father).unsqueeze_(0).unsqueeze_(0)
        else:
            lr_son_input = torch.Tensor(interpolated_lr_son).permute(2, 0, 1).unsqueeze_(0)
            hr_father = torch.Tensor(hr_father).permute(2, 0, 1).unsqueeze_(0)
            hr_cousin = torch.Tensor(hr_cousin).permute(2, 0, 1).unsqueeze_(0) ###########
        lr_son_input = lr_son_input.requires_grad_()

        if self.cuda == True:
            hr_father = hr_father.cuda()
            lr_son_input = lr_son_input.cuda()
            hr_cousin = hr_cousin.cuda()

        train_output = self.model(lr_son_input, hr_cousin)

        loss = criterion(hr_father, train_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.loss[self.iter] = loss

        return np.clip(np.squeeze(train_output.cpu().detach().numpy()), 0, 1)

    def forward_pass(self, lr_son, hr_father_shape=None):
        # First gate for the lr-son into the network is interpolation to the size of the father
        interpolated_lr_son = imresize(lr_son, self.sf, hr_father_shape, self.conf.upscale_method)
        if self.Y == True:
            interpolated_lr_son = (torch.Tensor(interpolated_lr_son)).unsqueeze_(0).unsqueeze_(0)
        else:
            interpolated_lr_son = (torch.Tensor(interpolated_lr_son).permute(2, 0, 1)).unsqueeze_(0)
        if self.cuda:
            interpolated_lr_son = interpolated_lr_son.cuda()
        # Create feed dict

        # Run network
        return np.clip(np.squeeze(self.model(interpolated_lr_son).cpu().detach().permute(0, 2, 3, 1).numpy()), 0, 1)

    ###################
    def forward_pass_ver21(self, lr_son, hr_father_shape=None, final_test_switch=True):
        [M, N, _] = lr_son.shape
        output = np.zeros((int(self.conf.scale_factors[0][1] * M), int(self.conf.scale_factors[0][0] * N), 3),dtype=np.float64)
        divide = np.zeros((int(self.conf.scale_factors[0][1] * M), int(self.conf.scale_factors[0][0] * N), 3),dtype=np.float64)
        ones = np.ones((self.conf.crop_size, self.conf.crop_size, 3))
        for i in range(0, M, int(self.conf.crop_size/(2*self.conf.scale_factors[0][1]))-4):
            for j in range(0, N, int(self.conf.crop_size/(2*self.conf.scale_factors[0][1]))-4):
                if i + int(self.conf.crop_size/self.conf.scale_factors[0][1]) > M and j + int(self.conf.crop_size/self.conf.scale_factors[0][0]) > N:
                    k = lr_son[M - int(self.conf.crop_size/self.conf.scale_factors[0][1]):M, N - int(self.conf.crop_size/self.conf.scale_factors[0][0]):N]
                    yyyyy = int(M - int(self.conf.crop_size/(2*self.conf.scale_factors[0][1])))
                    xxxxx = int(N - int(self.conf.crop_size/(2*self.conf.scale_factors[0][1])))
                elif i + int(self.conf.crop_size/self.conf.scale_factors[0][0]) > M:
                    k = lr_son[M - int(self.conf.crop_size/self.conf.scale_factors[0][0]):M, j:j + int(self.conf.crop_size/self.conf.scale_factors[0][1])]
                    yyyyy = int(M - int(self.conf.crop_size/(2*self.conf.scale_factors[0][1])))
                    xxxxx = int(j)
                elif j + int(self.conf.crop_size/self.conf.scale_factors[0][1]) > N:
                    k = lr_son[i:i + int(self.conf.crop_size/self.conf.scale_factors[0][1]), N - int(self.conf.crop_size/self.conf.scale_factors[0][0]):N]
                    yyyyy = int(i)
                    xxxxx = int(N - int(self.conf.crop_size/(2*self.conf.scale_factors[0][1])))
                else:
                    k = lr_son[i:i + int(self.conf.crop_size/self.conf.scale_factors[0][0]), j:j + int(self.conf.crop_size/self.conf.scale_factors[0][1])]
                    yyyyy = int(i)
                    xxxxx = int(j)
                width_idx, height_idx, HARD_SWITCH = self.js_PM.test_run(xxxxx, yyyyy)
                if HARD_SWITCH == False:
                    ending_hegiht = int(height_idx + int(self.conf.crop_size/2))
                    ending_width = int(width_idx + int(self.conf.crop_size/2))
                    if height_idx + int(self.conf.crop_size/self.conf.scale_factors[0][1]) > lr_son.shape[0]:
                        ending_hegiht = lr_son.shape[0]
                    if width_idx + int(self.conf.crop_size/self.conf.scale_factors[0][0]) > lr_son.shape[1]:
                        ending_width = lr_son.shape[1]


                    if ending_width-self.conf.crop_size < 0 and ending_hegiht - self.conf.crop_size < 0 :
                        patch = lr_son[0:self.conf.crop_size,0:self.conf.crop_size]
                    elif ending_width-self.conf.crop_size >= 0 and ending_hegiht - self.conf.crop_size < 0 :
                        patch = lr_son[0:self.conf.crop_size,ending_width-self.conf.crop_size:ending_width]
                    elif ending_width - self.conf.crop_size < 0 and ending_hegiht - self.conf.crop_size >= 0:
                        patch = lr_son[ending_hegiht-self.conf.crop_size:ending_hegiht,0:self.conf.crop_size]
                    else :
                        patch = lr_son[ending_hegiht-self.conf.crop_size:ending_hegiht,ending_width-self.conf.crop_size:ending_width]
                    #patch = lr_son[ending_hegiht - 128:ending_hegiht, ending_width - 128:ending_width]
                else :
                    patch = imresize(k, scale_factor=self.sf, kernel=self.conf.upscale_method)
                #patch = imresize(patch, scale_factor=1, kernel='cubic')
                hr_cousin = patch

                interpolated_lr_son = imresize(k, self.sf, (self.conf.crop_size, self.conf.crop_size), self.conf.upscale_method)
                if final_test_switch == True:
                    outputs = []
                    for r in range(0, 1+7*self.conf.output_flip, 1):
                        test_input = np.rot90(interpolated_lr_son, r) if r < 4 else np.fliplr(np.rot90(interpolated_lr_son, r))
                        test_reference = np.rot90(hr_cousin, r) if r < 4 else np.fliplr(np.rot90(hr_cousin, r))
                        test_input = torch.Tensor(test_input.copy()).permute(2, 0, 1).unsqueeze_(0)
                        test_input = test_input.cuda()
                        test_reference = torch.Tensor(test_reference.copy()).permute(2, 0, 1).unsqueeze_(0)
                        test_reference = test_reference.cuda()
                        tmp_output = np.clip(
                            np.squeeze(self.model(test_input, test_reference).cpu().detach().permute(0, 2, 3, 1).numpy()), 0, 1)
                        tmp_output = np.rot90(tmp_output, -r) if r < 4 else np.rot90(np.fliplr(tmp_output), -r)
                        for bp_iter in range(self.conf.back_projection_iters[self.sf_ind]):
                            tmp_output = back_projection(tmp_output, k, down_kernel=self.kernel,
                                                         up_kernel=self.conf.upscale_method, sf=self.sf)
                        outputs.append(tmp_output)
                    almost_final_sr = np.median(outputs, 0)

                    xxx = almost_final_sr
                else:
                    test_input = interpolated_lr_son
                    test_reference = hr_cousin
                    test_input = torch.Tensor(test_input).permute(2, 0, 1).unsqueeze_(0)
                    test_input = test_input.cuda()
                    test_reference = torch.Tensor(test_reference).permute(2, 0, 1).unsqueeze_(0)
                    test_reference = test_reference.cuda()
                    xxx = np.clip(np.squeeze(self.model(test_input, test_reference).cpu().detach().permute(0, 2, 3, 1).numpy()), 0, 1)
                #plt.imshow(xxx)
                #plt.show()
                divide[int(self.conf.scale_factors[0][1] * yyyyy):int(self.conf.scale_factors[0][1] * yyyyy) + self.conf.crop_size, int(self.conf.scale_factors[0][0] * xxxxx):int(self.conf.scale_factors[0][0] * xxxxx) + self.conf.crop_size] += ones
                output[int(self.conf.scale_factors[0][1] * yyyyy):int(self.conf.scale_factors[0][1] * yyyyy) + self.conf.crop_size, int(self.conf.scale_factors[0][0] * xxxxx):int(self.conf.scale_factors[0][0] * xxxxx) + self.conf.crop_size] += xxx
                ###
        #print(divide)
        output = np.divide(output, divide)
        #plt.imshow(output)
        #plt.show()

        return output


    def forward_pass_ver0517(self, lr_son, hr_father_shape=None, final_test_switch=True, rotation_Var=0):
        [M, N, _] = lr_son.shape
        output = np.zeros((int(self.conf.scale_factors[0][1] * M), int(self.conf.scale_factors[0][0] * N), 3),dtype=np.float64)
        divide = np.zeros((int(self.conf.scale_factors[0][1] * M), int(self.conf.scale_factors[0][0] * N), 3),dtype=np.float64)
        ones = np.ones((self.conf.crop_size, self.conf.crop_size, 3))
        for i in range(0, M, int(self.conf.crop_size/(4*self.conf.scale_factors[0][1]))):
            for j in range(0, N, int(self.conf.crop_size/(4*self.conf.scale_factors[0][1]))):
                if i + int(self.conf.crop_size/self.conf.scale_factors[0][1]) > M and j + int(self.conf.crop_size/self.conf.scale_factors[0][0]) > N:
                    k = lr_son[M - int(self.conf.crop_size/self.conf.scale_factors[0][1]):M, N - int(self.conf.crop_size/self.conf.scale_factors[0][0]):N]
                    yyyyy = int(M - int(self.conf.crop_size/self.conf.scale_factors[0][1]))
                    xxxxx = int(N - int(self.conf.crop_size/self.conf.scale_factors[0][0]))
                elif i + int(self.conf.crop_size/self.conf.scale_factors[0][0]) > M:
                    k = lr_son[M - int(self.conf.crop_size/self.conf.scale_factors[0][0]):M, j:j + int(self.conf.crop_size/self.conf.scale_factors[0][1])]
                    yyyyy = int(M - int(self.conf.crop_size/self.conf.scale_factors[0][0]))
                    xxxxx = int(j)
                elif j + int(self.conf.crop_size/self.conf.scale_factors[0][1]) > N:
                    k = lr_son[i:i + int(self.conf.crop_size/self.conf.scale_factors[0][1]), N - int(self.conf.crop_size/self.conf.scale_factors[0][0]):N]
                    yyyyy = int(i)
                    xxxxx = int(N - int(self.conf.crop_size/self.conf.scale_factors[0][1]))
                else:
                    k = lr_son[i:i + int(self.conf.crop_size/self.conf.scale_factors[0][0]), j:j + int(self.conf.crop_size/self.conf.scale_factors[0][1])]
                    yyyyy = int(i)
                    xxxxx = int(j)
                #rotation_y, rotation_x = index_rot(yyyyy,xxxxx,M,N,rotation_Var)
                #if rotation_y +128 >= self.input.shape[0] :
                #    rotation_y = rotation_y-128
                #if rotation_x + 128 >= self.input.shape[1]:
                #    rotation_x = rotation_x-128
                width_idx, height_idx, HARD_SWITCH = self.js_PM.test_run(xxxxx, yyyyy)
                if HARD_SWITCH == False:
                    ending_hegiht = int(height_idx + int(self.conf.crop_size/2))
                    ending_width = int(width_idx + int(self.conf.crop_size/2))
                    if height_idx + int(self.conf.crop_size/self.conf.scale_factors[0][1]) > lr_son.shape[0]:
                        ending_hegiht = lr_son.shape[0]
                    if width_idx + int(self.conf.crop_size/self.conf.scale_factors[0][0]) > lr_son.shape[1]:
                        ending_width = lr_son.shape[1]


                    if ending_width-self.conf.crop_size < 0 and ending_hegiht - self.conf.crop_size < 0 :
                        patch = lr_son[0:self.conf.crop_size,0:self.conf.crop_size]
                    elif ending_width-self.conf.crop_size >= 0 and ending_hegiht - self.conf.crop_size < 0 :
                        patch = lr_son[0:self.conf.crop_size,ending_width-self.conf.crop_size:ending_width]
                    elif ending_width - self.conf.crop_size < 0 and ending_hegiht - self.conf.crop_size >= 0:
                        patch = lr_son[ending_hegiht-self.conf.crop_size:ending_hegiht,0:self.conf.crop_size]
                    else :
                        patch = lr_son[ending_hegiht-self.conf.crop_size:ending_hegiht,ending_width-self.conf.crop_size:ending_width]
                    #patch = lr_son[ending_hegiht - 128:ending_hegiht, ending_width - 128:ending_width]
                else :
                    patch = imresize(k, scale_factor=self.sf, kernel=self.conf.upscale_method)
                #patch = imresize(patch, scale_factor=1, kernel='cubic')
                hr_cousin = patch
                #print('x idx , y idx : %d , %d'%(rotation_x,rotation_y))
                #print('width_idx , height_idx : %d , %d'%(width_idx, height_idx))
                #plt.imshow(k)
                #plt.show()
                #plt.imshow(patch)
                #plt.show()
                interpolated_lr_son = imresize(k, self.sf, (self.conf.crop_size, self.conf.crop_size), self.conf.upscale_method)

                test_input = interpolated_lr_son
                test_reference = hr_cousin.copy()
                test_input = torch.Tensor(test_input).permute(2, 0, 1).unsqueeze_(0)
                test_input = test_input.cuda()
                test_reference = torch.Tensor(test_reference).permute(2, 0, 1).unsqueeze_(0)
                test_reference = test_reference.cuda()
                xxx = np.clip(np.squeeze(self.model(test_input, test_reference).cpu().detach().permute(0, 2, 3, 1).numpy()), 0, 1)
                #plt.imshow(xxx)
                #plt.show()
                divide[int(self.conf.scale_factors[0][1] * yyyyy):int(self.conf.scale_factors[0][1] * yyyyy) + self.conf.crop_size, int(self.conf.scale_factors[0][0] * xxxxx):int(self.conf.scale_factors[0][0] * xxxxx) + self.conf.crop_size] += ones
                output[int(self.conf.scale_factors[0][1] * yyyyy):int(self.conf.scale_factors[0][1] * yyyyy) + self.conf.crop_size, int(self.conf.scale_factors[0][0] * xxxxx):int(self.conf.scale_factors[0][0] * xxxxx) + self.conf.crop_size] += xxx
                ###
        #print(divide)
        output = np.divide(output, divide)
        #plt.imshow(output)
        #plt.show()

        return output

    ##################################################끝부분 처
    def forwafsdfsafsfs(self, lr_son, hr_father_shape=None, rotK=0):  # k 값을 받아서 k값에 따른 각도와 플립 계산 후 인덱스도 변경.
        [M, N, _] = lr_son.shape
        output = np.zeros((int(2*M),int(2*N),3))
        divide = np.zeros((int(2*M),int(2*N),3))
        ones = np.ones((128,128,3))
        #hr_cousin = torch.Tensor(hr_cousin).permute(2, 0, 1).unsqueeze_(0)
        #hr_cousin = hr_cousin.cuda()
        for i in range(0,M,60):
            for j in range(0, N, 60):
                if i + 64 > M and j + 64 > N:
                    k = lr_son[M-64:M, N-64:N]
                    yyyyy = M-64
                    xxxxx = N-64
                elif i + 64 > M:
                    k = lr_son[M-64:M,j:j+64]
                    yyyyy = M-64
                    xxxxx = j
                elif j + 64 > N :
                    k = lr_son[i:i+64, N-64:N]
                    yyyyy = i
                    xxxxx = N-64
                else :
                    k = lr_son[i:i+64, j:j+64]
                    yyyyy = i
                    xxxxx = j
                ################
                if rotK%8 == 0:
                    shift_y = yyyyy#i
                    shift_x = xxxxx#j
                elif rotK%8 == 1:
                    shift_y = xxxxx
                    shift_x = self.www-(yyyyy+64)
                elif rotK%8 == 2:
                    shift_y = self.hhh-(yyyyy+64)
                    shift_x = self.www-(xxxxx+64)
                elif rotK%8 == 3:
                    shift_y = self.hhh-(xxxxx+64)
                    shift_x = yyyyy
                elif rotK%8 == 4:
                    shift_y = yyyyy
                    shift_x = self.www-(xxxxx+64)
                elif rotK%8 == 5:
                    shift_y = self.hhh-(xxxxx+64)
                    shift_x = self.www-(yyyyy+64)
                elif rotK%8 == 6:
                    shift_y = self.hhh-(yyyyy+64)
                    shift_x = xxxxx
                elif rotK%8 == 7:
                    shift_y = xxxxx
                    shift_x = yyyyy

                rot0_x, rot0_y = self.js_PM.test_run(shift_x, shift_y)
                #double_1d_idx, rot0_x,rot0_y , _, _ = self.js_PM.run(shift_x, shift_y)
                #####
                if rotK%8 == 0:
                    height_idx = rot0_y
                    width_idx = rot0_x
                elif rotK%8 == 1:
                    height_idx = self.www-rot0_x - 128
                    width_idx = rot0_y
                elif rotK%8 == 2:
                    height_idx = self.hhh-rot0_y - 128
                    width_idx = self.www-rot0_x - 128
                elif rotK%8 == 3:
                    height_idx = rot0_x
                    width_idx = self.hhh-rot0_y -128
                elif rotK%8 == 4:
                    height_idx = rot0_y
                    width_idx = self.www-rot0_x - 128
                elif rotK%8 == 5:
                    height_idx = self.www-rot0_x - 128
                    width_idx = self.hhh-rot0_y - 128
                elif rotK%8 == 6:
                    height_idx = self.hhh-rot0_y - 128
                    width_idx = rot0_x
                elif rotK%8 == 7:
                    height_idx = rot0_x
                    width_idx = rot0_y
                #print(height_idx,width_idx)
                plt.imshow(k)
                plt.show()
                ending_hegiht = height_idx + 128
                ending_width = width_idx + 128
                if height_idx + 128 > lr_son.shape[0]:
                    ending_hegiht = lr_son.shape[0]
                if width_idx + 128 > lr_son.shape[1]:
                    ending_width = lr_son.shape[1]
                patch = lr_son[ending_hegiht-128:ending_hegiht, ending_width-128:ending_width]
                hr_cousin = patch
                #print(ending_hegiht, ending_width)
                #print(hr_cousin.shape)
                plt.imshow(hr_cousin)
                plt.show()

                hr_cousin = torch.Tensor(hr_cousin).permute(2,0,1).unsqueeze_(0)
                hr_cousin = hr_cousin.cuda()

                #################
                interpolated_lr_son = imresize(k, self.sf, (128,128), self.conf.upscale_method)
                if self.Y == True:
                    interpolated_lr_son = (torch.Tensor(interpolated_lr_son)).unsqueeze_(0).unsqueeze_(0)
                else:
                    interpolated_lr_son = (torch.Tensor(interpolated_lr_son).permute(2, 0, 1)).unsqueeze_(0)
                if self.cuda:
                    interpolated_lr_son = interpolated_lr_son.cuda()
                xxx = np.clip(np.squeeze(self.model(interpolated_lr_son, hr_cousin).cpu().detach().permute(0, 2, 3, 1).numpy()),0, 1)
                output[(2*yyyyy):(2*yyyyy)+128,(2*xxxxx):(2*xxxxx)+128] += xxx
                divide[(2*yyyyy):(2*yyyyy)+128,(2*xxxxx):(2*xxxxx)+128] += ones
                ###
        output = np.divide(output,divide)
        #print(divide)
        #plt.imshow(output)
        #plt.show()

        return output
######################################################ㄹ


    def forward_pass_sliding_GT_TEST(self, lr_son, hr_father_shape=None, GT_IMG=None, rotK=0):  # k 값을 받아서 k값에 따른 각도와 플립 계산 후 인덱스도 변경.
        [M, N, _] = lr_son.shape
        output = np.zeros((int(2*M),int(2*N),3))
        divide = np.zeros((int(2*M),int(2*N),3))
        ones = np.ones((128,128,3))
        #hr_cousin = torch.Tensor(hr_cousin).permute(2, 0, 1).unsqueeze_(0)
        #hr_cousin = hr_cousin.cuda()
        for i in range(0,M,8):
            for j in range(0, N, 8):
                if i + 64 > M or j + 64 > N:
                    continue
                else :
                    k = lr_son[i:i+64, j:j+64]
                    patch = GT_IMG[i:i+128, j:j+128]
                hr_cousin = patch.copy()
                hr_cousin = torch.Tensor(hr_cousin).permute(2,0,1).unsqueeze_(0)
                hr_cousin = hr_cousin.cuda()

                #################
                interpolated_lr_son = imresize(k, self.sf, (128,128), self.conf.upscale_method)
                if self.Y == True:
                    interpolated_lr_son = (torch.Tensor(interpolated_lr_son)).unsqueeze_(0).unsqueeze_(0)
                else:
                    interpolated_lr_son = (torch.Tensor(interpolated_lr_son).permute(2, 0, 1)).unsqueeze_(0)
                if self.cuda:
                    interpolated_lr_son = interpolated_lr_son.cuda()
                xxx = np.clip(np.squeeze(self.model(interpolated_lr_son, hr_cousin).cpu().detach().permute(0, 2, 3, 1).numpy()),0, 1)
                output[(2*i):(2*i)+128,(2*j):(2*j)+128] += xxx
                divide[(2*i):(2*i)+128,(2*j):(2*j)+128] += ones
                ###
        output = np.divide(output,divide)
        #print(divide)
        #plt.imshow(output)
        #plt.show()

        return output

    def learning_rate_policy(self):
        # fit linear curve and check slope to determine whether to do nothing, reduce learning rate or finish
        if (not (1 + self.iter) % self.conf.learning_rate_policy_check_every
                and self.iter - self.learning_rate_change_iter_nums[-1] > self.conf.min_iters):
            # noinspection PyTupleAssignmentBalance
            # print(self.conf.run_test_every)
            [slope, _], [[var, _], _] = np.polyfit(self.mse_steps[-(self.conf.learning_rate_slope_range //
                                                                    self.conf.run_test_every):],
                                                   self.mse_rec[-(self.conf.learning_rate_slope_range //
                                                                  self.conf.run_test_every):],
                                                   1, cov=True)

            # We take the the standard deviation as a measure
            std = np.sqrt(var)

            # Verbose
            print('slope: ', slope, 'STD: ', std)

            # Determine learning rate maintaining or reduction by the ration between slope and noise
            if -self.conf.learning_rate_change_ratio * slope < std:
                self.learning_rate /= 10
                print("learning rate updated: ", self.learning_rate)

                # Keep track of learning rate changes for plotting purposes
                self.learning_rate_change_iter_nums.append(self.iter)

    def quick_test(self):
        # There are four evaluations needed to be calculated:

        # 1. True MSE (only if ground-truth was given), note: this error is before post-processing.
        # Run net on the input to get the output super-resolution (almost final result, only post-processing needed)
        #if self.gt_per_sf != None:
        #    self.sr = self.forward_pass(self.input)
        #    self.mse = (self.mse + [np.mean(np.ndarray.flatten(np.square(self.gt_per_sf - self.sr)))]
        #                if self.gt_per_sf is not None else None)

        # 2. Reconstruction MSE, run for reconstruction- try to reconstruct the input from a downscaled version of it
        self.reconstruct_output = self.forward_pass(self.father_to_son(self.fake_input), self.fake_input.shape)
        self.mse_rec.append(np.mean(np.ndarray.flatten(np.square(self.fake_input - self.reconstruct_output))))

        # 3. True MSE of simple interpolation for reference (only if ground-truth was given)
        #if self.gt_per_sf != None:
        #    interp_sr = imresize(self.input, self.sf, self.output_shape, self.conf.upscale_method)
        #    self.interp_mse = (self.interp_mse + [np.mean(np.ndarray.flatten(np.square(self.gt_per_sf - interp_sr)))]
        #                       if self.gt_per_sf is not None else None)

        # 4. Reconstruction MSE of simple interpolation over downscaled input
        interp_rec = imresize(self.father_to_son(self.fake_input), self.sf, self.fake_input.shape[0:2], self.conf.upscale_method)
        self.interp_rec_mse.append(np.mean(np.ndarray.flatten(np.square(self.fake_input - interp_rec))))

        # Track the iters in which tests are made for the graphics x axis
        self.mse_steps.append(self.iter)

        # Display test results if indicated
        if self.conf.display_test_results:
            print('iteration: ', self.iter, 'reconstruct mse:', self.mse_rec[-1], ', true mse:', (self.mse[-1]
                                                                                                  if self.mse else None))

        # plot losses if needed
        if self.conf.plot_losses:
            self.plot()

    def quick_test_patch(self, patch):
        # There are four evaluations needed to be calculated:

        # 1. True MSE (only if ground-truth was given), note: this error is before post-processing.
        # Run net on the input to get the output super-resolution (almost final result, only post-processing needed)
        #if self.gt_per_sf != None:
        #    self.sr = self.forward_pass(self.input)
        #    self.mse = (self.mse + [np.mean(np.ndarray.flatten(np.square(self.gt_per_sf - self.sr)))]
        #                if self.gt_per_sf is not None else None)

        # 2. Reconstruction MSE, run for reconstruction- try to reconstruct the input from a downscaled version of it
        self.reconstruct_output = self.forward_pass_ver0517(self.father_to_son(self.fake_input), self.fake_input.shape,final_test_switch=False,rotation_Var=0)
        self.mse_rec.append(np.mean(np.ndarray.flatten(np.square(self.fake_input - self.reconstruct_output))))

        # 3. True MSE of simple interpolation for reference (only if ground-truth was given)
        #if self.gt_per_sf != None:
        #    interp_sr = imresize(self.input, self.sf, self.output_shape, self.conf.upscale_method)
        #    self.interp_mse = (self.interp_mse + [np.mean(np.ndarray.flatten(np.square(self.gt_per_sf - interp_sr)))]
        #                       if self.gt_per_sf is not None else None)

        # 4. Reconstruction MSE of simple interpolation over downscaled input
        interp_rec = imresize(self.father_to_son(self.fake_input), self.sf, self.fake_input.shape[0:2], self.conf.upscale_method)
        self.interp_rec_mse.append(np.mean(np.ndarray.flatten(np.square(self.fake_input - interp_rec))))

        # Track the iters in which tests are made for the graphics x axis
        self.mse_steps.append(self.iter)

        # Display test results if indicated
        if self.conf.display_test_results:
            print('iteration: ', self.iter, 'reconstruct mse:', self.mse_rec[-1], ', true mse:', (self.mse[-1]
                                                                                                  if self.mse else None))

        # plot losses if needed
        if self.conf.plot_losses:
            self.plot()

    def train(self):
        # def loss and optimizer
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # main training loop
        for self.iter in range(self.conf.max_iters):
            # Use augmentation from original input image to create current father.
            # If other scale factors were applied before, their result is also used (hr_fathers_in)
            self.hr_father, self.shift, shift_back_from_center, shear_mat, rotation_mat, scale_mat, shift_to_center_mat, theta = random_augment(ims=self.hr_fathers_sources,
                                            base_scales=[1.0] + self.conf.scale_factors,
                                            leave_as_is_probability=self.conf.augment_leave_as_is_probability,
                                            no_interpolate_probability=self.conf.augment_no_interpolate_probability,
                                            min_scale=self.conf.augment_min_scale,
                                            max_scale=([1.0] + self.conf.scale_factors)[
                                                len(self.hr_fathers_sources) - 1],
                                            allow_rotation=self.conf.augment_allow_rotation,
                                            scale_diff_sigma=self.conf.augment_scale_diff_sigma,
                                            shear_sigma=self.conf.augment_shear_sigma,
                                            crop_size=self.conf.crop_size)
            #print(self.shift)
            if self.Reference == True:
                double_1d_idx,_,_, width_idx, height_idx =self.js_PM.run(self.shift[0],self.shift[1])
                #double_1d_idx,width_idx,height_idx, _, _ =self.js_PM.run(self.shift[0],self.shift[1])
                shift_mat_mine = np.array([[1, 0, - width_idx],
                          [0, 1, - height_idx],
                          [0, 0, 1]])
                if theta == 3 * pi / 2 or theta == 1 * pi / 2:
                    dst = np.clip(warpPerspective(self.hr_fathers_sources[0],
                                                    shift_back_from_center.dot(shift_mat_mine).dot(shear_mat).dot(scale_mat).dot(shift_to_center_mat),
                                                    (self.conf.crop_size, self.conf.crop_size), flags=INTER_CUBIC), 0,1)
                    gak = theta * 180 / pi
                    rot_mat = cv2.getRotationMatrix2D((int(self.conf.crop_size / 2), int(self.conf.crop_size / 2)), gak, 1)
                    patch = cv2.warpAffine(dst, rot_mat, (self.conf.crop_size, self.conf.crop_size))
                else :
                    patch  = np.clip(warpPerspective(self.hr_fathers_sources[0],shift_back_from_center.dot(shift_mat_mine).dot(shear_mat).dot(rotation_mat).dot(scale_mat).dot(shift_to_center_mat),
                                                    (self.conf.crop_size, self.conf.crop_size), flags=INTER_CUBIC),0,1)
            else :
                patch = None

           #print('******************************************', self.shift)
            if self.iter % 1000 == 0:
                plt.imshow(self.hr_father)
                plt.show()
                plt.imshow(patch)
                plt.show()
                plt.imsave("%05d_hr_father.png"%(iter),self.hr_father)
                plt.imsave("%05d_cousin.png"%(iter),patch)
            #print('***************************************************************************************')

            # Get lr-son from hr-father
            self.lr_son = self.father_to_son(self.hr_father)
            # should convert input and output to torch tensor
            #pixels = patch.getdata()
            #if cv2.countNonZero(patch) < 128*64 :
            #    patch = cv2.resize(self.lr_son, (128,128), cv2.INTER_CUBIC)
            # run network forward and back propagation, one iteration (This is the heart of the training)
            if self.Reference == True:
                self.train_output = self.forward_backward_pass_patch(self.lr_son, self.hr_father, criterion, optimizer, patch)
            else:
                self.train_output = self.forward_backward_pass(self.lr_son, self.hr_father, criterion, optimizer)

            # Display info and save weights
            if not self.iter % self.conf.display_every:
                print('sf:', self.sf * self.base_sf, ', iteration: ', self.iter, ', loss:  %.7f'% self.loss[self.iter])

            # Test network
            if self.conf.run_test and (not self.iter % self.conf.run_test_every):
                if self.Reference == True:
                    self.quick_test_patch(patch)
                else:
                    self.quick_test()
            # Consider changing learning rate or stop according to iteration number and losses slope
            self.learning_rate_policy()

            # stop when minimum learning rate was passed
            if self.learning_rate < self.conf.min_learning_rate:
                break


    def train_MINE(self):
        # def loss and optimizer
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # main training loop
        for self.iter in range(self.conf.max_iters):
            # Use augmentation from original input image to create current father.
            # If other scale factors were applied before, their result is also used (hr_fathers_in)
            self.hr_father, self.shift, kkk, scale = My_random_augment(img=self.hr_fathers_sources[0],crop_size=self.conf.crop_size,leave_as_is_probability=1)
            #print(self.shift)
            if self.Reference == True:
                width_idx, height_idx, HARD_SWITCH =self.js_PM.train_run(int(self.shift[0]),int(self.shift[1]))
                #double_1d_idx,width_idx,height_idx, _, _ =self.js_PM.run(self.shift[0],self.shift[1])
                #source = cv2.resize(self.hr_fathers_sources[0],(int(self.hr_fathers_sources[0].shape[1]*0.5*scale),int(self.hr_fathers_sources[0].shape[0]*0.5*scale)),cv2.INTER_CUBIC)  # 0.5는 scale factor
                if HARD_SWITCH == False:
                    source = imresize(self.hr_fathers_sources[0], (1/np.mean(self.conf.scale_factors[0]))*scale, kernel=self.kernel)
                    ending_hegiht = int(height_idx+int(self.conf.crop_size/2))
                    ending_width = int(width_idx+int(self.conf.crop_size/2))
                    #print(ending_width,ending_hegiht)
                    if height_idx+int(self.conf.crop_size/2) > source.shape[0] :
                        ending_hegiht = source.shape[0]
                    if width_idx + int(self.conf.crop_size/2) > source.shape[1] :
                        ending_width = source.shape[1]
                    if ending_width-self.conf.crop_size < 0 and ending_hegiht - self.conf.crop_size < 0 :
                        crop_patch = source[0:self.conf.crop_size,0:self.conf.crop_size]
                    elif ending_width-self.conf.crop_size >= 0 and ending_hegiht - self.conf.crop_size < 0 :
                        crop_patch = source[0:self.conf.crop_size,ending_width-self.conf.crop_size:ending_width]
                    elif ending_width - self.conf.crop_size < 0 and ending_hegiht - self.conf.crop_size >= 0:
                        crop_patch = source[ending_hegiht-self.conf.crop_size:ending_hegiht,0:self.conf.crop_size]
                    else :
                        crop_patch = source[ending_hegiht-self.conf.crop_size:ending_hegiht,ending_width-self.conf.crop_size:ending_width]

                    if kkk == 0 :
                        patch = crop_patch
                    else:
                        patch = np.rot90(crop_patch, kkk) if kkk < 4 else np.fliplr(np.rot90(crop_patch,kkk))
                else :
                    #print('hard_th******************************************')
                    patch =  imresize(self.hr_father, self.sf , kernel=self.conf.upscale_method)
                #patch = Add_jpg_compression(img=patch)
            else :
                patch = None

           #print('******************************************', self.shift)
            #print('***************************************************************************************')
            #print(patch)
            # Get lr-son from hr-father
            self.lr_son = self.father_to_son(self.hr_father)
            #if self.iter % 200 == 0:
            #print('shift idx: %d, %d'%(self.shift[0],self.shift[1]))
            #print('height idx , width idx : %d , %d'%(height_idx, width_idx))
            #plt.imshow(self.hr_father)
            #plt.show()
            #plt.imshow(patch)
            #plt.show()
            #    plt.imsave("%s/Patches/%s_%05d_hr_father.png"%(self.conf.result_path, os.path.basename(self.file_name)[:-4],self.iter),self.hr_father,vmin=0, vmax=1)
            #    plt.imsave("%s/Patches/%s_%05d_cousin.png"%(self.conf.result_path, os.path.basename(self.file_name)[:-4],self.iter),patch,vmin=0, vmax=1)
            # should convert input and output to torch tensor
            #pixels = patch.getdata()
            #if cv2.countNonZero(patch) < 128*64 :
            #    patch = cv2.resize(self.lr_son, (128,128), cv2.INTER_CUBIC)
            # run network forward and back propagation, one iteration (This is the heart of the training)
            if self.Reference == True:
                self.train_output = self.forward_backward_pass_patch(self.lr_son, self.hr_father, criterion, optimizer, patch)
            else:
                self.train_output = self.forward_backward_pass(self.lr_son, self.hr_father, criterion, optimizer)

            # Display info and save weights
            if not self.iter % self.conf.display_every:
                print('sf:', self.sf * self.base_sf, ', iteration: ', self.iter, ', loss:  %.7f'% self.loss[self.iter])

            # Test network
            if self.conf.run_test and (not self.iter % self.conf.run_test_every):
                if self.Reference == True:
                    self.quick_test_patch(patch)
                else:
                    self.quick_test()
            # Consider changing learning rate or stop according to iteration number and losses slope
            self.learning_rate_policy()

            # stop when minimum learning rate was passed
            if self.learning_rate < self.conf.min_learning_rate:
                break

            #if self.iter == 0 :
            if self.iter!=0 and self.iter % 3000== 0:
                post_processed_output = self.final_test()
                post_processed_output = post_processed_output[:self.conf.scale_factors[0][1]*self.hhh,:self.conf.scale_factors[0][0]*self.www,:]
                plt.imsave("%s/output/%s_%05d_final_test_result.png" % (self.conf.result_path, os.path.basename(self.file_name)[:-4],self.iter), post_processed_output, vmin=0, vmax=1)

    def father_to_son(self, hr_father):
        # Create son out of the father by downscaling and if indicated adding noise
        lr_son = imresize(hr_father, 1.0 / self.sf, kernel=self.kernel)
        rand_noise_patch = np.clip(lr_son + np.random.randn(*lr_son.shape) * self.conf.noise_std, 0, 1)
        real_lr = rand_noise_patch
        #real_lr = Add_jpg_compression(img=rand_noise_patch)
        return real_lr


    def final_test(self):
        # Run over 8 augmentations of input - 4 rotations and mirror (geometric self ensemble)
        outputs = []

        # The weird range means we only do it once if output_flip is disabled
        # We need to check if scale factor is symmetric to all dimensions, if not we will do 180 jumps rather than 90



        for k in range(0, 1 + 7 * self.conf.output_flip, 1 + int(self.sf[0] != self.sf[1])):
        #for k in range(0, 1, 1 + int(self.sf[0] != self.sf[1])):
            # Rotate 90*k degrees and mirror flip when k>=4
            test_input = np.rot90(self.input, k) if k < 4 else np.fliplr(np.rot90(self.input, k))
            #gt_img = np.rot90(self.gt_aaa, k) if k < 4 else np.fliplr(np.rot90(self.gt_aaa,k))
            # Apply network on the rotated input
            if self.Reference == True:
                #tmp_output = self.forward_pass_sliding_GT_TEST(test_input,rotK=k,GT_IMG=gt_img)
                tmp_output = self.forward_pass_ver0517(test_input,final_test_switch=True,rotation_Var=k)
            else :
                tmp_output = self.forward_pass(test_input)
            # Undo the rotation for the processed output (mind the opposite order of the flip and the rotation)
            tmp_output = np.rot90(tmp_output, -k) if k < 4 else np.rot90(np.fliplr(tmp_output), -k)

            # fix SR output with back projection technique for each augmentation
            for bp_iter in range(self.conf.back_projection_iters[self.sf_ind]):
                tmp_output = back_projection(tmp_output, self.input, down_kernel=self.kernel,
                                             up_kernel=self.conf.upscale_method, sf=self.sf)

            # save outputs from all augmentations
            outputs.append(tmp_output)

        # Take the median over all 8 outputs
        almost_final_sr = np.median(outputs, 0)

        # Again back projection for the final fused result
        for bp_iter in range(self.conf.back_projection_iters[self.sf_ind]):
            almost_final_sr = back_projection(almost_final_sr, self.input, down_kernel=self.kernel,
                                              up_kernel=self.conf.upscale_method, sf=self.sf)

        # Now we can keep the final result (in grayscale case, colors still need to be added, but we don't care
        # because it is done before saving and for every other purpose we use this result)
        self.final_sr = almost_final_sr

        # Add colors to result image in case net was activated only on grayscale
        return self.final_sr

    def final_test_MINE(self):

        test_input = self.fake_input

        almost_final_sr = self.forward_pass_ver21(test_input,final_test_switch=True)
        # Again back projection for the final fused result
        for bp_iter in range(self.conf.back_projection_iters[self.sf_ind]):
            almost_final_sr = back_projection(almost_final_sr, self.fake_input, down_kernel=self.kernel,
                                              up_kernel=self.conf.upscale_method, sf=self.sf)

        # Now we can keep the final result (in grayscale case, colors still need to be added, but we don't care
        # because it is done before saving and for every other purpose we use this result)
        self.final_sr = almost_final_sr

        # Add colors to result image in case net was activated only on grayscale
        return self.final_sr

    def base_change(self):
        # If there is no base scale large than the current one get out of here
        if len(self.conf.base_change_sfs) < self.base_ind + 1:
            return

        # Change base input image if required (this means current output becomes the new input)
        if abs(self.conf.scale_factors[self.sf_ind] - self.conf.base_change_sfs[self.base_ind]) < 0.001:
            if len(self.conf.base_change_sfs) > self.base_ind:
                # The new input is the current output
                self.input = self.final_sr

                # The new base scale_factor
                self.base_sf = self.conf.base_change_sfs[self.base_ind]

                # Keeping track- this is the index inside the base scales list (provided in the config)
                self.base_ind += 1

            print('base changed to %.2f' % self.base_sf)

    def plot(self):
        plots_data, labels = zip(*[(np.array(x), l) for (x, l)
                                   in zip([self.mse, self.mse_rec, self.interp_mse, self.interp_rec_mse],
                                          ['True MSE', 'Reconstruct MSE', 'Bicubic to ground truth MSE',
                                           'Bicubic to reconstruct MSE']) if x is not None])

        # For the first iteration create the figure
        if not self.iter:
            # Create figure and split it using GridSpec. Name each region as needed
            self.fig = plt.figure(figsize=(9.5, 9))
            grid = GridSpec(4, 4)
            self.loss_plot_space = plt.subplot(grid[:-1, :])
            self.lr_son_image_space = plt.subplot(grid[3, 0])
            self.hr_father_image_space = plt.subplot(grid[3, 3])
            self.out_image_space = plt.subplot(grid[3, 1])

            # Activate interactive mode for live plot updating
            plt.ion()

            # Set some parameters for the plots
            self.loss_plot_space.set_xlabel('step')
            self.loss_plot_space.set_ylabel('MSE')
            self.loss_plot_space.grid(True)
            self.loss_plot_space.set_yscale('log')
            self.loss_plot_space.legend()
            self.plots = [None] * 4

            # loop over all needed plot types. if some data is none than skip, if some data is one value tile it
            self.plots = self.loss_plot_space.plot(*[[0]] * 2 * len(plots_data))

        # Update plots
        for plot, plot_data in zip(self.plots, plots_data):
            plot.set_data(self.mse_steps, plot_data)

            self.loss_plot_space.set_xlim([0, self.iter + 1])
            all_losses = np.array(plots_data)
            self.loss_plot_space.set_ylim([np.min(all_losses) * 0.9, np.max(all_losses) * 1.1])

        # Mark learning rate changes
        for iter_num in self.learning_rate_change_iter_nums:
            self.loss_plot_space.axvline(iter_num)

        # Add legend to graphics
        self.loss_plot_space.legend(labels)

        # Show current input and output images
        self.lr_son_image_space.imshow(self.lr_son, vmin=0.0, vmax=1.0)
        self.out_image_space.imshow(self.train_output, vmin=0.0, vmax=1.0)
        self.hr_father_image_space.imshow(self.hr_father, vmin=0.0, vmax=1.0)

        # These line are needed in order to see the graphics at real time
        self.fig.canvas.draw()
        plt.pause(0.01)
