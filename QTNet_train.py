# -*- coding: UTF-8 -*-
# PyTorch lib
import argparse
import math
import os

import cv2
# Tools lib
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision


# Tools lib
# import cv2
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.det_conv0 = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.det_conv_mask = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.diconv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2),
            nn.ReLU()
        )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, dilation=4),
            nn.ReLU()
        )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, dilation=8),
            nn.ReLU()
        )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 16, dilation=16),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.outframe1 = nn.Sequential(
            nn.Conv2d(256, 3, 3, 1, 1),
            nn.ReLU()
        )
        self.outframe2 = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, input,device):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        mask = Variable(torch.ones(batch_size, 1, row, col)).to(device) / 2.
        h = Variable(torch.zeros(batch_size, 32, row, col)).to(device)
        c = Variable(torch.zeros(batch_size, 32, row, col)).to(device)
        mask_list = []
        attention_map = []
        
        x = self.conv1(input)
        res1 = x
        x = self.conv2(x)
        x = self.conv3(x)
        res2 = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.conv7(x)
        x = self.conv8(x)
        frame1 = self.outframe1(x)
        x = self.deconv1(x)
        if x.shape != res2.shape:
            print('!ok')
        x = x + res2
        x = self.conv9(x)
        frame2 = self.outframe2(x)
        x = self.deconv2(x)
        x = x + res1
        x = self.conv10(x)
        x = self.output(x)
        # x = input + x
        return mask_list, frame1, frame2, attention_map, x

def prepare_img_to_tensor(image,mean=(0.406, 0.456, 0.485),std=(0.225,0.224, 0.229)):
    image = np.array(image, dtype='float32') / 255.
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    image = image - mean
    image = image /std
    image = image[:, :, (2, 1, 0)]
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = image.to(device)
    return image


def trainable(net, trainable):
    for para in net.parameters():
        para.requires_grad = trainable


# Initialize VGG16 with pretrained weight on ImageNet
def vgg_init(device, model_weights):
    vgg_model = torchvision.models.vgg16()
    vgg_model.load_state_dict(torch.load(model_weights))
    vgg_model.to(device)
    # vgg_model = vgg_model.classifier[:-1]
    vgg_model.eval()
    trainable(vgg_model, False)
    return vgg_model


# Extract features from internal layers for perceptual loss
class Vgg(nn.Module):
    def __init__(self, vgg_model):
        super(Vgg, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '1': "relu1_1",
            '3': "relu1_2",
            '6': "relu2_1",
            '8': "relu2_2"
        }

    def forward(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def resize_image(image, scale_coefficient):

    # calculate the 50 percent of original dimensions
    width = int(image.shape[1] * scale_coefficient)
    height = int(image.shape[0] * scale_coefficient)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(image, dsize)
    return output

def loss_generator(generator_results, back_ground_truth):

    mseloss = nn.MSELoss()

    _s = [generator_results[1], generator_results[2], generator_results[4]]
    _t = [prepare_img_to_tensor(resize_image(back_ground_truth, 0.25)),
          prepare_img_to_tensor(resize_image(back_ground_truth, 0.5)), prepare_img_to_tensor(back_ground_truth)]
    _lamda = lamda_in_autoencoder
    lm_s_t = 0
    for i in range(len(_s)):
        lm_s_t += _lamda[i] * mseloss(_s[i], _t[i])
    lm_s_t = torch.mean(lm_s_t)

    lp_o_t = 0
    # loss2 = nn.MSELoss()
    vgg_to_gen = vgg16(generator_results[4])
    vgg_to_gt = vgg16(prepare_img_to_tensor(back_ground_truth))
    for i in range(len(vgg_to_gen)):
        lp_o_t += mseloss(vgg_to_gen[i], vgg_to_gt[i])
    lp_o_t = torch.mean(lp_o_t)

    # LGAN(O) = log(1 - D(G(I)))
    l_g =  lm_s_t + lp_o_t
    return l_g



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="normal_to_adverse", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    args = parser.parse_args()
    return args


def train():
    index = 0
    input_list = sorted(os.listdir(args.input_dir))
    # input_list = os.listdir(args.input_dir)
    gt_list = sorted(os.listdir(args.gt_dir))
    # gt_list = os.listdir(args.gt_dir)
    generator.apply(weights_init)
    for _e in range(previous_epoch + 1, epoch):
        print("======finish  ", _e ,' / ', epoch, "==========")

        for _i in range(len(input_list)):  
                img = cv2.imread(args.input_dir + input_list[_i])
                gt = cv2.imread(args.gt_dir + gt_list[_i])
                dsize = (416, 416)
                img = cv2.resize(img, dsize)
                gt = cv2.resize(gt, dsize)
                img_tensor = prepare_img_to_tensor(img)
                result = generator(img_tensor, device)
                loss1 = loss_generator(result, gt)
                optimizer_g.zero_grad()
                # Backpropagation
                loss1.backward()
                optimizer_g.step()
                torch.save(generator.state_dict(), os.path.join(args.save_weight, 
                        '_' + str(_e) + '.pth')
                        )

if __name__ == '__main__':
    args = get_args()
    if args.mode == "normal_to_adverse":
        # args.input_dir = './dataset/Normal_to_Foggy/images/Normal_train/'  # normal image 2975
        # args.gt_dir = './dataset/Normal_to_Foggy/images/Foggy_train/' # adverse image 2975
        args.save_weight = './runs/QTNet_weights/normal_to_foggy/'
    elif  args.mode == "adverse_to_normal":
        # args.input_dir = './dataset/Normal_to_Foggy/images/Foggy_train/'  # 
        # args.gt_dir = './dataset/Normal_to_Foggy/images/Normal_train/' # 
        args.save_weight = './runs/QTNet_weights/foggy_to_normal/'
    args.demo_img = './demo/output_foggy_drop_res/'
    path_to_save = os.path.join(args.save_weight)
    os.makedirs(path_to_save, exist_ok=True)


    model_weights = './runs/vgg16_caffe.pth'
    previous_epoch = 0
    epoch = 50 
    learning_rate = 0.0002
    mean=(0.406, 0.456, 0.485)

    
    std=(0.225,0.224, 0.229)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = Generator().to(device)
    vgg16 = Vgg(vgg_init(device, model_weights))
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    lamda_in_autoencoder = [0.01, 0.01, 0.01]
    train()